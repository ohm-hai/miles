import hashlib
import json
import pickle
import shlex
from types import SimpleNamespace

import pytest

from tests.fast.launch_scripts.py_harness import (
    REPO_ROOT,
    call_entrypoint,
    freeze_environment,
    import_launch_script,
    install_command_recorder,
)


_LAUNCHER = REPO_ROOT / "scripts/amd/run_glm5_2_744b_a40b.py"

_AUDITED_CHECKPOINT_INDEX_LAYOUTS = {
    "GLM-5.2": (59585, 282, 1506659919872),
    "GLM-5.2_5layer": (1618, 14, 45683868160),
}
_AUDITED_SOURCE_CONFIG_ASSETS = {
    "GLM-5.2": (3732, "185f93ee6d12548e16a847e279dc0c3c90b1524c970b0866b42fb545747d859a"),
    "GLM-5.2_5layer": (
        1690,
        "f426922f0ad4efaaa6cae6dd24a6f0c59b0df1a6a1265857c4e9a949a5fdb020",
    ),
}
_AUDITED_SOURCE_INDEX_ASSETS = {
    "GLM-5.2": (5408032, "5fd47a926aefce0f2c917f42523e5e0f3c87e23e389e767c3681536a62f5cf5e"),
    "GLM-5.2_5layer": (
        145189,
        "d898bb64c9258ed870a83d23fde8142cb71c0db5eba8dcacd9d2884b4c995ec9",
    ),
}
_AUDITED_RUNTIME_HF_ASSETS = {
    "chat_template.jinja": (
        5076,
        "172dc74a35e1752df75ecfb2b2cf9326d2852bb1379868ebeec9571654489679",
    ),
    "generation_config.json": (
        194,
        "ac76b43d8683d3b930126870fc8be73d8679308fe752fa1f381096d8354f6a55",
    ),
    "tokenizer.json": (
        20217442,
        "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d",
    ),
    "tokenizer_config.json": (
        761,
        "98b1271574f41abf89427ae2dda030d94dc9478f0edc5a8bd240db213c6fd5fc",
    ),
}
_TEST_RUNTIME_HF_PAYLOADS = {
    "chat_template.jinja": b"fixture-chat-template",
    "generation_config.json": b"fixture-generation-config",
    "tokenizer.json": b"fixture-tokenizer-json",
    "tokenizer_config.json": b"fixture-tokenizer-config",
}


def _valid_cluster_inventory():
    nodes = [
        {
            "Alive": True,
            "NodeID": f"node-{rank}",
            "NodeManagerAddress": f"10.0.0.{rank + 1}",
            "Resources": {"GPU": 8.0},
        }
        for rank in range(8)
    ]
    probes = [
        {
            "node_id": f"node-{rank}",
            "hostname": f"mi35x-{rank}",
            "gfx950_count": 8,
            "product_names": ["AMD Instinct MI355X"] * 8,
            "product_source": "sysfs",
            "error": None,
        }
        for rank in range(8)
    ]
    return nodes, probes, 64.0, "10.0.0.1"


@pytest.fixture
def launcher(monkeypatch):
    freeze_environment(monkeypatch, hardware="MI355X")
    monkeypatch.setenv("MILES_SCRIPT_EXTERNAL_RAY", "1")
    monkeypatch.setenv("RAY_ADDRESS", "http://10.0.0.1:8265")
    monkeypatch.setenv("MASTER_ADDR", "10.0.0.1")
    recording = install_command_recorder(monkeypatch)
    module = import_launch_script(_LAUNCHER)
    monkeypatch.setattr(module._cluster, "_collect_ray_cluster_inventory", _valid_cluster_inventory)
    # Keep checkpoint-validation fixtures small. A separate fresh-module test
    # below protects the production layouts from being relaxed accidentally.
    for model_name in _AUDITED_CHECKPOINT_INDEX_LAYOUTS:
        monkeypatch.setitem(module._CHECKPOINT_INDEX_LAYOUTS, model_name, (1, 1, 7))
    monkeypatch.setattr(
        module,
        "_RUNTIME_HF_ASSETS",
        {
            name: (len(payload), hashlib.sha256(payload).hexdigest())
            for name, payload in _TEST_RUNTIME_HF_PAYLOADS.items()
        },
    )
    monkeypatch.setattr(module, "_SOURCE_CONFIG_ASSETS", {})
    monkeypatch.setattr(module, "_SOURCE_INDEX_ASSETS", {})
    monkeypatch.setattr(
        module._cluster,
        "_visible_rocm_product_names",
        lambda: ("AMD Instinct MI355X",) * 8,
    )
    return module, recording


def _full_overrides(hardware: str = "MI355X") -> dict[str, object]:
    return {
        "hardware": hardware,
        "model_name": "GLM-5.2",
        "num_nodes": 8,
        "num_gpus_per_node": 8,
        "run_id": "unit-test",
    }


def _flag_value(args: str, flag: str) -> str:
    words = shlex.split(args)
    position = words.index(flag)
    return words[position + 1]


def _record_train(module, recording, tmp_path, overrides: dict[str, object]) -> str:
    call_entrypoint(module, "_execute_train", overrides, sandbox=tmp_path)
    assert recording.commands
    return recording.commands[-1]


def _write_valid_checkpoint(module, tmp_path, *, model_name: str = "GLM-5.2"):
    num_layers = 5 if model_name.endswith("_5layer") else 78
    checkpoint = tmp_path / model_name
    checkpoint.mkdir(parents=True)
    config = {
        **module._CRITICAL_CONFIG_VALUES,
        "num_hidden_layers": num_layers,
        "indexer_types": module._expected_indexer_types(num_layers),
        "mlp_layer_types": ["dense"] * 3 + ["sparse"] * (num_layers - 3),
    }
    config_payload = json.dumps(config).encode()
    (checkpoint / "config.json").write_bytes(config_payload)
    module._SOURCE_CONFIG_ASSETS[model_name] = (
        len(config_payload),
        hashlib.sha256(config_payload).hexdigest(),
    )
    shard_name = "model-00001-of-00001.safetensors"
    index_payload = json.dumps(
        {
            "metadata": {"total_size": 7},
            "weight_map": {"model.embed_tokens.weight": shard_name},
        }
    ).encode()
    (checkpoint / "model.safetensors.index.json").write_bytes(index_payload)
    module._SOURCE_INDEX_ASSETS[model_name] = (
        len(index_payload),
        hashlib.sha256(index_payload).hexdigest(),
    )
    (checkpoint / shard_name).write_bytes(b"fixture")
    for name, payload in _TEST_RUNTIME_HF_PAYLOADS.items():
        (checkpoint / name).write_bytes(payload)
    model_org = "Pinaster" if num_layers == 5 else "zai-org"
    module._write_artifact_manifest(
        checkpoint,
        module._source_manifest_values(model_org, model_name, module._MODEL_REVISIONS[model_name]),
    )
    return checkpoint, config


def _write_valid_fp8_checkpoint(module, tmp_path, *, model_name: str = "GLM-5.2"):
    source, config = _write_valid_checkpoint(module, tmp_path, model_name=model_name)
    checkpoint = tmp_path / f"{model_name}_fp8"
    checkpoint.mkdir()
    config["quantization_config"] = {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
    }
    (checkpoint / "config.json").write_text(json.dumps(config))
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 9},
                "weight_map": {
                    "model.embed_tokens.weight": "model-00001-of-00001.safetensors",
                    "model.layers.0.weight_scale_inv": "model-00001-of-00001.safetensors",
                },
            }
        )
    )
    (checkpoint / "model-00001-of-00001.safetensors").write_bytes(b"fp8-fixture")
    for name, payload in _TEST_RUNTIME_HF_PAYLOADS.items():
        (checkpoint / name).write_bytes(payload)
    source_manifest = module._source_manifest_values(
        "Pinaster" if model_name.endswith("_5layer") else "zai-org",
        model_name,
        module._MODEL_REVISIONS[model_name],
    )
    module._write_artifact_manifest(checkpoint, module._fp8_manifest_values(source_manifest))
    return source, checkpoint


def _write_valid_torch_dist_checkpoint(module, tmp_path, *, model_name: str = "GLM-5.2"):
    checkpoint = tmp_path / f"{model_name}_torch_dist"
    release = checkpoint / "release"
    release.mkdir(parents=True)
    (checkpoint / "latest_checkpointed_iteration.txt").write_text("release")
    (release / "metadata.json").write_text(
        json.dumps({"sharded_backend": "torch_dist", "sharded_backend_version": 1})
    )
    payload = b"torch-dist-fixture"
    (release / "__0_0.distcp").write_bytes(payload)
    with open(release / ".metadata", "wb") as metadata_file:
        pickle.dump(
            SimpleNamespace(
                storage_data={
                    "fixture": SimpleNamespace(
                        relative_path="__0_0.distcp",
                        offset=0,
                        length=len(payload),
                    )
                }
            ),
            metadata_file,
        )
    source_manifest = module._source_manifest_values(
        "Pinaster" if model_name.endswith("_5layer") else "zai-org",
        model_name,
        module._MODEL_REVISIONS[model_name],
    )
    module._write_artifact_manifest(
        checkpoint,
        module._torch_dist_manifest_values(model_name, source_manifest),
    )
    return checkpoint


def test_pinned_checkpoint_index_layouts_match_the_audited_revisions():
    module = import_launch_script(_LAUNCHER)

    assert module._CHECKPOINT_INDEX_LAYOUTS == _AUDITED_CHECKPOINT_INDEX_LAYOUTS


def test_runtime_hf_assets_match_the_audited_revisions():
    module = import_launch_script(_LAUNCHER)

    assert module._RUNTIME_HF_ASSETS == _AUDITED_RUNTIME_HF_ASSETS
    assert module._SOURCE_CONFIG_ASSETS == _AUDITED_SOURCE_CONFIG_ASSETS
    assert module._SOURCE_INDEX_ASSETS == _AUDITED_SOURCE_INDEX_ASSETS


@pytest.mark.parametrize("hardware", ["MI350X", "MI355X"])
def test_full_model_parallel_contract_is_the_guarded_8_by_8_shape(launcher, hardware):
    module, _ = launcher
    args = module.ScriptArgs(**_full_overrides(hardware))

    parallel = module._get_parallel_config(args)

    assert _flag_value(parallel, "--tensor-model-parallel-size") == "8"
    assert _flag_value(parallel, "--pipeline-model-parallel-size") == "4"
    assert _flag_value(parallel, "--context-parallel-size") == "1"
    assert _flag_value(parallel, "--expert-model-parallel-size") == "16"
    assert _flag_value(parallel, "--expert-tensor-parallel-size") == "1"
    assert _flag_value(parallel, "--decoder-first-pipeline-num-layers") == "18"
    assert _flag_value(parallel, "--decoder-last-pipeline-num-layers") == "20"
    assert "--sequence-parallel" in shlex.split(parallel)

    # With 78 layers the 18/20 edge split leaves two 20-layer middle stages.
    # All four stages begin on a DSA index-computing layer.
    assert tuple(module._pipeline_stage_starts(args)) == (1, 19, 39, 59)
    assert 64 // (8 * 4 * 1) == 2  # DP is derived by Megatron.


def test_five_layer_parallel_contract_keeps_the_manual_poc_four_gpu_shape(launcher):
    module, _ = launcher
    args = module.ScriptArgs(
        hardware="MI355X",
        model_name="GLM-5.2_5layer",
        num_nodes=1,
        num_gpus_per_node=4,
        run_id="unit-test",
    )

    parallel = module._get_parallel_config(args)

    assert _flag_value(parallel, "--tensor-model-parallel-size") == "4"
    assert _flag_value(parallel, "--pipeline-model-parallel-size") == "1"
    assert _flag_value(parallel, "--context-parallel-size") == "1"
    assert _flag_value(parallel, "--expert-model-parallel-size") == "4"
    assert _flag_value(parallel, "--expert-tensor-parallel-size") == "1"
    assert tuple(module._pipeline_stage_starts(args)) == (1,)
    assert args.enable_r3 is False
    assert args.freeze_router is False


@pytest.mark.parametrize("num_gpus", [1, 2, 8])
def test_five_layer_rejects_unvalidated_gpu_counts(launcher, num_gpus):
    module, _ = launcher

    with pytest.raises(NotImplementedError, match=r"five-layer-poc-h16.*1 node\(s\) x 4 GPUs"):
        module.ScriptArgs(
            hardware="MI355X",
            model_name="GLM-5.2_5layer",
            num_nodes=1,
            num_gpus_per_node=num_gpus,
            run_id="unit-test",
        )


def test_five_layer_h8_profile_requires_the_experimental_escape_hatch(launcher):
    module, _ = launcher

    with pytest.raises(NotImplementedError, match="five_layer_full_shape_h8"):
        module.ScriptArgs(
            hardware="MI355X",
            model_name="GLM-5.2_5layer",
            num_nodes=1,
            num_gpus_per_node=8,
            single_node_topology="full-shape-h8",
            run_id="unit-test",
        )


def test_five_layer_h8_profile_matches_the_full_local_head_shape(launcher):
    module, _ = launcher
    args = module.ScriptArgs(
        hardware="MI355X",
        model_name="GLM-5.2_5layer",
        num_nodes=1,
        num_gpus_per_node=8,
        single_node_topology="full-shape-h8",
        allow_unvalidated_features=True,
        run_id="unit-test",
    )

    parallel = module._get_parallel_config(args)
    sglang = module._sglang_args(args)

    assert _flag_value(parallel, "--tensor-model-parallel-size") == "8"
    assert _flag_value(parallel, "--pipeline-model-parallel-size") == "1"
    assert _flag_value(parallel, "--expert-model-parallel-size") == "8"
    assert _flag_value(sglang, "--sglang-tp-size") == "8"
    assert _flag_value(sglang, "--sglang-ep-size") == "8"
    assert tuple(module._pipeline_stage_starts(args)) == (1,)


def test_five_layer_rollout_only_is_a_launcher_controlled_stage(launcher):
    module, _ = launcher
    args = module.ScriptArgs(
        hardware="MI355X",
        model_name="GLM-5.2_5layer",
        num_nodes=1,
        num_gpus_per_node=4,
        rollout_only=True,
        run_id="unit-test",
    )

    misc_words = shlex.split(module._misc_args(args))
    assert "--debug-rollout-only" in misc_words
    assert "--rematerialize-param-from-master-weight" not in misc_words
    assert "--offload-train-target" not in misc_words
    assert "--stream-optimizer-state-to-disk" not in misc_words
    assert "--ref-load" not in shlex.split(module._checkpoint_args(args))
    assert args.skip_saving is True


def test_full_single_node_rollout_profile_is_explicit_and_guarded(launcher):
    module, _ = launcher
    overrides = {
        "hardware": "MI355X",
        "model_name": "GLM-5.2",
        "num_nodes": 1,
        "num_gpus_per_node": 8,
        "full_model_rollout_only": True,
        "run_id": "unit-test",
    }

    with pytest.raises(NotImplementedError, match="full_model_rollout_only"):
        module.ScriptArgs(**overrides)

    args = module.ScriptArgs(**(overrides | {"allow_unvalidated_features": True}))
    assert args._topology.rollout_only is True
    assert args.rollout_only is True
    assert args.skip_saving is True
    assert args.enable_r3 is False
    assert args.num_rollout == 1
    assert args.rollout_max_response_len == 128
    assert args.sglang_mem_fraction_static == 0.70
    assert args.sglang_max_running_requests == 8
    assert tuple(module._pipeline_stage_starts(args)) == (1,)


def test_full_model_rejects_the_five_layer_topology_selector(launcher):
    module, _ = launcher

    with pytest.raises(ValueError, match="single_node_topology applies only"):
        module.ScriptArgs(**(_full_overrides() | {"single_node_topology": "full-shape-h8"}))


def test_full_model_rollout_only_cannot_be_smuggled_through_extra_args(launcher):
    module, _ = launcher

    with pytest.raises(ValueError, match="launcher.*--rollout-only profile"):
        module.ScriptArgs(**(_full_overrides() | {"extra_args": "--debug-rollout-only"}))


def test_rollout_only_rejects_trainer_debug_replay(launcher):
    module, _ = launcher

    with pytest.raises(ValueError, match="cannot load trainer-only debug rollout data"):
        module.ScriptArgs(
            hardware="MI355X",
            model_name="GLM-5.2_5layer",
            num_nodes=1,
            num_gpus_per_node=4,
            rollout_only=True,
            extra_args="--load-debug-rollout-data /tmp/{rollout_id}.pt",
        )


def test_typed_rollout_only_preserves_the_full_8x8_gate_three_topology(launcher):
    module, _ = launcher
    args = module.ScriptArgs(**(_full_overrides() | {"rollout_only": True}))

    assert args._topology.name == "full-model-8x8"
    assert args.skip_saving is True
    assert args.num_rollout == 1
    assert args.rollout_max_response_len == 128
    assert args.enable_r3 is False
    assert "--debug-rollout-only" in shlex.split(module._misc_args(args))
    assert _flag_value(module._get_parallel_config(args), "--pipeline-model-parallel-size") == "4"
    assert _flag_value(module._rollout_args(args), "--rollout-batch-size") == "8"
    assert _flag_value(module._rollout_args(args), "--n-samples-per-prompt") == "1"
    assert _flag_value(module._rollout_args(args), "--global-batch-size") == "8"
    assert _flag_value(module._sglang_args(args), "--sglang-router-policy") == "round_robin"
    assert int(_flag_value(module._rollout_args(args), "--global-batch-size")) % 2 == 0  # DP2


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"num_rollout": 2}, "requires num_rollout=1"),
        ({"rollout_max_response_len": 129}, "single-node.*response length <=128"),
    ],
)
def test_full_single_node_rollout_probe_rejects_a_long_run(launcher, override, message):
    module, _ = launcher
    base = {
        "hardware": "MI355X",
        "model_name": "GLM-5.2",
        "num_nodes": 1,
        "num_gpus_per_node": 8,
        "full_model_rollout_only": True,
        "allow_unvalidated_features": True,
        "run_id": "unit-test",
    }

    with pytest.raises(ValueError, match=message):
        module.ScriptArgs(**(base | override))


def test_full_single_node_rollout_probe_rejects_all_free_form_overrides(launcher):
    module, _ = launcher

    with pytest.raises(ValueError, match="rejects extra_args"):
        module.ScriptArgs(
            hardware="MI355X",
            model_name="GLM-5.2",
            num_nodes=1,
            num_gpus_per_node=8,
            full_model_rollout_only=True,
            allow_unvalidated_features=True,
            extra_args="--hf-checkpoint /tmp/wrong",
            run_id="unit-test",
        )


@pytest.mark.parametrize(
    "extra_args",
    [
        "--model-name wrong",
        "--rollout-function-path bad.fn",
        "--disable-rollout-global-dataset",
        "--data-source-path /tmp/wrong",
        "--apply-chat-template-kwargs {}",
        "--chat-template-path /tmp/wrong",
        "--custom-rm-path bad.fn",
        "--custom-reward-post-process-path bad.fn",
        "--custom-convert-samples-to-train-data-path bad.fn",
        "--loss-type custom",
        "--custom-loss-function-path bad.fn",
        "--hf-checkpoint /tmp/wrong",
        "--hf-checkp=/tmp/wrong",
        "--tensor-model-parallel-size 1",
        "--debug-train-only",
    ],
)
def test_extra_args_rejects_every_nondiagnostic_option_without_abbreviations(launcher, extra_args):
    module, _ = launcher

    with pytest.raises(ValueError, match="only audited diagnostic options"):
        module.ScriptArgs(**(_full_overrides() | {"extra_args": extra_args}))


@pytest.mark.parametrize(
    "extra_args",
    [
        "--seed",
        "--seed --ci-test",
        "--ci-test=true",
        "--ci-test positional",
        "--seed 1234 --seed 5678",
    ],
)
def test_extra_args_rejects_malformed_diagnostic_options(launcher, extra_args):
    module, _ = launcher

    with pytest.raises(ValueError):
        module.ScriptArgs(**(_full_overrides() | {"extra_args": extra_args}))


def test_real_five_layer_stage_diagnostics_construct_script_args(launcher):
    module, _ = launcher
    base = {
        "hardware": "MI355X",
        "model_name": "GLM-5.2_5layer",
        "num_nodes": 1,
        "num_gpus_per_node": 4,
        "run_id": "unit-test",
    }

    rollout = module.ScriptArgs(
        **(
            base
            | {
                "rollout_only": True,
                "extra_args": (
                    "--rollout-seed 1234 --seed 1234 --ci-test "
                    "--sglang-enable-deterministic-inference --sglang-disable-cuda-graph "
                    "--save-debug-rollout-data /tmp/{rollout_id}.pt"
                ),
            }
        )
    )
    trainer = module.ScriptArgs(
        **(
            base
            | {
                "extra_args": (
                    "--rollout-seed 1234 --seed 1234 --ci-test --no-offload-train "
                    "--save-interval 1 --load-debug-rollout-data /tmp/{rollout_id}.pt"
                )
            }
        )
    )

    assert "--sglang-disable-cuda-graph" in shlex.split(rollout.extra_args)
    assert "--load-debug-rollout-data" in shlex.split(trainer.extra_args)
    assert "--debug-train-only" not in shlex.split(trainer.extra_args)
    assert "--rematerialize-param-from-master-weight" not in shlex.split(
        module._misc_args(trainer)
    )


def test_real_miles_parser_accepts_rollout_only_and_no_offload_commands(launcher):
    import argparse

    pytest.importorskip("sglang_router")
    pytest.importorskip("sglang")
    from miles.utils.arguments import (
        _validate_rematerialize_param_from_master_weight,
        get_miles_extra_args_provider,
    )

    module, _ = launcher
    base = {
        "hardware": "MI355X",
        "model_name": "GLM-5.2_5layer",
        "num_nodes": 1,
        "num_gpus_per_node": 4,
    }
    script_args = (
        module.ScriptArgs(**(base | {"rollout_only": True})),
        module.ScriptArgs(**(base | {"extra_args": "--no-offload-train"})),
    )

    for args in script_args:
        parser = argparse.ArgumentParser()
        get_miles_extra_args_provider()(parser)
        parsed, _ = parser.parse_known_args(
            shlex.split(module._misc_args(args) + args.extra_args)
            + ["--rollout-batch-size", "1"]
        )
        assert parsed.rematerialize_param_from_master_weight is False
        _validate_rematerialize_param_from_master_weight(parsed)


@pytest.mark.parametrize(
    "override",
    [
        {"offload_train_target": "disk", "offload_train_disk_dir": "/tmp/offload"},
        {"enable_optimizer_offload": True},
    ],
)
def test_rollout_only_rejects_trainer_offload_options(launcher, override):
    module, _ = launcher

    with pytest.raises(ValueError, match="Trainer offload options are invalid"):
        module.ScriptArgs(
            hardware="MI355X",
            model_name="GLM-5.2_5layer",
            num_nodes=1,
            num_gpus_per_node=4,
            rollout_only=True,
            **override,
        )


def test_fp8_gate_accepts_the_audited_quantized_weight_diagnostic(launcher):
    module, _ = launcher
    args = module.ScriptArgs(
        **(
            _full_overrides()
            | {
                "fp8_rollout": True,
                "allow_unvalidated_features": True,
                "extra_args": "--ci-test --check-weight-update-allow-quant-error",
            }
        )
    )

    assert "--check-weight-update-allow-quant-error" in shlex.split(args.extra_args)


def test_full_rollout_probe_requires_typed_graph_and_capture_fields(launcher):
    module, _ = launcher
    base = _full_overrides() | {"rollout_only": True}

    with pytest.raises(ValueError, match="rollout_probe_mode"):
        module.ScriptArgs(**(base | {"extra_args": "--sglang-disable-cuda-graph"}))
    with pytest.raises(ValueError, match="rollout_probe_capture"):
        module.ScriptArgs(**(base | {"extra_args": "--save-debug-rollout-data /tmp/out.pt"}))


def test_typed_full_rollout_probe_emits_eager_mode_and_quoted_capture(launcher):
    module, _ = launcher
    args = module.ScriptArgs(
        **(
            _full_overrides()
            | {
                "rollout_only": True,
                "rollout_probe_mode": "eager",
                "rollout_probe_capture": "/tmp/a capture/{rollout_id}.pt",
                "extra_args": (
                    "--use-miles-dashboard --dump-details /tmp/gate3/details "
                    "--debug-exit-after-rollout 1"
                ),
            }
        )
    )

    assert "--sglang-disable-cuda-graph" in shlex.split(module._sglang_args(args))
    assert _flag_value(module._misc_args(args), "--save-debug-rollout-data") == (
        "/tmp/a capture/{rollout_id}.pt"
    )
    assert "--use-miles-dashboard" in shlex.split(args.extra_args)


@pytest.mark.parametrize(
    "override",
    [
        {"rollout_probe_mode": "eager"},
        {"rollout_probe_capture": "/tmp/out.pt"},
        {"rollout_probe_capture": "   ", "rollout_only": True},
    ],
)
def test_probe_fields_reject_non_probe_or_blank_values(launcher, override):
    module, _ = launcher

    with pytest.raises(ValueError, match="rollout_probe"):
        module.ScriptArgs(**(_full_overrides() | override))


@pytest.mark.parametrize(
    ("num_nodes", "num_gpus_per_node"),
    [
        (1, 8),
        (4, 8),
        (8, 4),
        (8, 7),
        (16, 4),
    ],
)
def test_full_model_rejects_every_unvalidated_cluster_shape(launcher, num_nodes, num_gpus_per_node):
    module, _ = launcher

    with pytest.raises((NotImplementedError, ValueError)):
        args = module.ScriptArgs(
            hardware="MI355X",
            model_name="GLM-5.2",
            num_nodes=num_nodes,
            num_gpus_per_node=num_gpus_per_node,
            run_id="unit-test",
        )
        module._get_parallel_config(args)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"num_rollout": 0}, "num_rollout must be positive"),
        ({"rollout_max_prompt_len": 0}, "rollout_max_prompt_len must be positive"),
        ({"rollout_max_response_len": 0}, "rollout_max_response_len must be positive"),
        ({"max_tokens_per_gpu": 0}, "max_tokens_per_gpu must be positive"),
        ({"sglang_mem_fraction_static": 0.0}, "sglang_mem_fraction_static must be between"),
        ({"sglang_cuda_graph_max_bs": 0}, "sglang_cuda_graph_max_bs must be positive"),
        ({"sglang_max_running_requests": 0}, "sglang_max_running_requests must be positive"),
        ({"freeze_indexer": False}, "supports only --freeze-indexer"),
        ({"freeze_router": False}, "have not passed the bring-up gates"),
        ({"use_deepep": True}, "requires --fp8-rollout"),
        (
            {"stream_optimizer_state_to_disk": True},
            "requires offload_train_target=disk",
        ),
        (
            {
                "offload_train_target": "disk",
                "stream_optimizer_state_to_disk": True,
                "enable_optimizer_offload": True,
            },
            "excludes enable_optimizer_offload",
        ),
    ],
)
def test_full_model_rejects_unsafe_recipe_overrides(launcher, override, message):
    module, _ = launcher

    with pytest.raises((NotImplementedError, ValueError), match=message):
        module.ScriptArgs(**(_full_overrides() | override))


@pytest.mark.parametrize("feature", ["fp8_rollout", "enable_indexer_replay", "enable_mtp"])
def test_full_model_requires_an_explicit_escape_hatch_for_unvalidated_features(launcher, feature):
    module, _ = launcher

    with pytest.raises(NotImplementedError, match="have not passed the bring-up gates"):
        module.ScriptArgs(**(_full_overrides() | {feature: True}))


def test_full_profile_token_packing_target_covers_the_maximum_response(launcher):
    module, _ = launcher
    args = module.ScriptArgs(**_full_overrides())

    assert args.max_tokens_per_gpu == 16384
    assert args.max_tokens_per_gpu >= args.rollout_max_prompt_len + args.rollout_max_response_len

    with pytest.raises(ValueError, match="packing target must cover"):
        module.ScriptArgs(
            **(_full_overrides() | {"max_tokens_per_gpu": 4096, "rollout_max_response_len": 8192})
        )


def test_prepare_download_pins_model_and_dataset_revisions(launcher, tmp_path):
    module, recording = launcher
    overrides = _full_overrides() | {"model_dir": str(tmp_path / "models"), "data_dir": str(tmp_path / "data")}

    call_entrypoint(module, "_prepare_download", overrides, sandbox=tmp_path)

    assert f"--revision {module._MODEL_REVISIONS['GLM-5.2']}" in recording.commands[1]
    assert f"--revision {module._DATA_REVISION}" in recording.commands[2]
    assert "tools/validate_grpo_dataset.py" in recording.commands[3]
    assert "--profile dapo-math-17k" in recording.commands[3]
    assert f"--tokenizer {tmp_path}/models/GLM-5.2" in recording.commands[3]
    assert "--max-prompt-tokens 4096" in recording.commands[3]
    assert "--expected-min-prompt-tokens 73" in recording.commands[3]
    assert "--expected-max-prompt-tokens 1521" in recording.commands[3]


def test_five_layer_download_pins_the_expected_filtered_prompt_count(launcher, tmp_path):
    module, recording = launcher
    overrides = {
        "hardware": "MI355X",
        "model_name": "GLM-5.2_5layer",
        "num_nodes": 1,
        "num_gpus_per_node": 4,
        "model_dir": str(tmp_path / "models"),
        "data_dir": str(tmp_path / "data"),
    }

    call_entrypoint(module, "_prepare_download", overrides, sandbox=tmp_path)

    assert f"--tokenizer {tmp_path}/models/GLM-5.2_5layer" in recording.commands[3]
    assert "--max-prompt-tokens 1024" in recording.commands[3]
    assert "--expected-prompts-over-token-limit 7" in recording.commands[3]
    assert "--expected-min-prompt-tokens 73" in recording.commands[3]
    assert "--expected-max-prompt-tokens 1521" in recording.commands[3]


def test_first_fp8_prepare_validates_the_source_tokenizer_before_conversion(launcher, tmp_path):
    module, recording = launcher
    overrides = _full_overrides() | {
        "model_dir": str(tmp_path / "models"),
        "data_dir": str(tmp_path / "data"),
        "fp8_rollout": True,
        "allow_unvalidated_features": True,
    }

    call_entrypoint(module, "_prepare_download", overrides, sandbox=tmp_path)

    assert f"--tokenizer {tmp_path}/models/GLM-5.2" in recording.commands[3]
    assert "GLM-5.2_fp8" not in recording.commands[3]


def test_public_download_stages_and_validates_without_ray_or_gpu(launcher, monkeypatch, tmp_path):
    module, recording = launcher
    checkpoint, _ = _write_valid_checkpoint(module, tmp_path / "models")
    monkeypatch.delenv("MILES_SCRIPT_EXTERNAL_RAY")
    monkeypatch.setattr(
        module._cluster,
        "_visible_rocm_product_names",
        lambda: (_ for _ in ()).throw(AssertionError("download must not inspect GPUs")),
    )
    call_entrypoint(
        module,
        "download",
        _full_overrides()
        | {
            "model_dir": str(checkpoint.parent),
            "data_dir": str(tmp_path / "data"),
        },
        sandbox=tmp_path,
    )

    assert any("hf download zai-org/GLM-5.2" in command for command in recording.commands)
    assert any("tools/validate_grpo_dataset.py" in command for command in recording.commands)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("hidden_size", 4096),
        ("index_topk", 1024),
        ("num_hidden_layers", 77),
        ("indexer_types", ["full"] * 78),
        ("mlp_layer_types", ["sparse"] * 78),
        ("rope_parameters", {"rope_theta": 10000, "rope_type": "default"}),
    ],
)
def test_checkpoint_validation_rejects_architecture_drift(launcher, tmp_path, field, bad_value):
    module, _ = launcher
    checkpoint, config = _write_valid_checkpoint(module, tmp_path)
    config[field] = bad_value
    (checkpoint / "config.json").write_text(json.dumps(config))
    args = module.ScriptArgs(**(_full_overrides() | {"model_dir": str(tmp_path)}))
    module._write_artifact_manifest(checkpoint, module._source_manifest(args))

    with pytest.raises(RuntimeError, match=field):
        module._validate_glm_checkpoint(args)


def test_checkpoint_validation_accepts_a_complete_audited_layout(launcher, tmp_path):
    module, _ = launcher
    _write_valid_checkpoint(module, tmp_path)
    args = module.ScriptArgs(**(_full_overrides() | {"model_dir": str(tmp_path)}))

    module._validate_glm_checkpoint(args)


def test_checkpoint_validation_rejects_a_corrupted_runtime_hf_asset(launcher, tmp_path):
    module, _ = launcher
    checkpoint, _ = _write_valid_checkpoint(module, tmp_path)
    tokenizer_config = checkpoint / "tokenizer_config.json"
    tokenizer_config.write_bytes(b"x" * tokenizer_config.stat().st_size)
    args = module.ScriptArgs(**(_full_overrides() | {"model_dir": str(tmp_path)}))

    with pytest.raises(RuntimeError, match="Runtime HF asset.*SHA-256"):
        module._validate_glm_checkpoint(args)


@pytest.mark.parametrize("artifact", ["config", "index"])
def test_checkpoint_validation_rejects_semantically_plausible_source_identity_drift(
    launcher, tmp_path, artifact
):
    module, _ = launcher
    checkpoint, config = _write_valid_checkpoint(module, tmp_path)
    if artifact == "config":
        config["unvalidated_extra_field"] = True
        (checkpoint / "config.json").write_text(json.dumps(config))
    else:
        index_path = checkpoint / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())
        index["weight_map"] = {
            "model.embed_tokens.weighx": "model-00001-of-00001.safetensors"
        }
        index_path.write_text(json.dumps(index))
    args = module.ScriptArgs(**(_full_overrides() | {"model_dir": str(tmp_path)}))
    module._write_artifact_manifest(checkpoint, module._source_manifest(args))

    with pytest.raises(RuntimeError, match=f"Pinned source {artifact}.*(?:bytes|SHA-256)"):
        module._validate_glm_checkpoint(args)


@pytest.mark.parametrize("drift", ["weight_count", "shard_count", "total_size"])
def test_checkpoint_validation_rejects_pinned_index_layout_drift(launcher, tmp_path, drift):
    module, _ = launcher
    checkpoint, _ = _write_valid_checkpoint(module, tmp_path)
    index_path = checkpoint / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())

    if drift == "weight_count":
        index["weight_map"]["model.layers.0.weight"] = "model-00001-of-00001.safetensors"
    elif drift == "shard_count":
        second_shard = "model-00002-of-00002.safetensors"
        index["weight_map"]["model.layers.0.weight"] = second_shard
        (checkpoint / second_shard).write_bytes(b"fixture")
    else:
        index["metadata"]["total_size"] += 1
    index_path.write_text(json.dumps(index))
    args = module.ScriptArgs(**(_full_overrides() | {"model_dir": str(tmp_path)}))
    module._write_artifact_manifest(checkpoint, module._source_manifest(args))

    with pytest.raises(RuntimeError, match=r"weight/shard/size layout .* expected \(1, 1, 7\)"):
        module._validate_glm_checkpoint(args)


def test_checkpoint_validation_rejects_a_missing_weight_shard(launcher, tmp_path):
    module, _ = launcher
    checkpoint, _ = _write_valid_checkpoint(module, tmp_path)
    (checkpoint / "model-00001-of-00001.safetensors").unlink()
    args = module.ScriptArgs(**(_full_overrides() | {"model_dir": str(tmp_path)}))

    with pytest.raises(FileNotFoundError, match="missing or empty shard"):
        module._validate_glm_checkpoint(args)


def test_checkpoint_validation_rejects_an_empty_weight_map(launcher, tmp_path):
    module, _ = launcher
    checkpoint, _ = _write_valid_checkpoint(module, tmp_path)
    (checkpoint / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
    args = module.ScriptArgs(**(_full_overrides() | {"model_dir": str(tmp_path)}))

    with pytest.raises(RuntimeError, match="non-empty weight_map"):
        module._validate_glm_checkpoint(args)


def test_checkpoint_validation_rejects_a_shard_path_outside_the_artifact(launcher, tmp_path):
    module, _ = launcher
    checkpoint, _ = _write_valid_checkpoint(module, tmp_path)
    index_path = checkpoint / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["weight_map"]["model.embed_tokens.weight"] = "../outside.safetensors"
    index_path.write_text(json.dumps(index))
    args = module.ScriptArgs(**(_full_overrides() | {"model_dir": str(tmp_path)}))

    with pytest.raises(RuntimeError, match="unsafe or invalid shard names"):
        module._validate_glm_checkpoint(args)


def test_checkpoint_validation_rejects_a_stale_source_revision(launcher, tmp_path):
    module, _ = launcher
    checkpoint, _ = _write_valid_checkpoint(module, tmp_path)
    manifest_path = checkpoint / module._ARTIFACT_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    manifest["revision"] = "0" * 40
    manifest_path.write_text(json.dumps(manifest))
    args = module.ScriptArgs(**(_full_overrides() | {"model_dir": str(tmp_path)}))

    with pytest.raises(RuntimeError, match="does not match the requested recipe"):
        module._validate_glm_checkpoint(args)


def test_manifest_inventory_rejects_a_nonempty_but_truncated_shard(launcher, tmp_path):
    module, _ = launcher
    checkpoint, _ = _write_valid_checkpoint(module, tmp_path)
    (checkpoint / "model-00001-of-00001.safetensors").write_bytes(b"x")
    args = module.ScriptArgs(**(_full_overrides() | {"model_dir": str(tmp_path)}))

    with pytest.raises(RuntimeError, match="file inventory does not match"):
        module._validate_glm_checkpoint(args)


def test_artifact_bundle_validates_source_fp8_and_torch_dist_together(launcher, tmp_path):
    module, _ = launcher
    _write_valid_fp8_checkpoint(module, tmp_path)
    _write_valid_torch_dist_checkpoint(module, tmp_path)

    module._validate_artifact_bundle(
        tmp_path,
        model_name="GLM-5.2",
        model_org="zai-org",
        model_revision=module._MODEL_REVISIONS["GLM-5.2"],
        fp8_rollout=True,
        require_source=True,
    )


def test_node_local_artifact_bundle_rejects_a_corrupted_fp8_tokenizer(launcher, tmp_path):
    module, _ = launcher
    _, fp8_checkpoint = _write_valid_fp8_checkpoint(module, tmp_path)
    tokenizer = fp8_checkpoint / "tokenizer.json"
    tokenizer.write_bytes(b"x" * tokenizer.stat().st_size)

    with pytest.raises(RuntimeError, match="Runtime HF asset.*SHA-256"):
        module._validate_artifact_bundle(
            tmp_path,
            model_name="GLM-5.2",
            model_org="zai-org",
            model_revision=module._MODEL_REVISIONS["GLM-5.2"],
            fp8_rollout=True,
            require_source=False,
            rollout_only=True,
        )


def test_rollout_only_artifact_bundle_needs_hf_but_not_torch_dist(launcher, tmp_path):
    module, _ = launcher
    _write_valid_checkpoint(module, tmp_path)

    module._validate_artifact_bundle(
        tmp_path,
        model_name="GLM-5.2",
        model_org="zai-org",
        model_revision=module._MODEL_REVISIONS["GLM-5.2"],
        fp8_rollout=False,
        require_source=True,
        rollout_only=True,
    )

    with pytest.raises(RuntimeError, match="Refusing to reuse"):
        module._validate_artifact_bundle(
            tmp_path,
            model_name="GLM-5.2",
            model_org="zai-org",
            model_revision=module._MODEL_REVISIONS["GLM-5.2"],
            fp8_rollout=False,
            require_source=True,
        )


def test_artifact_bundle_rejects_a_missing_torch_dist_payload(launcher, tmp_path):
    module, _ = launcher
    _write_valid_checkpoint(module, tmp_path)
    torch_dist = _write_valid_torch_dist_checkpoint(module, tmp_path)
    (torch_dist / "release/__0_0.distcp").unlink()

    with pytest.raises(FileNotFoundError, match="__0_0.distcp is missing or empty"):
        module._validate_artifact_bundle(
            tmp_path,
            model_name="GLM-5.2",
            model_org="zai-org",
            model_revision=module._MODEL_REVISIONS["GLM-5.2"],
            fp8_rollout=False,
            require_source=True,
        )


def test_torch_dist_validation_rejects_a_nonexact_release_tracker(launcher, tmp_path):
    module, _ = launcher
    _write_valid_checkpoint(module, tmp_path)
    torch_dist = _write_valid_torch_dist_checkpoint(module, tmp_path)
    (torch_dist / "latest_checkpointed_iteration.txt").write_text("release\n")

    with pytest.raises(RuntimeError, match="must contain exactly 'release'"):
        module._validate_artifact_bundle(
            tmp_path,
            model_name="GLM-5.2",
            model_org="zai-org",
            model_revision=module._MODEL_REVISIONS["GLM-5.2"],
            fp8_rollout=False,
            require_source=True,
        )


def test_torch_dist_validation_rejects_a_truncated_referenced_range(launcher, tmp_path):
    module, _ = launcher
    _write_valid_checkpoint(module, tmp_path)
    torch_dist = _write_valid_torch_dist_checkpoint(module, tmp_path)
    (torch_dist / "release/__0_0.distcp").write_bytes(b"x")

    with pytest.raises(RuntimeError, match=r"is truncated: 1 bytes, metadata requires \d+"):
        module._validate_artifact_bundle(
            tmp_path,
            model_name="GLM-5.2",
            model_org="zai-org",
            model_revision=module._MODEL_REVISIONS["GLM-5.2"],
            fp8_rollout=False,
            require_source=True,
        )


def test_fp8_validation_rejects_a_wrong_quantization_recipe(launcher, tmp_path):
    module, _ = launcher
    _, checkpoint = _write_valid_fp8_checkpoint(module, tmp_path)
    config_path = checkpoint / "config.json"
    config = json.loads(config_path.read_text())
    config["quantization_config"]["weight_block_size"] = [64, 64]
    config_path.write_text(json.dumps(config))
    source_manifest = module._source_manifest_values(
        "zai-org", "GLM-5.2", module._MODEL_REVISIONS["GLM-5.2"]
    )
    module._write_artifact_manifest(checkpoint, module._fp8_manifest_values(source_manifest))

    with pytest.raises(RuntimeError, match="required block-FP8 rollout recipe"):
        module._validate_fp8_checkpoint_at(
            checkpoint,
            model_name="GLM-5.2",
            expected_manifest=module._fp8_manifest_values(source_manifest),
        )


def test_fp8_conversion_refuses_an_untracked_existing_artifact(launcher, tmp_path):
    module, _ = launcher
    fp8_dir = tmp_path / "GLM-5.2_fp8"
    fp8_dir.mkdir()
    (fp8_dir / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "x"}}))
    args = module.ScriptArgs(**(_full_overrides() | {"model_dir": str(tmp_path)}))

    with pytest.raises(RuntimeError, match="Refusing to reuse"):
        module._convert_to_fp8(args)


def test_full_conversion_uses_all_64_ranks_and_dsa_safe_pipeline_split(launcher, tmp_path):
    module, recording = launcher
    overrides = _full_overrides() | {"model_dir": str(tmp_path)}

    call_entrypoint(module, "_prepare_megatron_ckpt", overrides, sandbox=tmp_path)

    command = recording.commands[-1]
    assert command.startswith("[multi_node num_nodes=8]")
    assert "--nproc-per-node 8" in command
    assert "--nnodes={{nnodes}}" in command
    assert _flag_value(command, "--tensor-model-parallel-size") == "1"
    assert _flag_value(command, "--pipeline-model-parallel-size") == "4"
    assert _flag_value(command, "--expert-model-parallel-size") == "16"
    assert _flag_value(command, "--decoder-first-pipeline-num-layers") == "18"
    assert _flag_value(command, "--decoder-last-pipeline-num-layers") == "20"


def test_full_single_node_rollout_skips_megatron_conversion(launcher, tmp_path):
    module, recording = launcher
    overrides = {
        "hardware": "MI355X",
        "model_name": "GLM-5.2",
        "num_nodes": 1,
        "num_gpus_per_node": 8,
        "full_model_rollout_only": True,
        "allow_unvalidated_features": True,
        "model_dir": str(tmp_path),
    }

    call_entrypoint(module, "_prepare_megatron_ckpt", overrides, sandbox=tmp_path)

    assert recording.commands == []


def test_full_prepare_cp_copies_both_checkpoints_to_all_nodes(launcher, tmp_path):
    module, recording = launcher
    local_dir = tmp_path / "local"
    overrides = _full_overrides() | {
        "model_dir": str(tmp_path),
        "model_local_dir": str(local_dir),
    }

    call_entrypoint(module, "_prepare_cp", overrides, sandbox=tmp_path)

    assert len(recording.commands) == 4
    assert "validate-artifacts-internal" in recording.commands[0]
    assert "--require-source" in recording.commands[0]
    assert all(command.startswith("[multi_node num_nodes=8]") for command in recording.commands[1:])
    assert f"{tmp_path}/GLM-5.2_torch_dist/" in recording.commands[1]
    assert f"{local_dir}/GLM-5.2_torch_dist" in recording.commands[1]
    assert f"{tmp_path}/GLM-5.2/" in recording.commands[2]
    assert f"{local_dir}/GLM-5.2" in recording.commands[2]
    assert "validate-artifacts-internal" in recording.commands[3]
    assert f"--root {local_dir}" in recording.commands[3]
    assert "--require-source" not in recording.commands[3]


def test_full_single_node_rollout_copy_stages_only_the_hf_checkpoint(launcher, monkeypatch, tmp_path):
    module, recording = launcher
    monkeypatch.delenv("MILES_SCRIPT_EXTERNAL_RAY")
    monkeypatch.delenv("RAY_ADDRESS")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    local_dir = tmp_path / "local"
    overrides = {
        "hardware": "MI355X",
        "model_name": "GLM-5.2",
        "num_nodes": 1,
        "num_gpus_per_node": 8,
        "full_model_rollout_only": True,
        "allow_unvalidated_features": True,
        "model_dir": str(tmp_path),
        "model_local_dir": str(local_dir),
    }

    call_entrypoint(module, "_prepare_cp", overrides, sandbox=tmp_path)

    assert len(recording.commands) == 3
    assert "--rollout-only" in recording.commands[0]
    assert "--require-source" in recording.commands[0]
    assert "GLM-5.2_torch_dist" not in " ".join(recording.commands)
    assert f"{tmp_path}/GLM-5.2/" in recording.commands[1]
    assert f"{local_dir}/GLM-5.2" in recording.commands[1]
    assert "--rollout-only" in recording.commands[2]


def test_fp8_prepare_cp_validates_source_then_copies_and_validates_fp8(launcher, tmp_path):
    module, recording = launcher
    local_dir = tmp_path / "local"
    overrides = _full_overrides() | {
        "model_dir": str(tmp_path),
        "model_local_dir": str(local_dir),
        "fp8_rollout": True,
        "allow_unvalidated_features": True,
    }

    call_entrypoint(module, "_prepare_cp", overrides, sandbox=tmp_path)

    assert "--fp8-rollout --require-source" in recording.commands[0]
    assert f"{tmp_path}/GLM-5.2_fp8/" in recording.commands[2]
    assert f"{local_dir}/GLM-5.2_fp8" in recording.commands[2]
    assert "--fp8-rollout" in recording.commands[3]
    assert "--require-source" not in recording.commands[3]


def test_full_prepare_cp_never_trusts_head_local_sentinels(launcher, tmp_path):
    module, recording = launcher
    local_dir = tmp_path / "local"
    torch_dist_dst = local_dir / "GLM-5.2_torch_dist"
    hf_dst = local_dir / "GLM-5.2"
    torch_dist_dst.mkdir(parents=True)
    hf_dst.mkdir(parents=True)
    (torch_dist_dst / "latest_checkpointed_iteration.txt").write_text("release")
    (hf_dst / "model.safetensors.index.json").write_text("{}")
    args = module.ScriptArgs(
        **(
            _full_overrides()
            | {
                "model_dir": str(tmp_path),
                "model_local_dir": str(local_dir),
            }
        )
    )

    module._prepare_cp(args, skip_existing=True)

    assert len(recording.commands) == 4
    assert all(command.startswith("[multi_node num_nodes=8]") for command in recording.commands[1:])


def test_five_layer_copy_is_local_and_does_not_require_ray(launcher, monkeypatch, tmp_path):
    module, recording = launcher
    monkeypatch.delenv("MILES_SCRIPT_EXTERNAL_RAY")
    monkeypatch.delenv("RAY_ADDRESS")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    local_dir = tmp_path / "local"
    overrides = {
        "hardware": "MI355X",
        "model_name": "GLM-5.2_5layer",
        "num_nodes": 1,
        "num_gpus_per_node": 4,
        "model_dir": str(tmp_path),
        "model_local_dir": str(local_dir),
    }

    call_entrypoint(module, "_prepare_cp", overrides, sandbox=tmp_path)

    assert len(recording.commands) == 4
    assert all(not command.startswith("[multi_node") for command in recording.commands)
    assert all("rsync -a --info=progress2" in command for command in recording.commands[1:3])
    assert "validate-artifacts-internal" in recording.commands[0]
    assert "validate-artifacts-internal" in recording.commands[3]


def test_full_model_requires_an_external_ray_cluster(launcher, monkeypatch):
    module, _ = launcher
    monkeypatch.delenv("MILES_SCRIPT_EXTERNAL_RAY")
    args = module.ScriptArgs(**_full_overrides())

    with pytest.raises(RuntimeError, match="already joined Ray cluster"):
        module._execute_train(args)


@pytest.mark.parametrize("entrypoint", ["full_train", "prepare"])
def test_full_public_preparation_commands_preflight_ray_before_download(
    launcher, monkeypatch, tmp_path, entrypoint
):
    module, recording = launcher
    monkeypatch.delenv("MILES_SCRIPT_EXTERNAL_RAY")
    with pytest.raises(RuntimeError, match="already joined Ray cluster"):
        call_entrypoint(module, entrypoint, _full_overrides(), sandbox=tmp_path)

    assert recording.commands == []


def test_single_node_full_train_preflights_hardware_before_download(
    launcher, monkeypatch, tmp_path
):
    module, recording = launcher
    monkeypatch.delenv("MILES_SCRIPT_EXTERNAL_RAY")
    monkeypatch.delenv("RAY_ADDRESS")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setattr(
        module._cluster,
        "_visible_rocm_product_names",
        lambda: ("AMD Instinct MI355X",) * 7,
    )
    overrides = {
        "hardware": "MI355X",
        "model_name": "GLM-5.2",
        "num_nodes": 1,
        "num_gpus_per_node": 8,
        "full_model_rollout_only": True,
        "allow_unvalidated_features": True,
    }

    with pytest.raises(RuntimeError, match="exactly 8 visible homogeneous MI355X"):
        call_entrypoint(module, "full_train", overrides, sandbox=tmp_path)

    assert recording.commands == []


@pytest.mark.parametrize(
    "args",
    [
        {
            "hardware": "MI355X",
            "model_name": "GLM-5.2_5layer",
            "num_nodes": 1,
            "num_gpus_per_node": 4,
        },
        {
            "hardware": "MI355X",
            "model_name": "GLM-5.2",
            "num_nodes": 1,
            "num_gpus_per_node": 8,
            "full_model_rollout_only": True,
            "allow_unvalidated_features": True,
        },
    ],
)
def test_single_node_profiles_refuse_external_ray_submission(launcher, args):
    module, _ = launcher

    with pytest.raises(RuntimeError, match="refuses MILES_SCRIPT_EXTERNAL_RAY=1"):
        module._require_external_ray(module.ScriptArgs(**args), "training")


def test_single_node_profile_refuses_a_stale_ray_address(launcher, monkeypatch):
    module, _ = launcher
    monkeypatch.delenv("MILES_SCRIPT_EXTERNAL_RAY")
    args = module.ScriptArgs(
        hardware="MI355X",
        model_name="GLM-5.2_5layer",
        num_nodes=1,
        num_gpus_per_node=4,
    )

    with pytest.raises(RuntimeError, match="refuses a preconfigured RAY_ADDRESS"):
        module._require_external_ray(args, "training")


def test_single_node_profile_rejects_nonloopback_master_and_normalizes_localhost(
    launcher, monkeypatch
):
    module, _ = launcher
    monkeypatch.delenv("MILES_SCRIPT_EXTERNAL_RAY")
    monkeypatch.delenv("RAY_ADDRESS")
    args = module.ScriptArgs(
        hardware="MI355X",
        model_name="GLM-5.2_5layer",
        num_nodes=1,
        num_gpus_per_node=4,
    )
    monkeypatch.setenv("MASTER_ADDR", "10.0.0.1")

    with pytest.raises(RuntimeError, match="requires a loopback MASTER_ADDR"):
        module._require_external_ray(args, "training")

    monkeypatch.setenv("MASTER_ADDR", "localhost")
    module._require_external_ray(args, "training")
    assert module.os.environ["MASTER_ADDR"] == "127.0.0.1"


@pytest.mark.parametrize("hardware", ["MI350X", "MI355X"])
def test_single_node_hardware_validation_accepts_exact_visible_inventory(
    launcher, monkeypatch, hardware
):
    module, _ = launcher
    monkeypatch.setattr(
        module._cluster,
        "_visible_rocm_product_names",
        lambda: (f"AMD Instinct {hardware}",) * 4,
    )
    args = module.ScriptArgs(
        hardware=hardware,
        model_name="GLM-5.2_5layer",
        num_nodes=1,
        num_gpus_per_node=4,
    )

    module._cluster._validate_local_hardware(args)


@pytest.mark.parametrize(
    "product_names",
    [
        ("AMD Instinct MI355X",) * 3,
        ("AMD Instinct MI355X",) * 3 + ("AMD Instinct MI350X",),
        ("AMD Instinct MI355X",) * 3 + ("AMD GFX950",),
    ],
)
def test_single_node_hardware_validation_rejects_wrong_count_or_sku(
    launcher, monkeypatch, product_names
):
    module, _ = launcher
    monkeypatch.setattr(module._cluster, "_visible_rocm_product_names", lambda: product_names)
    args = module.ScriptArgs(
        hardware="MI355X",
        model_name="GLM-5.2_5layer",
        num_nodes=1,
        num_gpus_per_node=4,
    )

    with pytest.raises(RuntimeError, match="exactly 4 visible homogeneous MI355X"):
        module._cluster._validate_local_hardware(args)


def test_full_model_requires_an_explicit_ray_address(launcher, monkeypatch):
    module, _ = launcher
    monkeypatch.delenv("RAY_ADDRESS")
    args = module.ScriptArgs(**_full_overrides())

    with pytest.raises(RuntimeError, match="explicit RAY_ADDRESS"):
        module._execute_train(args)


def test_full_model_requires_a_ray_dashboard_address(launcher, monkeypatch):
    module, _ = launcher
    monkeypatch.setenv("RAY_ADDRESS", "auto")
    args = module.ScriptArgs(**_full_overrides())

    with pytest.raises(RuntimeError, match=r"HTTP\(S\) Ray dashboard URL"):
        module._execute_train(args)


def test_full_model_requires_an_explicit_master_address(launcher, monkeypatch):
    module, _ = launcher
    monkeypatch.delenv("MASTER_ADDR")
    args = module.ScriptArgs(**_full_overrides())

    with pytest.raises(RuntimeError, match="explicit MASTER_ADDR"):
        module._execute_train(args)


def test_full_model_rejects_a_loopback_master_address(launcher, monkeypatch):
    module, _ = launcher
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    args = module.ScriptArgs(**_full_overrides())

    with pytest.raises(RuntimeError, match="MASTER_ADDR.*outside the inspected cluster"):
        module._execute_train(args)


@pytest.mark.parametrize("hardware", ["MI350X", "MI355X"])
def test_full_model_accepts_a_homogeneous_cluster_of_the_requested_sku(launcher, monkeypatch, hardware):
    module, _ = launcher
    nodes, probes, available_gpus, driver_node_address = _valid_cluster_inventory()
    for probe in probes:
        probe["product_names"] = [f"AMD Instinct {hardware}"] * 8
    monkeypatch.setattr(
        module._cluster,
        "_collect_ray_cluster_inventory",
        lambda: (nodes, probes, available_gpus, driver_node_address),
    )
    args = module.ScriptArgs(**_full_overrides(hardware))

    module._require_external_ray(args, "training")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda nodes, probes: nodes.pop(), "exactly 8 alive nodes"),
        (lambda nodes, probes: nodes[0]["Resources"].update(GPU=7.0), "advertises 7 GPUs"),
        (lambda nodes, probes: probes[0].update(gfx950_count=7), "7 gfx950 agents"),
        (
            lambda nodes, probes: probes[0]["product_names"].__setitem__(0, "AMD Instinct MI350X"),
            "do not match requested MI355X",
        ),
        (lambda nodes, probes: probes[0].update(product_names=[]), "0 GPU product names"),
        (lambda nodes, probes: probes[0].update(error="rocminfo failed"), "hardware probe failed"),
    ],
)
def test_full_model_rejects_nonuniform_or_non_gfx950_ray_clusters(launcher, monkeypatch, mutate, message):
    module, _ = launcher
    nodes, probes, available_gpus, driver_node_address = _valid_cluster_inventory()
    mutate(nodes, probes)
    monkeypatch.setattr(
        module._cluster,
        "_collect_ray_cluster_inventory",
        lambda: (nodes, probes, available_gpus, driver_node_address),
    )
    args = module.ScriptArgs(**_full_overrides())

    with pytest.raises(RuntimeError, match=message):
        module._require_external_ray(args, "training")


def test_full_model_rejects_a_ray_cluster_with_busy_gpus(launcher, monkeypatch):
    module, _ = launcher
    nodes, probes, _, driver_node_address = _valid_cluster_inventory()
    monkeypatch.setattr(
        module._cluster,
        "_collect_ray_cluster_inventory",
        lambda: (nodes, probes, 63.0, driver_node_address),
    )
    args = module.ScriptArgs(**_full_overrides())

    with pytest.raises(RuntimeError, match="64 available GPUs, found 63"):
        module._require_external_ray(args, "training")


def test_full_model_rejects_a_dashboard_outside_the_inspected_cluster(launcher, monkeypatch):
    module, _ = launcher
    monkeypatch.setenv("RAY_ADDRESS", "http://192.0.2.10:8265")
    monkeypatch.setattr(
        module._cluster.socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (module._cluster.socket.AF_INET, module._cluster.socket.SOCK_STREAM, 6, "", ("192.0.2.10", 0))
        ],
    )
    args = module.ScriptArgs(**_full_overrides())

    with pytest.raises(RuntimeError, match="resolves outside the inspected cluster"):
        module._require_external_ray(args, "training")


def test_full_model_rejects_an_unresolvable_dashboard_host(launcher, monkeypatch):
    module, _ = launcher
    monkeypatch.setenv("RAY_ADDRESS", "http://ray.invalid:8265")

    def fail_resolution(*args, **kwargs):
        raise module._cluster.socket.gaierror("fixture resolution failure")

    monkeypatch.setattr(module._cluster.socket, "getaddrinfo", fail_resolution)
    args = module.ScriptArgs(**_full_overrides())

    with pytest.raises(RuntimeError, match="RAY_ADDRESS host.*could not be resolved"):
        module._require_external_ray(args, "training")


def test_amd_smi_json_parser_extracts_one_market_name_per_gpu(launcher):
    module, _ = launcher
    output = json.dumps(
        {
            "gpu_data": [
                {"gpu": rank, "asic": {"market_name": "AMD Instinct MI355X", "vendor_name": "AMD"}}
                for rank in range(8)
            ]
        }
    )

    assert module._amd_smi_product_names(output) == ["AMD Instinct MI355X"] * 8


def test_product_probe_falls_back_when_sysfs_does_not_identify_the_sku(launcher, monkeypatch):
    module, _ = launcher
    output = json.dumps(
        {
            "gpu_data": [
                {"gpu": rank, "asic": {"market_name": "MI350X"}}
                for rank in range(8)
            ]
        }
    )
    monkeypatch.setattr(module._cluster, "_sysfs_product_names", lambda: ["AMD GFX950"] * 8)
    monkeypatch.setattr(
        module._cluster.subprocess,
        "run",
        lambda *args, **kwargs: module._cluster.subprocess.CompletedProcess(
            args=args[0], returncode=0, stdout=output
        ),
    )

    product_names, source, error = module._probe_product_names(8)

    assert product_names == ["MI350X"] * 8
    assert source == "amd-smi"
    assert error is None


def test_external_ray_network_bypasses_every_validated_node_and_clears_proxies(launcher, monkeypatch):
    module, _ = launcher
    for name in module._cluster._PROXY_ENV_VARS:
        monkeypatch.setenv(name, "http://proxy.invalid:3128")
    args = module.ScriptArgs(**_full_overrides())

    module._require_external_ray(args, "training")

    assert all(name not in module.os.environ for name in module._cluster._PROXY_ENV_VARS)
    assert module.os.environ["no_proxy"] == module.os.environ["NO_PROXY"]
    bypass = module.os.environ["no_proxy"].split(",")
    assert "10.0.0.1" in bypass
    assert "10.0.0.8" in bypass


def test_full_model_train_command_uses_the_rocm_bf16_r3_recipe(launcher, tmp_path):
    module, recording = launcher

    command = _record_train(module, recording, tmp_path, _full_overrides())
    words = shlex.split(command)

    assert len(recording.commands) == 3
    assert recording.commands[0].startswith("[multi_node num_nodes=8]")
    assert "tools/validate_grpo_dataset.py" in recording.commands[0]
    assert "--tokenizer /root/local_data/GLM-5.2" in recording.commands[0]
    assert "--max-prompt-tokens 4096" in recording.commands[0]
    assert "--expected-min-prompt-tokens 73" in recording.commands[0]
    assert "--expected-max-prompt-tokens 1521" in recording.commands[0]
    assert recording.commands[1].startswith("[multi_node num_nodes=8]")
    assert "validate-artifacts-internal" in recording.commands[1]
    assert recording.commands[2] == command

    assert _flag_value(command, "--tensor-model-parallel-size") == "8"
    assert _flag_value(command, "--pipeline-model-parallel-size") == "4"
    assert _flag_value(command, "--context-parallel-size") == "1"
    assert _flag_value(command, "--expert-model-parallel-size") == "16"
    assert _flag_value(command, "--decoder-first-pipeline-num-layers") == "18"
    assert _flag_value(command, "--decoder-last-pipeline-num-layers") == "20"

    assert _flag_value(command, "--sglang-nsa-decode-backend") == "tilelang"
    assert _flag_value(command, "--sglang-nsa-prefill-backend") == "tilelang"
    assert _flag_value(command, "--sglang-cuda-graph-max-bs") == "1"
    assert _flag_value(command, "--sglang-max-running-requests") == "32"
    assert _flag_value(command, "--sglang-mem-fraction-static") == "0.7"
    assert "flashmla" not in command.lower()

    assert "--bf16" in words
    assert "--fp8-format" not in words
    assert "--fp8-recipe" not in words

    assert "--freeze-indexer" in words
    assert "--use-rollout-routing-replay" in words
    assert _flag_value(command, "--moe-token-dispatcher-type") == "alltoall"
    assert "deepep" not in command.lower()


def test_full_single_node_rollout_command_cannot_initialize_the_trainer(
    launcher, monkeypatch, tmp_path
):
    module, recording = launcher
    monkeypatch.delenv("MILES_SCRIPT_EXTERNAL_RAY")
    monkeypatch.delenv("RAY_ADDRESS")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    overrides = {
        "hardware": "MI355X",
        "model_name": "GLM-5.2",
        "num_nodes": 1,
        "num_gpus_per_node": 8,
        "full_model_rollout_only": True,
        "allow_unvalidated_features": True,
        "run_id": "unit-test",
    }

    command = _record_train(module, recording, tmp_path, overrides)
    words = shlex.split(command)

    assert len(recording.commands) >= 3
    assert any("--rollout-only" in recorded for recorded in recording.commands[:-1])
    assert "--debug-rollout-only" in words
    assert "--ref-load" not in words
    assert "--load" not in words
    assert "--save" not in words
    assert "--use-rollout-routing-replay" not in words
    assert "--rematerialize-param-from-master-weight" not in words
    assert "--offload-train-target" not in words
    assert "--stream-optimizer-state-to-disk" not in words
    assert _flag_value(command, "--num-rollout") == "1"
    assert _flag_value(command, "--rollout-batch-size") == "1"
    assert _flag_value(command, "--n-samples-per-prompt") == "1"
    assert _flag_value(command, "--global-batch-size") == "1"
    assert _flag_value(command, "--rollout-temperature") == "0"
    assert _flag_value(command, "--rollout-seed") == "1234"
    assert _flag_value(command, "--rollout-max-response-len") == "128"
    assert _flag_value(command, "--tensor-model-parallel-size") == "8"
    assert _flag_value(command, "--pipeline-model-parallel-size") == "1"
    assert _flag_value(command, "--expert-model-parallel-size") == "8"
    assert _flag_value(command, "--sglang-tp-size") == "8"
    assert _flag_value(command, "--sglang-ep-size") == "8"


def test_fp8_rollout_is_isolated_from_bf16_actor_training(launcher, tmp_path):
    module, recording = launcher
    overrides = _full_overrides() | {
        "fp8_rollout": True,
        "use_deepep": True,
        "allow_unvalidated_features": True,
    }

    command = _record_train(module, recording, tmp_path, overrides)
    words = shlex.split(command)

    assert "--tokenizer /root/local_data/GLM-5.2_fp8" in recording.commands[0]
    assert _flag_value(command, "--hf-checkpoint").endswith("/GLM-5.2_fp8")
    assert _flag_value(command, "--sglang-moe-a2a-backend") == "mori"
    assert _flag_value(command, "--sglang-deepep-mode") == "auto"
    assert "--bf16" in words
    assert "--fp8-format" not in words


def test_disk_offload_omits_incompatible_cpu_rematerialization(launcher, tmp_path):
    module, recording = launcher
    overrides = _full_overrides() | {
        "offload_train_target": "disk",
        "offload_train_disk_dir": "/local_nvme/glm52-offload",
        "stream_optimizer_state_to_disk": True,
    }

    command = _record_train(module, recording, tmp_path, overrides)
    words = shlex.split(command)

    assert _flag_value(command, "--offload-train-target") == "disk"
    assert _flag_value(command, "--offload-train-disk-dir") == "/local_nvme/glm52-offload"
    assert "--stream-optimizer-state-to-disk" in words
    assert "--rematerialize-param-from-master-weight" not in words
