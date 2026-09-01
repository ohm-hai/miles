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
    recording = install_command_recorder(monkeypatch)
    module = import_launch_script(_LAUNCHER)
    monkeypatch.setattr(module, "_collect_ray_cluster_inventory", _valid_cluster_inventory)
    # Keep checkpoint-validation fixtures small. A separate fresh-module test
    # below protects the production layouts from being relaxed accidentally.
    for model_name in _AUDITED_CHECKPOINT_INDEX_LAYOUTS:
        monkeypatch.setitem(module._CHECKPOINT_INDEX_LAYOUTS, model_name, (1, 1, 7))
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
    (checkpoint / "config.json").write_text(json.dumps(config))
    shard_name = "model-00001-of-00001.safetensors"
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 7},
                "weight_map": {"model.embed_tokens.weight": shard_name},
            }
        )
    )
    (checkpoint / shard_name).write_bytes(b"fixture")
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

    with pytest.raises(NotImplementedError, match="qualified only on 1 node x 4 GPUs"):
        module.ScriptArgs(
            hardware="MI355X",
            model_name="GLM-5.2_5layer",
            num_nodes=1,
            num_gpus_per_node=num_gpus,
            run_id="unit-test",
        )


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


@pytest.mark.parametrize("hardware", ["MI350X", "MI355X"])
def test_full_model_accepts_a_homogeneous_cluster_of_the_requested_sku(launcher, monkeypatch, hardware):
    module, _ = launcher
    nodes, probes, available_gpus, driver_node_address = _valid_cluster_inventory()
    for probe in probes:
        probe["product_names"] = [f"AMD Instinct {hardware}"] * 8
    monkeypatch.setattr(
        module,
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
        module,
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
        module,
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
        module.socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (module.socket.AF_INET, module.socket.SOCK_STREAM, 6, "", ("192.0.2.10", 0))
        ],
    )
    args = module.ScriptArgs(**_full_overrides())

    with pytest.raises(RuntimeError, match="resolves outside the inspected cluster"):
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
    monkeypatch.setattr(module, "_sysfs_product_names", lambda: ["AMD GFX950"] * 8)
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: module.subprocess.CompletedProcess(args=args[0], returncode=0, stdout=output),
    )

    product_names, source, error = module._probe_product_names(8)

    assert product_names == ["MI350X"] * 8
    assert source == "amd-smi"
    assert error is None


def test_external_ray_network_bypasses_every_validated_node_and_clears_proxies(launcher, monkeypatch):
    module, _ = launcher
    for name in module._PROXY_ENV_VARS:
        monkeypatch.setenv(name, "http://proxy.invalid:3128")
    args = module.ScriptArgs(**_full_overrides())

    module._require_external_ray(args, "training")

    assert all(name not in module.os.environ for name in module._PROXY_ENV_VARS)
    assert module.os.environ["no_proxy"] == module.os.environ["NO_PROXY"]
    bypass = module.os.environ["no_proxy"].split(",")
    assert "10.0.0.1" in bypass
    assert "10.0.0.8" in bypass


def test_full_model_train_command_uses_the_rocm_bf16_r3_recipe(launcher, tmp_path):
    module, recording = launcher

    command = _record_train(module, recording, tmp_path, _full_overrides())
    words = shlex.split(command)

    assert len(recording.commands) == 2
    assert recording.commands[0].startswith("[multi_node num_nodes=8]")
    assert "validate-artifacts-internal" in recording.commands[0]
    assert recording.commands[1] == command

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


def test_fp8_rollout_is_isolated_from_bf16_actor_training(launcher, tmp_path):
    module, recording = launcher
    overrides = _full_overrides() | {
        "fp8_rollout": True,
        "use_deepep": True,
        "allow_unvalidated_features": True,
    }

    command = _record_train(module, recording, tmp_path, overrides)
    words = shlex.split(command)

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
