"""GLM-5.2 full-parameter GRPO training on AMD MI350X / MI355X.

The five-layer profile is the existing 4-GPU MI355X proof of concept. The full
profile is a guarded first-bring-up configuration for exactly 8 nodes x 8 GPUs:
TP8 / PP4 / DP2 / EP16 / CP1 with pipeline layers 18/20/20/20. That topology
has full-model GB300 evidence, but has not yet been qualified on AMD; follow the
AMD bring-up runbook before treating it as supported.

Every full-profile command requires an already joined, dedicated Ray cluster.
Export ``MILES_SCRIPT_EXTERNAL_RAY=1``, an explicit ``RAY_ADDRESS``, and a
non-loopback ``MASTER_ADDR`` on one of the Ray nodes before conversion, copying,
or training so the launcher validates rather than replaces the 8-node allocation.

Args:
  --model-name: ``GLM-5.2_5layer`` (4-GPU smoke) or ``GLM-5.2`` (64-GPU bring-up).
  --model-revision: Immutable Hugging Face commit; defaults to the audited checkpoint revision.
  --data-revision: Immutable dapo-math-17k commit used by the acceptance recipe.
  --hardware: ``MI350X`` or ``MI355X``; ``auto`` detects the local device.
  --single-node-topology: Keep the five-layer H16 PoC or select the experimental H8 shape.
  --rollout-only: Run load/prefill/decode with lightweight control actors but no trainer model/optimizer.
  --full-model-rollout-only: Select the guarded full-checkpoint 1-node x 8-GPU rollout gate.
  --rollout-probe-mode: Use the default graph path or an eager full-model rollout probe.
  --rollout-probe-capture: Save full-model probe samples to this rollout-ID path template.
  --allow-unvalidated-features: Required for experimental H8, full single-node, FP8, or DeepEP paths.
  --fp8-rollout: Quantize only the SGLang weights to block FP8. Trainer weights stay BF16.
  --enable-optimizer-offload: Move optimizer state to host memory if GPU memory is tight.
  --offload-train-target: Park the colocated trainer in host RAM (``cpu``) or node-local NVMe (``disk``).
  --model-dir: Shared model/checkpoint source. ``--model-local-dir`` is node-local storage.

Examples:
  python scripts/amd/run_glm5_2_744b_a40b.py download --hardware MI355X \
      --model-name GLM-5.2 --num-nodes 8 --num-gpus-per-node 8

  python scripts/amd/run_glm5_2_744b_a40b.py full-train \
      --hardware MI355X --model-name GLM-5.2_5layer --num-gpus-per-node 4

  python scripts/amd/run_glm5_2_744b_a40b.py full-train --hardware MI355X \
      --model-name GLM-5.2 --num-nodes 1 --num-gpus-per-node 8 \
      --full-model-rollout-only --allow-unvalidated-features

  MILES_SCRIPT_EXTERNAL_RAY=1 RAY_ADDRESS=http://10.0.0.1:8265 MASTER_ADDR=10.0.0.1 \
      python scripts/amd/run_glm5_2_744b_a40b.py train \
      --hardware MI355X --model-name GLM-5.2 --num-nodes 8 \
      --num-gpus-per-node 8 --fp8-rollout --allow-unvalidated-features
"""

import json
import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

import typer

import miles.utils.external_utils.command_utils as U
from scripts.amd import _glm5_2_amd_cluster as _cluster
from scripts.amd import _glm5_2_amd_artifacts as _artifacts
from scripts.amd import _glm5_2_amd_profiles as _profiles
from scripts.amd import _glm5_2_amd_spec as _spec
from scripts.amd import _glm5_2_grpo_dataset as _dataset

app = typer.Typer()

_MODEL_NAMES = Literal["GLM-5.2", "GLM-5.2_5layer"]
_FULL_PIPELINE_LAYERS = (18, 20, 20, 20)
_MODEL_REVISIONS = _spec.MODEL_REVISIONS
_DATA_REVISION = _spec.DATA_REVISION
_ARTIFACT_MANIFEST_NAME = ".miles-artifact.json"
_CHECKPOINT_INDEX_LAYOUTS = _spec.CHECKPOINT_INDEX_LAYOUTS
_CRITICAL_CONFIG_VALUES = _spec.CRITICAL_CONFIG_VALUES
_RUNTIME_HF_ASSETS = _spec.RUNTIME_HF_ASSETS
_SOURCE_CONFIG_ASSETS = _spec.SOURCE_CONFIG_ASSETS
_SOURCE_INDEX_ASSETS = _spec.SOURCE_INDEX_ASSETS


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["normal", "debug_minimal"] = "normal"
    run_id: str = U.create_run_id()
    model_org: str = ""
    model_name: _MODEL_NAMES = "GLM-5.2_5layer"
    model_revision: str = ""
    data_revision: str = _DATA_REVISION
    megatron_model_type: str = field(init=False)
    hardware: Literal["auto", "MI350X", "MI355X"] = "auto"
    num_gpus_per_node: int | None = None
    single_node_topology: _profiles.SingleNodeTopology = "poc-h16"
    full_model_rollout_only: bool = False
    _topology: _profiles.Topology = field(init=False, repr=False)

    fp8_rollout: bool = False
    use_deepep: bool = False
    enable_optimizer_offload: bool = False
    offload_train_target: Literal["cpu", "disk"] = "cpu"
    offload_train_disk_dir: str = "/local_nvme/miles_train_offload"
    stream_optimizer_state_to_disk: bool = False
    enable_r3: bool | None = None
    freeze_router: bool | None = None
    freeze_indexer: bool = True
    enable_indexer_replay: bool = False
    enable_mtp: bool = False
    allow_unvalidated_features: bool = False
    rollout_only: bool = False
    rollout_probe_mode: Literal["graph", "eager"] = "graph"
    rollout_probe_capture: str | None = None
    skip_saving: bool = False

    num_rollout: int | None = None
    rollout_max_prompt_len: int | None = None
    rollout_max_response_len: int | None = None
    max_tokens_per_gpu: int | None = None
    sglang_mem_fraction_static: float | None = None
    sglang_cuda_graph_max_bs: int | None = None
    sglang_max_running_requests: int | None = None
    extra_args: str = ""

    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    model_local_dir: str = "/root/local_data"
    megatron_path: str = "/root/Megatron-LM"

    def __post_init__(self):
        self.hardware = U.resolve_hardware(self)
        pruned = _is_pruned(self)
        self._topology = _profiles.resolve(
            model_name=self.model_name,
            single_node_topology=self.single_node_topology,
            full_model_rollout_only=self.full_model_rollout_only,
        )
        self.rollout_only = self.rollout_only or self._topology.rollout_only
        if self.num_gpus_per_node is None:
            self.num_gpus_per_node = self._topology.num_gpus_per_node

        if self.model_name == "GLM-5.2":
            self.model_org = self.model_org or "zai-org"
            self.megatron_model_type = "glm5.2-744B-A40B"
        else:
            self.model_org = self.model_org or "Pinaster"
            self.megatron_model_type = "glm5.2-744B-A40B_5layer"
        self.model_revision = self.model_revision or _MODEL_REVISIONS[self.model_name]

        if pruned:
            self.mode = "debug_minimal"
        if self.num_rollout is None:
            self.num_rollout = 1 if self.rollout_only else 3000
        if self.rollout_max_response_len is None:
            self.rollout_max_response_len = 100 if pruned else (128 if self.rollout_only else 8192)
        if self.rollout_max_prompt_len is None:
            self.rollout_max_prompt_len = 1024 if pruned else 4096
        if self.max_tokens_per_gpu is None:
            self.max_tokens_per_gpu = 2048 if pruned else 16384
        if self.sglang_mem_fraction_static is None:
            self.sglang_mem_fraction_static = 0.70
        if self.sglang_cuda_graph_max_bs is None:
            self.sglang_cuda_graph_max_bs = 32 if pruned else 1
        if self.sglang_max_running_requests is None:
            self.sglang_max_running_requests = 256 if pruned else (8 if self._topology.rollout_only else 32)
        if self.enable_r3 is None:
            self.enable_r3 = not pruned and not self.rollout_only
        if self.freeze_router is None:
            self.freeze_router = not pruned
        if self.rollout_only:
            self.skip_saving = True

        _validate_profile(self)


def _is_pruned(args: ScriptArgs) -> bool:
    return args.model_name == "GLM-5.2_5layer"


def _is_rollout_only(args: ScriptArgs) -> bool:
    return args.rollout_only


def _validate_profile(args: ScriptArgs) -> None:
    shape = (args.num_nodes, args.num_gpus_per_node)
    expected_shape = (args._topology.num_nodes, args._topology.num_gpus_per_node)
    if shape != expected_shape:
        raise NotImplementedError(
            f"The AMD GLM-5.2 {args._topology.name} profile is defined only for "
            f"{expected_shape[0]} node(s) x {expected_shape[1]} GPUs; got {shape[0]} x {shape[1]}."
        )
    if _is_pruned(args):
        if args.enable_mtp:
            raise ValueError("GLM-5.2_5layer does not contain the checkpoint's MTP layer")
    assert args.num_rollout is not None and args.rollout_max_response_len is not None
    _profiles.validate_rollout_contract(
        args._topology,
        rollout_only=args.rollout_only,
        num_rollout=args.num_rollout,
        max_response_len=args.rollout_max_response_len,
        extra_args=args.extra_args,
    )

    if not args.freeze_indexer:
        raise NotImplementedError(
            "The current GLM-5.2 sparse-attention output does not consume the indexer's "
            "selected scores, so this recipe supports only --freeze-indexer."
        )
    for name, revision in (("model_revision", args.model_revision), ("data_revision", args.data_revision)):
        if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision.lower()):
            raise ValueError(f"{name} must be a full 40-character commit SHA, got {revision!r}")
    if args.use_deepep and not args.fp8_rollout:
        raise ValueError("The provisional AMD Mori/DeepEP path requires --fp8-rollout")
    experimental_features = {
        "fp8_rollout": args.fp8_rollout,
        "use_deepep": args.use_deepep,
        "enable_indexer_replay": args.enable_indexer_replay,
        "enable_mtp": args.enable_mtp,
        "train_router": not _is_pruned(args) and not args.freeze_router,
        **(
            {args._topology.experimental_feature: True}
            if args._topology.experimental_feature is not None
            else {}
        ),
    }
    enabled_experiments = [name for name, enabled in experimental_features.items() if enabled]
    if enabled_experiments and not args.allow_unvalidated_features:
        raise NotImplementedError(
            "These AMD GLM-5.2 features have not passed the bring-up gates: "
            f"{', '.join(enabled_experiments)}. Pass --allow-unvalidated-features only in an "
            "explicit experimental run."
        )
    if args.offload_train_target == "disk" and not args.offload_train_disk_dir:
        raise ValueError("offload_train_disk_dir is required for disk trainer offload")
    if args.stream_optimizer_state_to_disk and args.offload_train_target != "disk":
        raise ValueError("stream_optimizer_state_to_disk requires offload_train_target=disk")
    if args.stream_optimizer_state_to_disk and args.enable_optimizer_offload:
        raise ValueError("stream_optimizer_state_to_disk excludes enable_optimizer_offload")
    if args.num_rollout is None or args.num_rollout <= 0:
        raise ValueError("num_rollout must be positive")
    if args.rollout_max_response_len is not None and args.rollout_max_response_len <= 0:
        raise ValueError("rollout_max_response_len must be positive")
    if args.rollout_max_prompt_len is not None and args.rollout_max_prompt_len <= 0:
        raise ValueError("rollout_max_prompt_len must be positive")
    if args.max_tokens_per_gpu is not None and args.max_tokens_per_gpu <= 0:
        raise ValueError("max_tokens_per_gpu must be positive")
    if (
        not _is_pruned(args)
        and args.max_tokens_per_gpu is not None
        and args.rollout_max_response_len is not None
        and args.rollout_max_prompt_len is not None
        and args.max_tokens_per_gpu < args.rollout_max_prompt_len + args.rollout_max_response_len
    ):
        raise ValueError(
            "The full-profile max_tokens_per_gpu packing target must cover "
            "rollout_max_prompt_len + rollout_max_response_len"
        )
    if args.sglang_mem_fraction_static is not None and not 0 < args.sglang_mem_fraction_static < 1:
        raise ValueError("sglang_mem_fraction_static must be between 0 and 1")
    if args.sglang_cuda_graph_max_bs is not None and args.sglang_cuda_graph_max_bs <= 0:
        raise ValueError("sglang_cuda_graph_max_bs must be positive")
    if args.sglang_max_running_requests is not None and args.sglang_max_running_requests <= 0:
        raise ValueError("sglang_max_running_requests must be positive")
    no_train_offload = "--no-offload-train" in _profiles._validate_diagnostic_extra_args(
        args.extra_args
    )
    if args.rollout_only and (
        args.enable_optimizer_offload
        or args.offload_train_target != "cpu"
        or args.stream_optimizer_state_to_disk
    ):
        raise ValueError("Trainer offload options are invalid for a rollout-only profile")
    if no_train_offload and (
        args.offload_train_target != "cpu" or args.stream_optimizer_state_to_disk
    ):
        raise ValueError("--no-offload-train cannot be combined with disk trainer offload")
    full_rollout_probe = args.rollout_only and not _is_pruned(args)
    if args.rollout_probe_mode != "graph" and not full_rollout_probe:
        raise ValueError("rollout_probe_mode is valid only for a full-model rollout-only profile")
    if args.rollout_probe_capture is not None:
        if not args.rollout_probe_capture.strip():
            raise ValueError("rollout_probe_capture must be a nonblank path template")
        if not full_rollout_probe:
            raise ValueError("rollout_probe_capture is valid only for a full-model rollout-only profile")


def _is_index_compute_layer(layer_number: int) -> bool:
    return layer_number <= 3 or (layer_number - 3) % 4 == 0


def _pipeline_stage_starts(args: ScriptArgs) -> tuple[int, ...]:
    return args._topology.stage_starts


_amd_smi_product_names = _cluster._amd_smi_product_names
_probe_product_names = _cluster._probe_product_names
_require_external_ray = _cluster._require_external_ray


def _get_parallel_config(args: ScriptArgs) -> str:
    starts = _pipeline_stage_starts(args)
    assert all(_is_index_compute_layer(layer) for layer in starts), (
        f"Every GLM DSA pipeline stage must start on an index-compute layer; got {starts}"
    )
    return args._topology.megatron_args()


def _expected_indexer_types(num_layers: int) -> list[str]:
    return ["full" if _is_index_compute_layer(layer) else "shared" for layer in range(1, num_layers + 1)]


_nonempty_file_size = _artifacts._nonempty_file_size


def _read_safetensor_index(checkpoint_dir: Path) -> tuple[dict[str, object], set[str]]:
    index_path = checkpoint_dir / "model.safetensors.index.json"
    _nonempty_file_size(index_path)

    with open(index_path) as index_file:
        index = json.load(index_file)
    if not isinstance(index, dict):
        raise RuntimeError(f"{index_path} must contain a JSON object")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError(f"{index_path} must contain a non-empty weight_map")

    invalid_shards = sorted(
        str(shard)
        for shard in weight_map.values()
        if not isinstance(shard, str)
        or Path(shard).name != shard
        or not shard.endswith(".safetensors")
    )
    if invalid_shards:
        raise RuntimeError(f"{index_path} contains unsafe or invalid shard names: {invalid_shards[:3]}")
    shards = set(weight_map.values())
    missing_shards = sorted(shard for shard in shards if not (checkpoint_dir / shard).is_file())
    empty_shards = sorted(
        shard for shard in shards if (checkpoint_dir / shard).is_file() and (checkpoint_dir / shard).stat().st_size == 0
    )
    if missing_shards or empty_shards:
        invalid = missing_shards + empty_shards
        preview = ", ".join(invalid[:3])
        raise FileNotFoundError(
            f"{index_path} references {len(invalid)} missing or empty shard(s), including {preview}"
        )
    return index, shards


def _validate_safetensor_index(
    checkpoint_dir: Path,
    *,
    expected_num_weights: int,
    expected_num_shards: int,
    expected_total_size: int,
) -> None:
    index, shards = _read_safetensor_index(checkpoint_dir)
    weight_map = index["weight_map"]
    metadata = index.get("metadata")
    total_size = metadata.get("total_size") if isinstance(metadata, dict) else None
    actual_layout = (len(weight_map), len(shards), total_size)
    expected_layout = (expected_num_weights, expected_num_shards, expected_total_size)
    if actual_layout != expected_layout:
        raise RuntimeError(
            f"{checkpoint_dir / 'model.safetensors.index.json'} has weight/shard/size layout "
            f"{actual_layout}, expected {expected_layout} "
            "for the pinned checkpoint revision"
        )


def _validate_glm_config(checkpoint_dir: Path, model_name: str) -> dict[str, object]:
    config_path = checkpoint_dir / "config.json"
    _nonempty_file_size(config_path)
    with open(config_path) as config_file:
        config = json.load(config_file)
    if not isinstance(config, dict):
        raise RuntimeError(f"{config_path} must contain a JSON object")

    if "auto_map" in config:
        raise RuntimeError(f"{config_path} must not contain auto_map. Update the checkpoint first.")

    expected_num_layers = 5 if model_name == "GLM-5.2_5layer" else 78
    expected_values = {
        **_CRITICAL_CONFIG_VALUES,
        "num_hidden_layers": expected_num_layers,
        "indexer_types": _expected_indexer_types(expected_num_layers),
        "mlp_layer_types": ["dense"] * 3 + ["sparse"] * (expected_num_layers - 3),
    }
    mismatches = {
        name: {"expected": expected, "actual": config.get(name)}
        for name, expected in expected_values.items()
        if config.get(name) != expected
    }
    if mismatches:
        details = "; ".join(
            f"{name}: expected {values['expected']!r}, got {values['actual']!r}"
            for name, values in mismatches.items()
        )
        raise RuntimeError(f"{config_path} does not match the audited GLM-5.2 architecture: {details}")
    return config


def _validate_glm_checkpoint_at(
    checkpoint_dir: Path,
    *,
    model_name: str,
    expected_manifest: dict[str, object],
) -> None:
    _require_matching_manifest(checkpoint_dir, expected_manifest)
    _validate_glm_config(checkpoint_dir, model_name)
    _artifacts.validate_exact_file(
        checkpoint_dir / "config.json",
        _SOURCE_CONFIG_ASSETS[model_name],
        description="Pinned source config",
    )

    expected_num_weights, expected_num_shards, expected_total_size = _CHECKPOINT_INDEX_LAYOUTS[model_name]
    _validate_safetensor_index(
        checkpoint_dir,
        expected_num_weights=expected_num_weights,
        expected_num_shards=expected_num_shards,
        expected_total_size=expected_total_size,
    )
    _artifacts.validate_exact_file(
        checkpoint_dir / "model.safetensors.index.json",
        _SOURCE_INDEX_ASSETS[model_name],
        description="Pinned source index",
    )


def _validate_glm_checkpoint(args: ScriptArgs) -> None:
    """Reject incomplete, remote-code, or wrong-architecture checkpoints before conversion."""
    _validate_glm_checkpoint_at(
        Path(args.model_dir) / args.model_name,
        model_name=args.model_name,
        expected_manifest=_source_manifest(args),
    )


def _artifact_manifest(path: Path) -> dict[str, object] | None:
    manifest_path = path / _ARTIFACT_MANIFEST_NAME
    if not manifest_path.exists():
        return None
    with open(manifest_path) as manifest_file:
        return json.load(manifest_file)


_validate_torch_dist_references = _artifacts._validate_torch_dist_references


def _artifact_file_inventory(path: Path, manifest: dict[str, object]) -> dict[str, int]:
    artifact = manifest.get("artifact")
    if artifact in {"huggingface-checkpoint", "sglang-block-fp8-checkpoint"}:
        _, shards = _read_safetensor_index(path)
        relative_paths = [Path("config.json"), Path("model.safetensors.index.json")]
        relative_paths.extend(Path(shard) for shard in shards)
        relative_paths.extend(
            Path(relative)
            for relative in _artifacts.runtime_hf_inventory(path, _RUNTIME_HF_ASSETS)
        )
    elif artifact == "megatron-torch-dist-checkpoint":
        tracker = path / "latest_checkpointed_iteration.txt"
        _nonempty_file_size(tracker)
        if tracker.read_text() != "release":
            raise RuntimeError(f"{tracker} must contain exactly 'release'")
        release_dir = path / "release"
        if not release_dir.is_dir():
            raise FileNotFoundError(f"Converted checkpoint payload directory {release_dir} is missing")
        config_path = release_dir / "metadata.json"
        _nonempty_file_size(config_path)
        with open(config_path) as config_file:
            checkpoint_config = json.load(config_file)
        if not isinstance(checkpoint_config, dict) or checkpoint_config.get("sharded_backend") != "torch_dist":
            raise RuntimeError(f"{config_path} does not identify a torch_dist checkpoint")
        _nonempty_file_size(release_dir / ".metadata")
        _validate_torch_dist_references(release_dir)
        relative_paths = [Path("latest_checkpointed_iteration.txt")]
        relative_paths.extend(item.relative_to(path) for item in release_dir.rglob("*") if item.is_file())
        if not any(relative.suffix == ".distcp" for relative in relative_paths):
            raise FileNotFoundError(f"Converted checkpoint payload directory {release_dir} has no .distcp shards")
    else:
        raise RuntimeError(f"Unsupported artifact manifest type {artifact!r} at {path}")

    return {
        relative.as_posix(): _nonempty_file_size(path / relative)
        for relative in sorted(relative_paths, key=lambda item: item.as_posix())
    }


def _write_artifact_manifest(path: Path, manifest: dict[str, object]) -> None:
    if path.is_dir():
        payload = {**manifest, "files": _artifact_file_inventory(path, manifest)}
        manifest_path = path / _ARTIFACT_MANIFEST_NAME
        temporary_path = manifest_path.with_suffix(".tmp")
        temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary_path.replace(manifest_path)


def _require_matching_manifest(path: Path, expected: dict[str, object]) -> None:
    actual = _artifact_manifest(path)
    actual_recipe = dict(actual) if isinstance(actual, dict) else actual
    recorded_files = actual_recipe.pop("files", None) if isinstance(actual_recipe, dict) else None
    if actual_recipe != expected:
        raise RuntimeError(
            f"Refusing to reuse {path}: {_ARTIFACT_MANIFEST_NAME} does not match the requested recipe. "
            f"Expected {expected!r}, got {actual_recipe!r}. Move the stale artifact aside and prepare again."
        )
    actual_files = _artifact_file_inventory(path, expected)
    if recorded_files != actual_files:
        raise RuntimeError(
            f"Refusing to reuse {path}: {_ARTIFACT_MANIFEST_NAME} file inventory does not match "
            "the current artifact. The copy is stale, incomplete, or modified; prepare it again."
        )


def _source_manifest_values(model_org: str, model_name: str, model_revision: str) -> dict[str, object]:
    return {
        "artifact": "huggingface-checkpoint",
        "repository": f"{model_org}/{model_name}",
        "revision": model_revision,
    }


def _source_manifest(args: ScriptArgs) -> dict[str, object]:
    return _source_manifest_values(args.model_org, args.model_name, args.model_revision)


def _fp8_manifest_values(source_manifest: dict[str, object]) -> dict[str, object]:
    return {
        "artifact": "sglang-block-fp8-checkpoint",
        "block_size": [128, 128],
        "source": source_manifest,
        "strategy": "block",
    }


def _fp8_manifest(args: ScriptArgs) -> dict[str, object]:
    return _fp8_manifest_values(_source_manifest(args))


def _torch_dist_manifest_values(model_name: str, source_manifest: dict[str, object]) -> dict[str, object]:
    pruned = model_name == "GLM-5.2_5layer"
    return {
        "artifact": "megatron-torch-dist-checkpoint",
        "expert_model_parallel_size": 1 if pruned else 16,
        "pipeline_layers": [5] if pruned else list(_FULL_PIPELINE_LAYERS),
        "pipeline_model_parallel_size": 1 if pruned else 4,
        "source": source_manifest,
        "tensor_model_parallel_size": 1,
    }


def _torch_dist_manifest(args: ScriptArgs) -> dict[str, object]:
    return _torch_dist_manifest_values(args.model_name, _source_manifest(args))


def _validate_fp8_checkpoint_at(
    checkpoint_dir: Path,
    *,
    model_name: str,
    expected_manifest: dict[str, object],
) -> None:
    _require_matching_manifest(checkpoint_dir, expected_manifest)
    config = _validate_glm_config(checkpoint_dir, model_name)
    expected_quantization = {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
    }
    actual_quantization = config.get("quantization_config")
    if not isinstance(actual_quantization, dict) or any(
        actual_quantization.get(key) != value for key, value in expected_quantization.items()
    ):
        raise RuntimeError(
            f"{checkpoint_dir / 'config.json'} does not describe the required block-FP8 rollout recipe"
        )
    index, shards = _read_safetensor_index(checkpoint_dir)
    expected_shards = _CHECKPOINT_INDEX_LAYOUTS[model_name][1]
    metadata = index.get("metadata")
    total_size = metadata.get("total_size") if isinstance(metadata, dict) else None
    if len(shards) != expected_shards or not isinstance(total_size, int) or total_size <= 0:
        raise RuntimeError(
            f"{checkpoint_dir / 'model.safetensors.index.json'} has {len(shards)} shards and "
            f"total_size={total_size!r}; expected {expected_shards} shards and a positive total_size"
        )


def _validate_artifact_bundle(
    root: Path,
    *,
    model_name: str,
    model_org: str,
    model_revision: str,
    fp8_rollout: bool,
    require_source: bool,
    rollout_only: bool = False,
) -> None:
    source_manifest = _source_manifest_values(model_org, model_name, model_revision)
    if require_source or not fp8_rollout:
        _validate_glm_checkpoint_at(
            root / model_name,
            model_name=model_name,
            expected_manifest=source_manifest,
        )
    if fp8_rollout:
        _validate_fp8_checkpoint_at(
            root / f"{model_name}_fp8",
            model_name=model_name,
            expected_manifest=_fp8_manifest_values(source_manifest),
        )
    if not rollout_only:
        _require_matching_manifest(
            root / f"{model_name}_torch_dist",
            _torch_dist_manifest_values(model_name, source_manifest),
        )


def _artifact_validation_command(args: ScriptArgs, root: str, *, require_source: bool) -> str:
    script = U.repo_base_dir / "scripts/amd/run_glm5_2_744b_a40b.py"
    command = (
        f"python {shlex.quote(str(script))} validate-artifacts-internal "
        f"--root {shlex.quote(root)} --model-name {shlex.quote(args.model_name)} "
        f"--model-org {shlex.quote(args.model_org)} --model-revision {shlex.quote(args.model_revision)}"
    )
    if args.fp8_rollout:
        command += " --fp8-rollout"
    if require_source:
        command += " --require-source"
    if _is_rollout_only(args):
        command += " --rollout-only"
    return command


def _validate_shared_artifacts(args: ScriptArgs) -> None:
    U.exec_command_cpu(_artifact_validation_command(args, args.model_dir, require_source=True))


def _validate_node_local_artifacts(args: ScriptArgs) -> None:
    command = _artifact_validation_command(args, args.model_local_dir, require_source=False)
    if args.num_nodes == 1:
        U.exec_command_cpu(command)
    else:
        U.exec_command_multi_node(command, num_nodes=args.num_nodes)


def _convert_to_fp8(args: ScriptArgs) -> None:
    """Create block-FP8 rollout weights; Megatron training remains BF16."""
    src = f"{args.model_dir}/{args.model_name}"
    dst = f"{args.model_dir}/{args.model_name}_fp8"
    sentinel = Path(dst) / "model.safetensors.index.json"
    if sentinel.exists():
        _validate_fp8_checkpoint_at(
            Path(dst), model_name=args.model_name, expected_manifest=_fp8_manifest(args)
        )
        print(f"_convert_to_fp8 skip {dst} since {sentinel} exists")
        return
    U.exec_command_gpu(
        f"python {U.repo_base_dir}/tools/convert_hf_to_fp8.py "
        f"--model-dir {src} --save-dir {dst} "
        "--strategy block --block-size 128 128 --max-workers 16"
    )
    _write_artifact_manifest(Path(dst), _fp8_manifest(args))
    if Path(dst).is_dir():
        _validate_fp8_checkpoint_at(
            Path(dst), model_name=args.model_name, expected_manifest=_fp8_manifest(args)
        )


def _prepare_download(args: ScriptArgs) -> None:
    U.exec_command_cpu(f"mkdir -p {args.model_dir} {args.data_dir}")
    U.exec_command_cpu(
        f"hf download {args.model_org}/{args.model_name} --revision {args.model_revision} "
        f"--local-dir {args.model_dir}/{args.model_name}"
    )
    U.hf_download_dataset(
        "zhuzilin/dapo-math-17k",
        data_dir=args.data_dir,
        revision=args.data_revision,
    )
    _dataset._validate_after_download(args)
    _write_artifact_manifest(Path(args.model_dir) / args.model_name, _source_manifest(args))


def _prepare_megatron_ckpt(args: ScriptArgs) -> None:
    if _is_rollout_only(args):
        print("Skip Megatron checkpoint conversion for the rollout-only profile")
        return
    if _is_pruned(args):
        extra_args = (
            "--tensor-model-parallel-size 1 "
            "--pipeline-model-parallel-size 1 "
            "--expert-model-parallel-size 1 "
            "--expert-tensor-parallel-size 1 "
        )
        num_gpus_per_node = 1
        multinode = False
        num_nodes = None
    else:
        _require_external_ray(args, "checkpoint conversion")
        extra_args = (
            "--tensor-model-parallel-size 1 "
            "--pipeline-model-parallel-size 4 "
            "--decoder-first-pipeline-num-layers 18 "
            "--decoder-last-pipeline-num-layers 20 "
            "--expert-model-parallel-size 16 "
            "--expert-tensor-parallel-size 1 "
        )
        num_gpus_per_node = args.num_gpus_per_node
        multinode = True
        num_nodes = args.num_nodes

    torch_dist_dir = Path(args.model_dir) / f"{args.model_name}_torch_dist"
    tracker = torch_dist_dir / "latest_checkpointed_iteration.txt"
    expected_manifest = _torch_dist_manifest(args)
    if tracker.exists() and tracker.read_text().strip() == "release":
        _require_matching_manifest(torch_dist_dir, expected_manifest)

    U.convert_checkpoint(
        model_name=args.model_name,
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=num_gpus_per_node,
        multinode=multinode,
        num_nodes=num_nodes,
        extra_args=extra_args,
        dir_dst=args.model_dir,
        hf_checkpoint=f"{args.model_dir}/{args.model_name}",
        megatron_path=args.megatron_path,
    )
    if tracker.exists() and tracker.read_text().strip() == "release":
        _write_artifact_manifest(torch_dist_dir, expected_manifest)


def _prepare_cp(args: ScriptArgs, *, skip_existing: bool = False) -> None:
    _require_external_ray(args, "checkpoint copy")
    _validate_shared_artifacts(args)

    def copy_checkpoint(path_src: str, path_dst: str) -> None:
        if args.num_nodes == 1:
            U.exec_command_cpu(f"mkdir -p {path_dst} && rsync -a --info=progress2 {path_src}/ {path_dst}")
        else:
            U.rsync_simple(path_src=path_src, path_dst=path_dst, num_nodes=args.num_nodes)

    may_skip_local_copy = skip_existing and args.num_nodes == 1
    if not _is_rollout_only(args):
        torch_dist_dst = f"{args.model_local_dir}/{args.model_name}_torch_dist"
        torch_dist_sentinel = Path(torch_dist_dst) / "latest_checkpointed_iteration.txt"
        if not (may_skip_local_copy and torch_dist_sentinel.exists()):
            copy_checkpoint(f"{args.model_dir}/{args.model_name}_torch_dist", torch_dist_dst)

    hf_name = f"{args.model_name}_fp8" if args.fp8_rollout else args.model_name
    hf_dst = f"{args.model_local_dir}/{hf_name}"
    hf_sentinel = Path(hf_dst) / "model.safetensors.index.json"
    if not (may_skip_local_copy and hf_sentinel.exists()):
        copy_checkpoint(f"{args.model_dir}/{hf_name}", hf_dst)
    _validate_node_local_artifacts(args)


def _checkpoint_args(args: ScriptArgs) -> str:
    hf_name = f"{args.model_name}_fp8" if args.fp8_rollout else args.model_name
    checkpoint = f"--hf-checkpoint {args.model_local_dir}/{hf_name} "
    if not _is_rollout_only(args):
        checkpoint += f"--ref-load {args.model_local_dir}/{args.model_name}_torch_dist "
    if not args.skip_saving:
        load_save_path = f"{args.output_dir}/{args.run_id}/checkpoints"
        checkpoint += (
            f"--load {load_save_path} --save {load_save_path} "
            "--save-interval 20 --save-retain-interval 20 "
        )
    return checkpoint


def _rollout_args(args: ScriptArgs) -> str:
    assert args.rollout_max_prompt_len is not None and args.rollout_max_response_len is not None
    full_rollout = _profiles.full_model_rollout_args(
        args._topology,
        rollout_only=args.rollout_only,
        data_dir=args.data_dir,
        max_prompt_len=args.rollout_max_prompt_len,
        max_response_len=args.rollout_max_response_len,
    )
    if full_rollout is not None:
        return full_rollout
    return (
        f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt --label-key label --apply-chat-template --rollout-shuffle "
        "--rm-type deepscaler "
        f"--num-rollout {args.num_rollout} --rollout-batch-size 8 --n-samples-per-prompt 8 "
        f"--rollout-max-prompt-len {args.rollout_max_prompt_len} "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
        "--rollout-temperature 1 --global-batch-size 64 --balance-data "
    )


def _performance_args(args: ScriptArgs) -> str:
    return _get_parallel_config(args) + (
        "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {args.max_tokens_per_gpu} "
        "--data-pad-size-multiplier 1024 --log-probs-chunk-size 16384 "
    )


def _optimizer_args(args: ScriptArgs) -> str:
    optimizer = (
        "--optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 "
        "--adam-beta1 0.9 --adam-beta2 0.98 "
    )
    if args.enable_optimizer_offload:
        optimizer += "--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer "
    return optimizer


def _sglang_args(args: ScriptArgs) -> str:
    world_size = args.num_gpus_per_node
    router_policy = "round_robin" if args.rollout_only and not _is_pruned(args) else "consistent_hashing"
    sglang = (
        f"--rollout-num-gpus-per-engine {world_size} "
        f"--sglang-tp-size {world_size} --sglang-dp-size 1 --sglang-ep-size {world_size} "
        f"--sglang-mem-fraction-static {args.sglang_mem_fraction_static} "
        f"--sglang-router-policy {router_policy} "
    )
    if args.rollout_probe_mode == "eager":
        sglang += "--sglang-disable-cuda-graph "
    if args.fp8_rollout and args.use_deepep:
        sglang += "--sglang-moe-a2a-backend mori --sglang-deepep-mode auto "
    if args.enable_mtp:
        sglang += (
            "--sglang-speculative-algorithm EAGLE --sglang-speculative-num-steps 1 "
            "--sglang-speculative-eagle-topk 1 --sglang-speculative-num-draft-tokens 2 "
            "--sglang-speculative-draft-attention-backend nsa "
        )
    return sglang + (
        "--sglang-kv-cache-dtype fp8_e4m3 "
        "--sglang-nsa-decode-backend tilelang --sglang-nsa-prefill-backend tilelang "
        "--sglang-attention-backend nsa --sglang-page-size 64 "
        f"--sglang-cuda-graph-max-bs {args.sglang_cuda_graph_max_bs} "
        f"--sglang-max-running-requests {args.sglang_max_running_requests} "
        f"--sglang-chunked-prefill-size {2048 * world_size} "
        "--sglang-watchdog-timeout 3600 "
    )


def _misc_args(args: ScriptArgs) -> str:
    misc = (
        "--bf16 --transformer-impl transformer_engine "
        "--attention-dropout 0.0 --hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 "
        "--attention-backend flash --allgather-cp "
        f"--update-weight-buffer-size {2 * 1024**3} "
        f"--actor-num-nodes {args.num_nodes} --actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        "--colocate "
        "--moe-token-dispatcher-type alltoall "
        "--train-memory-margin-bytes 4294967296 "
        "--rollout-health-check-interval 300 --rollout-health-check-timeout 300 "
    )
    if not _is_rollout_only(args):
        no_train_offload = "--no-offload-train" in _profiles._validate_diagnostic_extra_args(
            args.extra_args
        )
        if args.offload_train_target == "cpu":
            if not no_train_offload:
                misc += "--rematerialize-param-from-master-weight "
        else:
            misc += (
                f"--offload-train-target disk --offload-train-disk-dir {args.offload_train_disk_dir} "
            )
            if args.stream_optimizer_state_to_disk:
                misc += "--stream-optimizer-state-to-disk "
    if args.freeze_indexer:
        misc += "--freeze-indexer "
    if args.freeze_router:
        misc += "--moe-router-freeze-gate --freeze-e-score-correction-bias "
    if args.enable_r3:
        misc += "--use-rollout-routing-replay "
    if args.enable_indexer_replay:
        misc += "--use-rollout-indexer-replay "
    if _is_rollout_only(args):
        misc += "--debug-rollout-only "
    if args.rollout_probe_capture is not None:
        misc += f"--save-debug-rollout-data {shlex.quote(args.rollout_probe_capture)} "
    return misc


def _configure_rocm_process_env() -> None:
    # Ray's coordinator processes still import torch and must see the physical
    # devices before Miles assigns per-actor visibility.
    os.environ.setdefault("RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES", "1")
    os.environ.setdefault("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", "1")
    if hip_visible_devices := os.environ.get("HIP_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = hip_visible_devices
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")


def _execute_train(args: ScriptArgs) -> None:
    _require_external_ray(args, "training")
    _cluster._validate_local_hardware(args)
    _dataset._validate_before_train(args)
    _validate_node_local_artifacts(args)
    _configure_rocm_process_env()

    grpo_args = (
        "--advantage-estimator grpo --kl-loss-coef 0.00 --kl-loss-type low_var_kl "
        "--kl-coef 0.00 --entropy-coef 0.00 --eps-clip 0.2 --eps-clip-high 0.28 "
        "--use-tis --tis-clip-low 0.5 --tis-clip 2.0 "
    )
    train_args = (
        f"{_checkpoint_args(args)} "
        f"{_rollout_args(args)} "
        f"{_optimizer_args(args)} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__, run_id=args.run_id)} "
        f"{_performance_args(args)} "
        f"{_sglang_args(args)} "
        f"{_misc_args(args)} "
        f"{args.extra_args} "
    ).strip()
    extra_env_vars = {
        "MILES_HARDWARE_PLATFORM": "rocm",
        "SGLANG_NSA_FORCE_MLA": "1",
        "INDEXER_ROPE_NEOX_STYLE": "0",
        "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES": "1",
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
    }
    if args.enable_mtp:
        extra_env_vars["SGLANG_USE_AITER_AG"] = "false"

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        extra_env_vars=extra_env_vars,
        megatron_path=args.megatron_path,
    )


@app.command("validate-artifacts-internal", hidden=True)
def _validate_artifacts_internal(
    root: Annotated[Path, typer.Option()],
    model_name: Annotated[str, typer.Option()],
    model_org: Annotated[str, typer.Option()],
    model_revision: Annotated[str, typer.Option()],
    fp8_rollout: bool = False,
    require_source: bool = False,
    rollout_only: bool = False,
) -> None:
    """Validate one host's prepared artifacts; invoked synchronously by the public commands."""
    if model_name not in _CHECKPOINT_INDEX_LAYOUTS:
        raise ValueError(f"Unsupported GLM checkpoint {model_name!r}")
    _validate_artifact_bundle(
        root,
        model_name=model_name,
        model_org=model_org,
        model_revision=model_revision,
        fp8_rollout=fp8_rollout,
        require_source=require_source,
        rollout_only=rollout_only,
    )


@app.command()
@U.dataclass_cli
def full_train(args: ScriptArgs) -> None:
    """Download, convert, copy to node-local storage, and train."""
    _require_external_ray(args, "full training")
    _cluster._validate_local_hardware(args)
    _prepare_download(args)
    _validate_glm_checkpoint(args)
    if args.fp8_rollout:
        _convert_to_fp8(args)
    _prepare_megatron_ckpt(args)
    _prepare_cp(args, skip_existing=True)
    _execute_train(args)


@app.command()
@U.dataclass_cli
def prepare(args: ScriptArgs) -> None:
    """Download and convert on the shared filesystem."""
    _require_external_ray(args, "preparation")
    if args.fp8_rollout or not _is_rollout_only(args):
        _cluster._validate_local_hardware(args)
    _prepare_download(args)
    _validate_glm_checkpoint(args)
    if args.fp8_rollout:
        _convert_to_fp8(args)
    _prepare_megatron_ckpt(args)


@app.command()
@U.dataclass_cli
def download(args: ScriptArgs) -> None:
    """Download and validate pinned model/data artifacts without requiring Ray or GPUs."""
    _prepare_download(args)
    _validate_glm_checkpoint(args)


@app.command()
@U.dataclass_cli
def prepare_cp(args: ScriptArgs) -> None:
    """Copy trainer and rollout checkpoints to every node's local storage."""
    _prepare_cp(args)


@app.command()
@U.dataclass_cli
def train(args: ScriptArgs) -> None:
    """Run training after prepare and prepare-cp have completed."""
    _execute_train(args)


@app.callback()
def _callback() -> None:
    pass


if __name__ == "__main__":
    app()
