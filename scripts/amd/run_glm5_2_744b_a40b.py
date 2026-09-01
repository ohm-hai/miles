"""GLM-5.2 full-parameter GRPO training on AMD MI350X / MI355X.

The five-layer profile is the existing 4-GPU MI355X proof of concept. The full
profile is a guarded first-bring-up configuration for exactly 8 nodes x 8 GPUs:
TP8 / PP4 / DP2 / EP16 / CP1 with pipeline layers 18/20/20/20. That topology
has full-model GB300 evidence, but has not yet been qualified on AMD; follow the
AMD bring-up runbook before treating it as supported.

Every full-profile command requires an already joined, dedicated Ray cluster.
Export ``MILES_SCRIPT_EXTERNAL_RAY=1`` and an explicit ``RAY_ADDRESS`` before
conversion, copying, or training so the launcher validates rather than replaces
the 8-node allocation.

Args:
  --model-name: ``GLM-5.2_5layer`` (4-GPU smoke) or ``GLM-5.2`` (64-GPU bring-up).
  --model-revision: Immutable Hugging Face commit; defaults to the audited checkpoint revision.
  --data-revision: Immutable dapo-math-17k commit used by the acceptance recipe.
  --hardware: ``MI350X`` or ``MI355X``; ``auto`` detects the local device.
  --fp8-rollout: Quantize only the SGLang weights to block FP8. Trainer weights stay BF16.
  --enable-optimizer-offload: Move optimizer state to host memory if GPU memory is tight.
  --offload-train-target: Park the colocated trainer in host RAM (``cpu``) or node-local NVMe (``disk``).
  --model-dir: Shared model/checkpoint source. ``--model-local-dir`` is node-local storage.

Examples:
  python scripts/amd/run_glm5_2_744b_a40b.py full-train \
      --hardware MI355X --model-name GLM-5.2_5layer --num-gpus-per-node 4

  MILES_SCRIPT_EXTERNAL_RAY=1 RAY_ADDRESS=http://10.0.0.1:8265 \
      python scripts/amd/run_glm5_2_744b_a40b.py train \
      --hardware MI355X --model-name GLM-5.2 --num-nodes 8 \
      --num-gpus-per-node 8 --fp8-rollout
"""

import json
import os
import re
import shlex
import socket
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlparse

import ray
import typer
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

import miles.utils.external_utils.command_utils as U

app = typer.Typer()

_MODEL_NAMES = Literal["GLM-5.2", "GLM-5.2_5layer"]
_FULL_PIPELINE_LAYERS = (18, 20, 20, 20)
_MODEL_REVISIONS = {
    "GLM-5.2": "b4734de4facf877f85769a911abafc5283eab3d9",
    "GLM-5.2_5layer": "1c749139f70e158e4420ba67f342bef1de2e650d",
}
_DATA_REVISION = "2e65612930298bde4c5d58fd97b3f23a483aaff9"
_EXPECTED_FULL_NUM_NODES = 8
_EXPECTED_GPUS_PER_NODE = 8
_PROXY_ENV_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
_ARTIFACT_MANIFEST_NAME = ".miles-artifact.json"
_CHECKPOINT_INDEX_LAYOUTS = {
    # (number of weight entries, number of shard files, tensor payload bytes)
    "GLM-5.2": (59585, 282, 1506659919872),
    "GLM-5.2_5layer": (1618, 14, 45683868160),
}
_CRITICAL_CONFIG_VALUES = {
    "architectures": ["GlmMoeDsaForCausalLM"],
    "dtype": "bfloat16",
    "first_k_dense_replace": 3,
    "head_dim": 192,
    "hidden_size": 6144,
    "index_head_dim": 128,
    "index_n_heads": 32,
    "index_skip_topk_offset": 3,
    "index_topk": 2048,
    "index_topk_freq": 4,
    "indexer_rope_interleave": True,
    "intermediate_size": 12288,
    "kv_lora_rank": 512,
    "model_type": "glm_moe_dsa",
    "moe_intermediate_size": 2048,
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "num_attention_heads": 64,
    "num_experts_per_tok": 8,
    "num_nextn_predict_layers": 1,
    "q_lora_rank": 2048,
    "qk_head_dim": 256,
    "qk_nope_head_dim": 192,
    "qk_rope_head_dim": 64,
    "rope_parameters": {"rope_theta": 8000000, "rope_type": "default"},
    "routed_scaling_factor": 2.5,
    "scoring_func": "sigmoid",
    "topk_method": "noaux_tc",
    "v_head_dim": 256,
    "vocab_size": 154880,
}


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
    skip_saving: bool = False

    num_rollout: int = 3000
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
        if self.num_gpus_per_node is None:
            # Four GPUs is the only AMD-validated five-layer shape. The full
            # profile uses every GPU in an eight-GPU MI35x node.
            self.num_gpus_per_node = 4 if pruned else U.NUM_GPUS_OF_HARDWARE[self.hardware]

        if self.model_name == "GLM-5.2":
            self.model_org = self.model_org or "zai-org"
            self.megatron_model_type = "glm5.2-744B-A40B"
        else:
            self.model_org = self.model_org or "Pinaster"
            self.megatron_model_type = "glm5.2-744B-A40B_5layer"
        self.model_revision = self.model_revision or _MODEL_REVISIONS[self.model_name]

        if pruned:
            self.mode = "debug_minimal"
        if self.rollout_max_response_len is None:
            self.rollout_max_response_len = 100 if pruned else 8192
        if self.rollout_max_prompt_len is None:
            self.rollout_max_prompt_len = 1024 if pruned else 4096
        if self.max_tokens_per_gpu is None:
            self.max_tokens_per_gpu = 2048 if pruned else 16384
        if self.sglang_mem_fraction_static is None:
            self.sglang_mem_fraction_static = 0.70
        if self.sglang_cuda_graph_max_bs is None:
            self.sglang_cuda_graph_max_bs = 32 if pruned else 1
        if self.sglang_max_running_requests is None:
            self.sglang_max_running_requests = 256 if pruned else 32
        if self.enable_r3 is None:
            self.enable_r3 = not pruned
        if self.freeze_router is None:
            self.freeze_router = not pruned

        _validate_profile(self)


def _is_pruned(args: ScriptArgs) -> bool:
    return args.model_name == "GLM-5.2_5layer"


def _validate_profile(args: ScriptArgs) -> None:
    shape = (args.num_nodes, args.num_gpus_per_node)
    if _is_pruned(args):
        if shape != (1, 4):
            raise NotImplementedError(
                "The AMD GLM-5.2 five-layer profile is qualified only on 1 node x 4 GPUs; "
                f"got {shape[0]} x {shape[1]}."
            )
        if args.enable_mtp:
            raise ValueError("GLM-5.2_5layer does not contain the checkpoint's MTP layer")
    elif shape != (8, 8):
        raise NotImplementedError(
            "The AMD full GLM-5.2 bring-up profile is defined only for 8 nodes x 8 GPUs "
            f"(64 total); got {shape[0]} x {shape[1]}."
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
    if args.num_rollout <= 0:
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


def _is_index_compute_layer(layer_number: int) -> bool:
    return layer_number <= 3 or (layer_number - 3) % 4 == 0


def _pipeline_stage_starts(args: ScriptArgs) -> tuple[int, ...]:
    if _is_pruned(args):
        return (1,)
    starts = [1]
    for num_layers in _FULL_PIPELINE_LAYERS[:-1]:
        starts.append(starts[-1] + num_layers)
    return tuple(starts)


def _sysfs_product_names() -> list[str]:
    product_names = []
    for path in sorted(Path("/sys/class/drm").glob("card*/device/product_name")):
        if re.fullmatch(r"card\d+", path.parents[1].name):
            try:
                product_names.append(path.read_text().strip())
            except OSError:
                continue
    return [name for name in product_names if name]


def _find_named_values(value: object, names: set[str]) -> list[str]:
    if isinstance(value, dict):
        found = [str(item) for key, item in value.items() if key.lower() in names and isinstance(item, str)]
        for item in value.values():
            found.extend(_find_named_values(item, names))
        return found
    if isinstance(value, list):
        return [found for item in value for found in _find_named_values(item, names)]
    return []


def _amd_smi_product_names(output: str) -> list[str]:
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        return re.findall(r"^\s*(?:MARKET_NAME|PRODUCT_NAME):\s*(.+?)\s*$", output, flags=re.MULTILINE)

    if isinstance(payload, dict):
        gpu_data = payload.get("gpu_data")
        records = gpu_data if isinstance(gpu_data, list) else [payload]
    else:
        records = payload
    records = records if isinstance(records, list) else [records]
    product_names = []
    for record in records:
        market_names = _find_named_values(record, {"market_name"})
        fallback_names = _find_named_values(record, {"product_name"})
        candidates = market_names or fallback_names
        matching_name = next((name for name in candidates if re.search(r"\bMI3(?:50|55)X\b", name, re.I)), None)
        if matching_name:
            product_names.append(matching_name)
    return product_names


def _product_sku(product_name: object) -> str | None:
    match = re.search(r"\b(MI350X|MI355X)\b", str(product_name), re.I)
    return match.group(1).upper() if match else None


def _probe_product_names(expected_count: int) -> tuple[list[str], str, str | None]:
    sysfs_names = _sysfs_product_names()
    if len(sysfs_names) == expected_count and all(_product_sku(name) for name in sysfs_names):
        return sysfs_names, "sysfs", None

    try:
        result = subprocess.run(
            ["amd-smi", "static", "--asic", "--json"],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return sysfs_names, "sysfs-incomplete", repr(error)

    amd_smi_names = _amd_smi_product_names(result.stdout)
    error = None if result.returncode == 0 else f"amd-smi exited with {result.returncode}"
    return amd_smi_names, "amd-smi", error


def _probe_mi35x_node() -> dict[str, object]:
    """Inspect one Ray node without allocating a GPU or invoking a shell."""
    try:
        result = subprocess.run(
            ["rocminfo"],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {
            "node_id": str(ray.get_runtime_context().get_node_id()),
            "hostname": socket.gethostname(),
            "gfx950_count": 0,
            "product_names": [],
            "product_source": None,
            "error": repr(error),
        }

    gfx950_count = len(re.findall(r"^\s*Name:\s+gfx950\s*$", result.stdout, flags=re.MULTILINE))
    product_names, product_source, product_error = _probe_product_names(gfx950_count)
    errors = []
    if result.returncode != 0:
        errors.append(f"rocminfo exited with {result.returncode}")
    if product_error:
        errors.append(product_error)
    return {
        "node_id": str(ray.get_runtime_context().get_node_id()),
        "hostname": socket.gethostname(),
        "gfx950_count": gfx950_count,
        "product_names": product_names,
        "product_source": product_source,
        "error": "; ".join(errors) or None,
    }


def _collect_ray_cluster_inventory() -> tuple[
    list[dict[str, object]], list[dict[str, object]], float, str
]:
    """Collect immutable node metadata plus a hard-affinity hardware probe per node."""
    initialized_here = not ray.is_initialized()
    if initialized_here:
        # Multi-node preparation already relies on address="auto" in
        # exec_command_multi_node. Requiring the same local cluster discovery
        # here prevents inspecting one cluster and submitting to another.
        ray.init(address="auto", log_to_driver=False)

    try:
        nodes = sorted(
            (node for node in ray.nodes() if node.get("Alive")),
            key=lambda node: str(node["NodeManagerAddress"]),
        )
        remote_probe = ray.remote(num_cpus=0)(_probe_mi35x_node)
        probe_refs = [
            remote_probe.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=str(node["NodeID"]), soft=False)
            ).remote()
            for node in nodes
        ]
        probes = ray.get(probe_refs, timeout=120)
        available_gpus = float(ray.available_resources().get("GPU", 0))
        return nodes, probes, available_gpus, ray.util.get_node_ip_address()
    finally:
        if initialized_here:
            ray.shutdown()


def _ray_address_host(ray_address: str) -> str | None:
    parsed = urlparse(ray_address)
    return parsed.hostname if parsed.scheme in {"http", "https"} else None


def _resolved_dashboard_addresses(host: str, driver_node_address: str) -> set[str]:
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)}
    except socket.gaierror:
        return set()
    if host == "localhost" or addresses & {"127.0.0.1", "::1"}:
        addresses.add(driver_node_address)
    return addresses


def _configure_external_ray_network(node_addresses: tuple[str, ...] = ()) -> None:
    """Keep control-plane and collectives traffic out of HTTP proxy paths."""
    for name in _PROXY_ENV_VARS:
        os.environ.pop(name, None)

    ray_host = _ray_address_host(os.environ.get("RAY_ADDRESS", ""))
    configured = os.environ.get("no_proxy", "") + "," + os.environ.get("NO_PROXY", "")
    bypass = ["127.0.0.1", "localhost"]
    bypass.extend(entry.strip() for entry in configured.split(",") if entry.strip())
    bypass.extend(address for address in (ray_host, *node_addresses) if address)
    value = ",".join(dict.fromkeys(bypass))
    os.environ["no_proxy"] = value
    os.environ["NO_PROXY"] = value


def _validate_external_ray_cluster(args: ScriptArgs) -> tuple[str, ...]:
    try:
        nodes, probes, available_gpus, driver_node_address = _collect_ray_cluster_inventory()
    except Exception as error:
        raise RuntimeError(
            "Could not inspect the external Ray cluster. Run the launcher on a joined Ray node "
            "where ray.init(address='auto') reaches the same cluster as RAY_ADDRESS."
        ) from error

    assert args.num_nodes == _EXPECTED_FULL_NUM_NODES
    assert args.num_gpus_per_node == _EXPECTED_GPUS_PER_NODE
    expected_total_gpus = args.num_nodes * args.num_gpus_per_node
    errors = []
    if len(nodes) != args.num_nodes:
        errors.append(f"expected exactly {args.num_nodes} alive nodes, found {len(nodes)}")
    if available_gpus != expected_total_gpus:
        errors.append(f"expected {expected_total_gpus} available GPUs, found {available_gpus:g}")

    probes_by_node = {str(probe["node_id"]): probe for probe in probes}
    records = []
    product_skus = []
    for node in nodes:
        node_id = str(node["NodeID"])
        address = str(node["NodeManagerAddress"])
        gpu_resources = float(node.get("Resources", {}).get("GPU", 0))
        probe = probes_by_node.get(node_id)
        if gpu_resources != args.num_gpus_per_node:
            errors.append(
                f"node {address} advertises {gpu_resources:g} GPUs instead of {args.num_gpus_per_node}"
            )
        if probe is None:
            errors.append(f"node {address} did not return its hard-affinity hardware probe")
            continue
        if probe.get("error"):
            errors.append(f"node {address} hardware probe failed: {probe['error']}")
        if probe.get("gfx950_count") != args.num_gpus_per_node:
            errors.append(
                f"node {address} reported {probe.get('gfx950_count', 0)} gfx950 agents "
                f"instead of {args.num_gpus_per_node}"
            )
        product_names = probe.get("product_names")
        if not isinstance(product_names, list) or len(product_names) != args.num_gpus_per_node:
            errors.append(
                f"node {address} reported {len(product_names) if isinstance(product_names, list) else 0} "
                f"GPU product names instead of {args.num_gpus_per_node}"
            )
            product_names = []
        normalized_skus = [sku for product_name in product_names if (sku := _product_sku(product_name))]
        if normalized_skus != [args.hardware] * args.num_gpus_per_node:
            errors.append(
                f"node {address} product names do not match requested {args.hardware}: {product_names}"
            )
        product_skus.extend(normalized_skus)
        records.append(
            {
                "node_id": node_id,
                "address": address,
                "hostname": probe.get("hostname"),
                "gpu_resources": gpu_resources,
                "gfx950_agents": probe.get("gfx950_count"),
                "product_names": product_names,
                "product_source": probe.get("product_source"),
            }
        )

    unexpected_probe_ids = set(probes_by_node) - {str(node["NodeID"]) for node in nodes}
    if unexpected_probe_ids:
        errors.append(f"hardware probes returned unexpected node IDs: {sorted(unexpected_probe_ids)}")
    if set(product_skus) != {args.hardware}:
        errors.append(f"cluster GPU products are not homogeneous {args.hardware}: {sorted(set(product_skus))}")

    node_addresses = tuple(str(node["NodeManagerAddress"]) for node in nodes)
    ray_host = _ray_address_host(os.environ["RAY_ADDRESS"])
    if ray_host is None:
        errors.append("RAY_ADDRESS must be an HTTP(S) Ray dashboard URL with an explicit host")
    else:
        resolved_dashboard = _resolved_dashboard_addresses(ray_host, driver_node_address)
        if resolved_dashboard and not resolved_dashboard.intersection(node_addresses):
            errors.append(
                f"RAY_ADDRESS host {ray_host!r} resolves outside the inspected cluster: "
                f"{sorted(resolved_dashboard)}"
            )
    if errors:
        raise RuntimeError(
            f"External Ray cluster is not a dedicated homogeneous 8x8 {args.hardware} allocation: "
            + "; ".join(errors)
        )

    print("Validated dedicated GLM-5.2 Ray cluster: " + json.dumps(records, sort_keys=True))
    return node_addresses


def _require_external_ray(args: ScriptArgs, action: str) -> None:
    if args.num_nodes <= 1:
        return
    if not U.get_bool_env_var("MILES_SCRIPT_EXTERNAL_RAY"):
        raise RuntimeError(
            f"Multi-node GLM-5.2 {action} requires an already joined Ray cluster. "
            "Export MILES_SCRIPT_EXTERNAL_RAY=1 first."
        )
    if not os.environ.get("RAY_ADDRESS", "").strip():
        raise RuntimeError(
            f"Multi-node GLM-5.2 {action} requires an explicit RAY_ADDRESS for the existing cluster."
        )
    if _ray_address_host(os.environ["RAY_ADDRESS"]) is None:
        raise RuntimeError(
            f"Multi-node GLM-5.2 {action} requires RAY_ADDRESS to be an HTTP(S) Ray dashboard URL."
        )

    _configure_external_ray_network()
    node_addresses = _validate_external_ray_cluster(args)
    _configure_external_ray_network(node_addresses)


def _get_parallel_config(args: ScriptArgs) -> str:
    starts = _pipeline_stage_starts(args)
    assert all(_is_index_compute_layer(layer) for layer in starts), (
        f"Every GLM DSA pipeline stage must start on an index-compute layer; got {starts}"
    )
    if _is_pruned(args):
        return (
            "--tensor-model-parallel-size 4 "
            "--sequence-parallel "
            "--pipeline-model-parallel-size 1 "
            "--context-parallel-size 1 "
            "--expert-model-parallel-size 4 "
            "--expert-tensor-parallel-size 1 "
        )
    return (
        "--tensor-model-parallel-size 8 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 4 "
        "--decoder-first-pipeline-num-layers 18 "
        "--decoder-last-pipeline-num-layers 20 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 16 "
        "--expert-tensor-parallel-size 1 "
    )


def _expected_indexer_types(num_layers: int) -> list[str]:
    return ["full" if _is_index_compute_layer(layer) else "shared" for layer in range(1, num_layers + 1)]


def _nonempty_file_size(path: Path) -> int:
    if not path.is_file() or (size := path.stat().st_size) <= 0:
        raise FileNotFoundError(f"Required artifact file {path} is missing or empty")
    return size


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

    expected_num_weights, expected_num_shards, expected_total_size = _CHECKPOINT_INDEX_LAYOUTS[model_name]
    _validate_safetensor_index(
        checkpoint_dir,
        expected_num_weights=expected_num_weights,
        expected_num_shards=expected_num_shards,
        expected_total_size=expected_total_size,
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


def _validate_torch_dist_references(release_dir: Path) -> None:
    # Import only in the internal validator: torch.distributed.checkpoint is
    # expensive and every normal launcher invocation should remain lightweight.
    from torch.distributed.checkpoint import FileSystemReader

    try:
        metadata = FileSystemReader(release_dir).read_metadata()
    except Exception as error:
        raise RuntimeError(f"Could not read torch_dist metadata from {release_dir}") from error
    storage_data = getattr(metadata, "storage_data", None)
    if not isinstance(storage_data, dict) or not storage_data:
        raise RuntimeError(f"{release_dir / '.metadata'} contains no torch_dist storage records")

    required_sizes: dict[Path, int] = {}
    for storage in storage_data.values():
        relative = Path(str(getattr(storage, "relative_path", "")))
        offset = getattr(storage, "offset", None)
        length = getattr(storage, "length", None)
        if (
            not relative.name
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.suffix != ".distcp"
            or not isinstance(offset, int)
            or not isinstance(length, int)
            or offset < 0
            or length <= 0
        ):
            raise RuntimeError(f"{release_dir / '.metadata'} contains an invalid storage record {storage!r}")
        required_sizes[relative] = max(required_sizes.get(relative, 0), offset + length)

    for relative, required_size in required_sizes.items():
        actual_size = _nonempty_file_size(release_dir / relative)
        if actual_size < required_size:
            raise RuntimeError(
                f"torch_dist shard {release_dir / relative} is truncated: "
                f"{actual_size} bytes, metadata requires {required_size}"
            )


def _artifact_file_inventory(path: Path, manifest: dict[str, object]) -> dict[str, int]:
    artifact = manifest.get("artifact")
    if artifact in {"huggingface-checkpoint", "sglang-block-fp8-checkpoint"}:
        _, shards = _read_safetensor_index(path)
        relative_paths = [Path("config.json"), Path("model.safetensors.index.json")]
        relative_paths.extend(Path(shard) for shard in shards)
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
    _write_artifact_manifest(Path(args.model_dir) / args.model_name, _source_manifest(args))


def _prepare_megatron_ckpt(args: ScriptArgs) -> None:
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

    torch_dist_dst = f"{args.model_local_dir}/{args.model_name}_torch_dist"
    torch_dist_sentinel = Path(torch_dist_dst) / "latest_checkpointed_iteration.txt"
    may_skip_local_copy = skip_existing and args.num_nodes == 1
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
    checkpoint = (
        f"--hf-checkpoint {args.model_local_dir}/{hf_name} "
        f"--ref-load {args.model_local_dir}/{args.model_name}_torch_dist "
    )
    if not args.skip_saving:
        load_save_path = f"{args.output_dir}/{args.run_id}/checkpoints"
        checkpoint += (
            f"--load {load_save_path} --save {load_save_path} "
            "--save-interval 20 --save-retain-interval 20 "
        )
    return checkpoint


def _rollout_args(args: ScriptArgs) -> str:
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
    sglang = (
        f"--rollout-num-gpus-per-engine {world_size} "
        f"--sglang-tp-size {world_size} --sglang-dp-size 1 --sglang-ep-size {world_size} "
        f"--sglang-mem-fraction-static {args.sglang_mem_fraction_static} "
        "--sglang-router-policy consistent_hashing "
    )
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
    if args.offload_train_target == "cpu":
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
    )
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
    )


@app.command()
@U.dataclass_cli
def full_train(args: ScriptArgs) -> None:
    """Download, convert, copy to node-local storage, and train."""
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
    _prepare_download(args)
    _validate_glm_checkpoint(args)
    if args.fp8_rollout:
        _convert_to_fp8(args)
    _prepare_megatron_ckpt(args)


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
