"""Executable GLM-5.2 AMD single-node qualification ladder.

The default replays the complete five-layer TP4/H16 ladder on four MI35x GPUs.
Use ``--profile full-shape-h8`` on an eight-GPU node to exercise the TP8/H8
trainer and SGLang shapes used by the full recipe. The full 744B checkpoint
cannot train on one node; ``--stage full-rollout`` instead runs separately
asserted eager and graph-mode NSA prefill/decode through one TP8/EP8 engine.
The rollout-only path retains lightweight Megatron control ranks and
torch.distributed/tokenizer setup, but has no model parameters, optimizer, or
backward; standalone SGLang process isolation is not implemented.

On an eight-GPU host, expose only devices 0-3 through both
``HIP_VISIBLE_DEVICES`` and ``CUDA_VISIBLE_DEVICES`` for the H16 run. Restore
all eight devices before the H8 or full-checkpoint stages. The driver verifies
the exact visible count and homogeneous MI350X/MI355X SKU before execution.

Examples:
  python tests/e2e/megatron/model_scripts/test_glm5_2_amd_single_node_stages.py
  python tests/e2e/megatron/model_scripts/test_glm5_2_amd_single_node_stages.py \
      --profile full-shape-h8 --stage five-layer-all
  python tests/e2e/megatron/model_scripts/test_glm5_2_amd_single_node_stages.py \
      --stage full-rollout
"""

import argparse
import math
import os
import shlex
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from scripts.amd.run_glm5_2_744b_a40b import (
    ScriptArgs,
    _execute_train,
    _prepare_cp,
    _prepare_download,
    _prepare_megatron_ckpt,
    _validate_glm_checkpoint,
)
from tests.ci.ci_register import register_rocm_ci

import miles.utils.external_utils.command_utils as U

if TYPE_CHECKING:
    from miles.utils.types import Sample

register_rocm_ci(
    est_time=3600,
    suite="nightly-stage-c-4-gpu-mi350",
    labels=["megatron", "model-scripts", "amd"],
    disabled="Enable after the staged GLM-5.2 H16 ladder passes on a physical MI350X runner.",
)

_Profile = Literal["poc-h16", "full-shape-h8"]
_Hardware = Literal["MI350X", "MI355X"]
_FullRolloutMode = Literal["eager", "graph"]
_Stage = Literal[
    "kernel",
    "five-layer-prepare",
    "five-layer-rollout",
    "five-layer-trainer",
    "five-layer-grpo",
    "five-layer-all",
    "full-rollout-eager",
    "full-rollout-graph",
    "full-rollout",
]
_PROXY_ENV_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
_BF16_LOGPROB_ABS_TOLERANCE = 0.03


def _local_gpu_inventory() -> tuple[str, ...]:
    import torch

    if torch.version.hip is None or not torch.cuda.is_available():
        raise RuntimeError("The GLM-5.2 AMD qualification driver requires visible ROCm GPUs")
    return tuple(torch.cuda.get_device_name(rank) for rank in range(torch.cuda.device_count()))


def _hardware(expected_gpus: int) -> _Hardware:
    requested = os.getenv("MILES_GLM52_AMD_HARDWARE", "auto").upper()
    if requested not in {"AUTO", "MI350X", "MI355X"}:
        raise ValueError("MILES_GLM52_AMD_HARDWARE must be auto, MI350X, or MI355X")

    product_names = _local_gpu_inventory()
    if len(product_names) != expected_gpus:
        raise RuntimeError(f"Expected {expected_gpus} visible GPUs, found {len(product_names)}: {product_names}")
    product_skus = tuple(
        next((sku for sku in ("MI350X", "MI355X") if sku in product_name.upper()), None)
        for product_name in product_names
    )
    if len(set(product_skus)) != 1 or product_skus[0] is None:
        raise RuntimeError(f"Expected homogeneous MI350X or MI355X GPUs, found: {product_names}")
    detected = cast(_Hardware, product_skus[0])
    if requested != "AUTO" and requested != detected:
        raise RuntimeError(f"Requested {requested}, but the visible GPUs are homogeneous {detected}: {product_names}")
    print(f"Validated {expected_gpus} visible {detected} GPUs: {product_names}")
    return detected


def _root() -> Path:
    return Path(os.getenv("MILES_GLM52_TEST_ROOT", "/root/shared_data/glm52-amd-single-node"))


def _replay_path(profile: _Profile) -> str:
    return str(_root() / "rollout_data" / profile / "{rollout_id}.pt")


def _full_capture_path(mode: _FullRolloutMode) -> str:
    return str(_root() / "rollout_data" / f"full-{mode}" / "{rollout_id}.pt")


def _run_id(profile: str, stage: str) -> str:
    prefix = os.getenv("MILES_GLM52_RUN_ID_PREFIX", "glm52-amd-single-node")
    return f"{prefix}-{profile}-{stage}-{U.create_run_id()}"


def _five_layer_args(profile: _Profile, stage: str) -> ScriptArgs:
    h8 = profile == "full-shape-h8"
    num_gpus = 8 if h8 else 4
    rollout_only = stage == "rollout"
    extra_args = "--rollout-seed 1234 --seed 1234 "
    if rollout_only:
        extra_args += (
            "--ci-test --sglang-enable-deterministic-inference --sglang-disable-cuda-graph "
            f"--save-debug-rollout-data {shlex.quote(_replay_path(profile))} "
        )
    elif stage == "trainer":
        extra_args += (
            "--ci-test --no-offload-train --save-interval 1 "
            f"--load-debug-rollout-data {shlex.quote(_replay_path(profile))} "
        )
    elif stage == "grpo":
        details = shlex.quote(str(_root() / "details" / profile))
        extra_args += f"--ci-test --enable-event-analyzer --save-interval 1 --dump-details {details} "

    return ScriptArgs(
        hardware=_hardware(num_gpus),
        model_name="GLM-5.2_5layer",
        num_nodes=1,
        num_gpus_per_node=num_gpus,
        single_node_topology=profile,
        allow_unvalidated_features=h8,
        rollout_only=rollout_only,
        num_rollout=1 if stage != "grpo" else 2,
        enable_optimizer_offload=not rollout_only,
        run_id=_run_id(profile, stage),
        output_dir=str(_root()),
        model_dir=os.getenv("MILES_GLM52_MODEL_DIR", "/root/models"),
        model_local_dir=os.getenv("MILES_GLM52_MODEL_LOCAL_DIR", "/root/local_data"),
        data_dir=os.getenv("MILES_GLM52_DATA_DIR", "/root/datasets"),
        extra_args=extra_args,
    )


def _configure_single_node_env() -> None:
    external_ray_vars = [
        name
        for name in ("MILES_SCRIPT_EXTERNAL_RAY", "RAY_ADDRESS")
        if os.environ.get(name, "").strip()
    ]
    if external_ray_vars:
        raise RuntimeError(
            "The single-node GLM-5.2 driver refuses external Ray configuration; "
            f"unset {' '.join(external_ray_vars)} in a clean shell"
        )
    master_addr = os.environ.get("MASTER_ADDR", "").strip()
    if master_addr not in {"", "127.0.0.1", "localhost", "::1"}:
        raise RuntimeError(
            "The single-node GLM-5.2 driver refuses a non-loopback MASTER_ADDR; "
            "unset MASTER_ADDR in a clean shell"
        )
    for name in _PROXY_ENV_VARS:
        os.environ.pop(name, None)


def _prepare_five_layer(profile: _Profile) -> None:
    _configure_single_node_env()
    args = _five_layer_args(profile, "prepare")
    U.exec_command_cpu(f"mkdir -p {shlex.quote(str(_root()))}")
    _prepare_download(args)
    _validate_glm_checkpoint(args)
    _prepare_megatron_ckpt(args)
    _prepare_cp(args, skip_existing=True)


def _assert_rollout_capture(
    path_template: str,
    *,
    expected_samples: int,
    previous_mtime_ns: int | None = None,
) -> list["Sample"]:
    import torch

    from miles.utils.types import Sample

    path = Path(path_template.format(rollout_id=0))
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Rollout capture is missing or empty: {path}")
    if previous_mtime_ns is not None and path.stat().st_mtime_ns <= previous_mtime_ns:
        raise RuntimeError(f"Rollout capture was not refreshed by this run: {path}")

    payload = torch.load(path, weights_only=False)
    samples = [Sample.from_dict(sample) for sample in payload.get("samples", [])]
    if payload.get("rollout_id") != 0 or len(samples) != expected_samples:
        raise RuntimeError(
            f"Expected rollout 0 with {expected_samples} samples in {path}, got {len(samples)}"
        )
    for sample in samples:
        sample.validate()
        if sample.status not in {Sample.Status.COMPLETED, Sample.Status.TRUNCATED}:
            raise RuntimeError(f"Rollout sample {sample.index} ended with {sample.status.value}")
        if sample.response_length <= 0 or sample.rollout_log_probs is None:
            raise RuntimeError(f"Rollout sample {sample.index} has no scored response tokens")
        if not all(math.isfinite(value) for value in sample.rollout_log_probs):
            raise RuntimeError(f"Rollout sample {sample.index} contains non-finite log probabilities")
        if not isinstance(sample.reward, (int, float)) or not math.isfinite(float(sample.reward)):
            raise RuntimeError(f"Rollout sample {sample.index} has no finite DAPO reward")
    return samples


def _assert_full_rollout_parity(eager: "Sample", graph: "Sample") -> None:
    eager_prompt = tuple(eager.tokens[: -eager.response_length])
    graph_prompt = tuple(graph.tokens[: -graph.response_length])
    if eager_prompt != graph_prompt:
        raise RuntimeError("Full eager and graph probes used different prompt token IDs")

    eager_response = tuple(eager.tokens[-eager.response_length :])
    graph_response = tuple(graph.tokens[-graph.response_length :])
    if eager_response != graph_response:
        raise RuntimeError("Full eager and graph probes produced different response token IDs")

    assert eager.rollout_log_probs is not None and graph.rollout_log_probs is not None
    log_probs = (*eager.rollout_log_probs, *graph.rollout_log_probs)
    if not all(math.isfinite(value) for value in log_probs):
        raise RuntimeError("Full eager/graph parity received non-finite rollout log probabilities")
    max_abs_diff = max(
        abs(eager_value - graph_value)
        for eager_value, graph_value in zip(
            eager.rollout_log_probs,
            graph.rollout_log_probs,
            strict=True,
        )
    )
    if max_abs_diff > _BF16_LOGPROB_ABS_TOLERANCE:
        raise RuntimeError(
            "Full eager/graph rollout log probabilities differ by "
            f"{max_abs_diff:.6g}; BF16 tolerance is {_BF16_LOGPROB_ABS_TOLERANCE}"
        )


def _assert_checkpoint(args: ScriptArgs) -> None:
    tracker = Path(args.output_dir) / args.run_id / "checkpoints/latest_checkpointed_iteration.txt"
    if not tracker.is_file() or not tracker.read_text().strip():
        raise RuntimeError(f"Training stage completed without a checkpoint tracker: {tracker}")


def _run_kernel_stage(profile: _Profile) -> None:
    _configure_single_node_env()
    _hardware(8 if profile == "full-shape-h8" else 4)
    test = U.repo_base_dir / "tests/manual/models/glm5/test_tilelang_sparse_mla.py"
    U.exec_command_gpu(f"{shlex.quote(sys.executable)} -m pytest -v {shlex.quote(str(test))}")


def _run_five_layer_stage(profile: _Profile, stage: Literal["rollout", "trainer", "grpo"]) -> None:
    _configure_single_node_env()
    args = _five_layer_args(profile, stage)
    replay_path = _replay_path(profile)
    if stage == "rollout":
        capture = Path(replay_path.format(rollout_id=0))
        previous_mtime_ns = capture.stat().st_mtime_ns if capture.exists() else None
        _execute_train(args)
        _assert_rollout_capture(replay_path, expected_samples=64, previous_mtime_ns=previous_mtime_ns)
        return
    if stage == "trainer":
        _assert_rollout_capture(replay_path, expected_samples=64)
    _execute_train(args)
    _assert_checkpoint(args)


def _full_rollout_args(mode: _FullRolloutMode) -> ScriptArgs:
    return ScriptArgs(
        hardware=_hardware(8),
        model_name="GLM-5.2",
        num_nodes=1,
        num_gpus_per_node=8,
        full_model_rollout_only=True,
        allow_unvalidated_features=True,
        rollout_probe_mode=mode,
        rollout_probe_capture=_full_capture_path(mode),
        run_id=_run_id("full", f"rollout-{mode}"),
        output_dir=str(_root()),
        model_dir=os.getenv("MILES_GLM52_MODEL_DIR", "/root/models"),
        model_local_dir=os.getenv("MILES_GLM52_MODEL_LOCAL_DIR", "/root/local_data"),
        data_dir=os.getenv("MILES_GLM52_DATA_DIR", "/root/datasets"),
    )


def _prepare_full_rollout(args: ScriptArgs) -> None:
    _configure_single_node_env()
    _prepare_download(args)
    _validate_glm_checkpoint(args)
    _prepare_megatron_ckpt(args)
    _prepare_cp(args, skip_existing=True)


def _run_full_rollout(mode: _FullRolloutMode, *, prepare: bool = True) -> list["Sample"]:
    _configure_single_node_env()
    args = _full_rollout_args(mode)
    if prepare:
        _prepare_full_rollout(args)
    capture_path = _full_capture_path(mode)
    capture = Path(capture_path.format(rollout_id=0))
    previous_mtime_ns = capture.stat().st_mtime_ns if capture.exists() else None
    _execute_train(args)
    return _assert_rollout_capture(
        capture_path,
        expected_samples=1,
        previous_mtime_ns=previous_mtime_ns,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("poc-h16", "full-shape-h8"), default="poc-h16")
    parser.add_argument(
        "--stage",
        choices=(
            "kernel",
            "five-layer-prepare",
            "five-layer-rollout",
            "five-layer-trainer",
            "five-layer-grpo",
            "five-layer-all",
            "full-rollout-eager",
            "full-rollout-graph",
            "full-rollout",
        ),
        default="five-layer-all",
    )
    return parser.parse_args()


def main() -> None:
    cli = _parse_args()
    profile: _Profile = cli.profile
    stage: _Stage = cli.stage
    if stage == "kernel":
        _run_kernel_stage(profile)
    elif stage == "full-rollout":
        eager_samples = _run_full_rollout("eager")
        graph_samples = _run_full_rollout("graph", prepare=False)
        _assert_full_rollout_parity(eager_samples[0], graph_samples[0])
    elif stage in {"full-rollout-eager", "full-rollout-graph"}:
        _run_full_rollout(cast(_FullRolloutMode, stage.removeprefix("full-rollout-")))
    elif stage == "five-layer-prepare":
        _prepare_five_layer(profile)
    elif stage == "five-layer-all":
        _run_kernel_stage(profile)
        _prepare_five_layer(profile)
        for substage in ("rollout", "trainer", "grpo"):
            _run_five_layer_stage(profile, substage)
    else:
        substage = stage.removeprefix("five-layer-")
        _run_five_layer_stage(profile, substage)


if __name__ == "__main__":
    main()
