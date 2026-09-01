"""Immutable topology contracts for the AMD GLM-5.2 launcher."""

import shlex
from dataclasses import dataclass
from typing import Literal


SingleNodeTopology = Literal["poc-h16", "full-shape-h8"]

# ``extra_args`` is intentionally a narrow diagnostics channel. The integer is
# the number of following values accepted by each exact option spelling.
_DIAGNOSTIC_EXTRA_ARG_ARITY = {
    "--ci-disable-logprobs-checker": 0,
    "--ci-disable-weight-update-checker": 0,
    "--ci-test": 0,
    "--check-weight-update-allow-quant-error": 0,
    "--debug-deterministic-collective": 0,
    "--debug-exit-after-rollout": 1,
    "--deterministic-mode": 0,
    "--dump-details": 1,
    "--enable-event-analyzer": 0,
    "--load-debug-rollout-data": 1,
    "--no-offload-train": 0,
    "--rollout-seed": 1,
    "--save-debug-rollout-data": 1,
    "--save-interval": 1,
    "--save-local-weight-checksum": 0,
    "--seed": 1,
    "--sglang-disable-cuda-graph": 0,
    "--sglang-enable-deterministic-inference": 0,
    "--use-miles-dashboard": 0,
}


def _validate_diagnostic_extra_args(extra_args: str) -> set[str]:
    tokens = shlex.split(extra_args)
    flags: set[str] = set()
    position = 0
    while position < len(tokens):
        token = tokens[position]
        flag, has_equals, inline_value = token.partition("=")
        if flag in {"--debug-rollout-only", "--no-debug-rollout-only"}:
            raise ValueError("Use the launcher's --rollout-only profile instead of extra_args")
        if flag not in _DIAGNOSTIC_EXTRA_ARG_ARITY:
            raise ValueError(
                f"extra_args accepts only audited diagnostic options; unsupported token {token!r}"
            )
        if flag in flags:
            raise ValueError(f"extra_args repeats diagnostic option {flag}")
        flags.add(flag)
        arity = _DIAGNOSTIC_EXTRA_ARG_ARITY[flag]
        if arity == 0:
            if has_equals:
                raise ValueError(f"Diagnostic option {flag} does not accept a value")
            position += 1
            continue
        if has_equals:
            if not inline_value:
                raise ValueError(f"Diagnostic option {flag} requires one value")
            position += 1
            continue
        if position + 1 >= len(tokens) or tokens[position + 1].startswith("--"):
            raise ValueError(f"Diagnostic option {flag} requires one value")
        position += 2
    return flags


@dataclass(frozen=True)
class Topology:
    name: str
    num_nodes: int
    num_gpus_per_node: int
    tensor_parallel: int
    pipeline_parallel: int
    expert_parallel: int
    pipeline_layers: tuple[int, ...]
    experimental_feature: str | None = None
    rollout_only: bool = False

    @property
    def stage_starts(self) -> tuple[int, ...]:
        starts = [1]
        for num_layers in self.pipeline_layers[:-1]:
            starts.append(starts[-1] + num_layers)
        return tuple(starts)

    def megatron_args(self) -> str:
        args = (
            f"--tensor-model-parallel-size {self.tensor_parallel} "
            "--sequence-parallel "
            f"--pipeline-model-parallel-size {self.pipeline_parallel} "
        )
        if self.pipeline_parallel == 4:
            args += (
                f"--decoder-first-pipeline-num-layers {self.pipeline_layers[0]} "
                f"--decoder-last-pipeline-num-layers {self.pipeline_layers[-1]} "
            )
        return args + (
            "--context-parallel-size 1 "
            f"--expert-model-parallel-size {self.expert_parallel} "
            "--expert-tensor-parallel-size 1 "
        )


def validate_rollout_contract(
    topology: Topology,
    *,
    rollout_only: bool,
    num_rollout: int,
    max_response_len: int,
    extra_args: str,
) -> None:
    if topology.rollout_only and extra_args.strip():
        raise ValueError("The full-model rollout-only profile rejects extra_args; use its typed fields")
    extra_flags = _validate_diagnostic_extra_args(extra_args)
    typed_full_rollout_flags = {
        "--save-debug-rollout-data": "rollout_probe_capture",
        "--sglang-disable-cuda-graph": "rollout_probe_mode",
    }
    for flag, field in typed_full_rollout_flags.items():
        if rollout_only and topology.name.startswith("full-model") and flag in extra_flags:
            raise ValueError(f"Use {field} instead of {flag} for a full-model rollout probe")
    if rollout_only and "--load-debug-rollout-data" in extra_flags:
        raise ValueError("A rollout-only profile cannot load trainer-only debug rollout data")
    if rollout_only and num_rollout != 1:
        raise ValueError("A rollout-only probe requires num_rollout=1")
    if not topology.rollout_only:
        return
    if max_response_len > 128:
        raise ValueError("The full-model single-node rollout probe requires response length <=128")


def full_model_rollout_args(
    topology: Topology,
    *,
    rollout_only: bool,
    data_dir: str,
    max_prompt_len: int,
    max_response_len: int,
) -> str | None:
    if not rollout_only or not topology.name.startswith("full-model"):
        return None
    # One request is sufficient for each engine. The 8x8 profile therefore
    # issues eight deterministic requests so all eight node-local engines run.
    rollout_batch_size = topology.num_nodes
    return (
        f"--prompt-data {data_dir}/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt --label-key label --apply-chat-template --rm-type deepscaler "
        f"--num-rollout 1 --rollout-batch-size {rollout_batch_size} --n-samples-per-prompt 1 "
        f"--rollout-max-prompt-len {max_prompt_len} --rollout-max-response-len {max_response_len} "
        f"--rollout-temperature 0 --rollout-seed 1234 --global-batch-size {rollout_batch_size} "
    )


def resolve(
    *,
    model_name: str,
    single_node_topology: SingleNodeTopology,
    full_model_rollout_only: bool,
) -> Topology:
    if model_name == "GLM-5.2_5layer":
        if full_model_rollout_only:
            raise ValueError("full_model_rollout_only is valid only for the full GLM-5.2 checkpoint")
        if single_node_topology == "full-shape-h8":
            return Topology(
                name="five-layer-full-shape-h8",
                num_nodes=1,
                num_gpus_per_node=8,
                tensor_parallel=8,
                pipeline_parallel=1,
                expert_parallel=8,
                pipeline_layers=(5,),
                experimental_feature="five_layer_full_shape_h8",
            )
        return Topology(
            name="five-layer-poc-h16",
            num_nodes=1,
            num_gpus_per_node=4,
            tensor_parallel=4,
            pipeline_parallel=1,
            expert_parallel=4,
            pipeline_layers=(5,),
        )

    if single_node_topology != "poc-h16":
        raise ValueError("single_node_topology applies only to GLM-5.2_5layer")
    if full_model_rollout_only:
        return Topology(
            name="full-model-rollout-only-h8",
            num_nodes=1,
            num_gpus_per_node=8,
            tensor_parallel=8,
            pipeline_parallel=1,
            expert_parallel=8,
            pipeline_layers=(78,),
            experimental_feature="full_model_rollout_only",
            rollout_only=True,
        )
    return Topology(
        name="full-model-8x8",
        num_nodes=8,
        num_gpus_per_node=8,
        tensor_parallel=8,
        pipeline_parallel=4,
        expert_parallel=16,
        pipeline_layers=(18, 20, 20, 20),
    )
