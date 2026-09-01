"""Dataset preflight hooks for the AMD GLM-5.2 GRPO launcher."""

import shlex
from typing import Protocol

import miles.utils.external_utils.command_utils as U


class _DatasetArgs(Protocol):
    data_dir: str
    fp8_rollout: bool
    model_local_dir: str
    model_dir: str
    model_name: str
    num_nodes: int
    rollout_max_prompt_len: int | None


def _validation_command(args: _DatasetArgs, *, tokenizer_root: str, tokenizer_name: str) -> str:
    dataset = f"{args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl"
    validator = U.repo_base_dir / "tools/validate_grpo_dataset.py"
    command = f"python {shlex.quote(str(validator))} {shlex.quote(dataset)} --profile dapo-math-17k"
    command += f" --tokenizer {shlex.quote(f'{tokenizer_root}/{tokenizer_name}')}"
    assert args.rollout_max_prompt_len is not None
    command += f" --max-prompt-tokens {args.rollout_max_prompt_len} --expected-min-prompt-tokens 73 --expected-max-prompt-tokens 1521"
    # The five-layer PoC intentionally lets RolloutDataSource filter seven
    # audited prompts. Pin that count so tokenizer drift cannot silently
    # change which samples reach the smoke run.
    if args.model_name == "GLM-5.2_5layer" and args.rollout_max_prompt_len == 1024:
        command += " --expected-prompts-over-token-limit 7"
    return command


def _validate_after_download(args: _DatasetArgs) -> None:
    U.exec_command_cpu(
        _validation_command(args, tokenizer_root=args.model_dir, tokenizer_name=args.model_name)
    )


def _validate_before_train(args: _DatasetArgs) -> None:
    tokenizer_name = f"{args.model_name}_fp8" if args.fp8_rollout else args.model_name
    command = _validation_command(
        args,
        tokenizer_root=args.model_local_dir,
        tokenizer_name=tokenizer_name,
    )
    if args.num_nodes == 1:
        U.exec_command_cpu(command)
    else:
        U.exec_command_multi_node(command, num_nodes=args.num_nodes)
