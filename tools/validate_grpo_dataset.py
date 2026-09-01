"""Validate an immutable GRPO JSONL dataset before allocating GPUs.

The built-in ``dapo-math-17k`` profile is the exact dataset consumed by the
GLM-5.2 AMD launcher. Validation is streaming and uses only the standard
library unless optional tokenizer validation is requested.

Example:
  python tools/validate_grpo_dataset.py \
      /shared/datasets/dapo-math-17k/dapo-math-17k.jsonl \
      --profile dapo-math-17k

Add ``--tokenizer /shared/models/GLM-5.2 --max-prompt-tokens 4096`` inside the
training image to exercise the local chat template and tokenizer as well.
"""

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class DatasetProfile:
    """Immutable payload and row-level contract for one dataset."""

    name: str
    expected_sha256: str
    expected_size: int
    expected_rows: int
    prompt_key: str = "prompt"
    label_key: str = "label"
    prompt_markers: tuple[str, ...] = ()
    require_unique_prompts: bool = True


DAPO_MATH_17K_PROFILE = DatasetProfile(
    name="dapo-math-17k",
    expected_sha256="cc9c39c2aa19177abe9464741e121cf4cac90fd25484ef3cdf86535101e3a5b6",
    expected_size=10_490_834,
    expected_rows=17_398,
    prompt_markers=("Answer:", "\\boxed"),
)

PROFILES = {DAPO_MATH_17K_PROFILE.name: DAPO_MATH_17K_PROFILE}


class ChatTokenizer(Protocol):
    chat_template: str | None

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_dict: bool,
        tools: None,
    ) -> list[int]: ...


@dataclass(frozen=True)
class ValidationSummary:
    profile: str
    path: str
    sha256: str
    size_bytes: int
    rows: int
    unique_prompts: int
    min_prompt_chars: int
    max_prompt_chars: int
    min_prompt_tokens: int | None
    max_prompt_tokens: int | None
    prompts_over_token_limit: int | None


def _row_error(line_number: int, detail: str) -> ValueError:
    return ValueError(f"line {line_number}: {detail}")


def _validate_row(
    row: object,
    *,
    line_number: int,
    profile: DatasetProfile,
) -> tuple[list[dict[str, str]], str]:
    if not isinstance(row, dict):
        raise _row_error(line_number, "expected a JSON object")

    expected_keys = {profile.prompt_key, profile.label_key}
    if set(row) != expected_keys:
        raise _row_error(
            line_number,
            f"expected exactly keys {sorted(expected_keys)}, got {sorted(row)}",
        )

    prompt = row[profile.prompt_key]
    if not isinstance(prompt, list) or len(prompt) != 1:
        raise _row_error(line_number, "prompt must contain exactly one chat message")
    message = prompt[0]
    if not isinstance(message, dict) or set(message) != {"role", "content"}:
        raise _row_error(line_number, "prompt message must contain exactly role and content")
    if message["role"] != "user":
        raise _row_error(line_number, "prompt message role must be 'user'")
    content = message["content"]
    if not isinstance(content, str) or not content.strip():
        raise _row_error(line_number, "prompt message content must be a non-empty string")
    for marker in profile.prompt_markers:
        if marker not in content:
            raise _row_error(line_number, f"prompt is missing required marker {marker!r}")

    label = row[profile.label_key]
    if not isinstance(label, str) or not label.strip():
        raise _row_error(line_number, "label must be a non-empty string")
    return prompt, content


def _token_count(
    tokenizer: ChatTokenizer,
    prompt: list[dict[str, str]],
    *,
    line_number: int,
) -> int:
    try:
        token_ids = tokenizer.apply_chat_template(
            prompt,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
            tools=None,
        )
    except Exception as error:
        raise _row_error(line_number, f"chat-template tokenization failed: {error}") from error
    if not isinstance(token_ids, list) or any(not isinstance(token_id, int) for token_id in token_ids):
        raise _row_error(line_number, "chat-template tokenization did not return a list of token IDs")
    if not token_ids:
        raise _row_error(line_number, "chat-template tokenization produced an empty prompt")
    return len(token_ids)


def validate_dataset(
    path: Path,
    *,
    profile: DatasetProfile,
    tokenizer: ChatTokenizer | None = None,
    max_prompt_tokens: int | None = None,
    expected_prompts_over_token_limit: int = 0,
    expected_min_prompt_tokens: int | None = None,
    expected_max_prompt_tokens: int | None = None,
) -> ValidationSummary:
    """Validate payload identity, every row, and optionally rendered prompt lengths."""
    if max_prompt_tokens is not None and max_prompt_tokens <= 0:
        raise ValueError("max_prompt_tokens must be positive")
    if max_prompt_tokens is not None and tokenizer is None:
        raise ValueError("max_prompt_tokens requires a tokenizer")
    if (expected_min_prompt_tokens is not None or expected_max_prompt_tokens is not None) and tokenizer is None:
        raise ValueError("expected prompt-token bounds require a tokenizer")
    for name, value in (
        ("expected_min_prompt_tokens", expected_min_prompt_tokens),
        ("expected_max_prompt_tokens", expected_max_prompt_tokens),
    ):
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be positive")
    if expected_prompts_over_token_limit < 0:
        raise ValueError("expected_prompts_over_token_limit cannot be negative")
    if max_prompt_tokens is None and expected_prompts_over_token_limit:
        raise ValueError("expected_prompts_over_token_limit requires max_prompt_tokens")
    if tokenizer is not None and not getattr(tokenizer, "chat_template", None):
        raise ValueError("tokenizer does not define a chat template")
    if not path.is_file():
        raise FileNotFoundError(f"dataset does not exist: {path}")

    digest = hashlib.sha256()
    size_bytes = 0
    prompt_fingerprints: set[str] = set()
    prompt_char_lengths: list[int] = []
    prompt_token_lengths: list[int] = []

    with path.open("rb") as dataset_file:
        for line_number, raw_line in enumerate(dataset_file, start=1):
            digest.update(raw_line)
            size_bytes += len(raw_line)
            if not raw_line.strip():
                raise _row_error(line_number, "blank lines are not allowed")
            try:
                row = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise _row_error(line_number, f"invalid UTF-8 JSON: {error}") from error

            prompt, content = _validate_row(row, line_number=line_number, profile=profile)
            prompt_fingerprints.add(json.dumps(prompt, ensure_ascii=False, sort_keys=True))
            prompt_char_lengths.append(len(content))
            if tokenizer is not None:
                prompt_token_lengths.append(_token_count(tokenizer, prompt, line_number=line_number))

    rows = len(prompt_char_lengths)
    actual_sha256 = digest.hexdigest()
    mismatches = []
    if actual_sha256 != profile.expected_sha256:
        mismatches.append(f"sha256={actual_sha256}, expected {profile.expected_sha256}")
    if size_bytes != profile.expected_size:
        mismatches.append(f"size={size_bytes}, expected {profile.expected_size}")
    if rows != profile.expected_rows:
        mismatches.append(f"rows={rows}, expected {profile.expected_rows}")
    if profile.require_unique_prompts and len(prompt_fingerprints) != rows:
        mismatches.append(f"unique_prompts={len(prompt_fingerprints)}, expected {rows}")
    min_prompt_tokens = min(prompt_token_lengths) if prompt_token_lengths else None
    max_prompt_tokens_seen = max(prompt_token_lengths) if prompt_token_lengths else None
    if expected_min_prompt_tokens is not None and min_prompt_tokens != expected_min_prompt_tokens:
        mismatches.append(f"min_prompt_tokens={min_prompt_tokens}, expected {expected_min_prompt_tokens}")
    if expected_max_prompt_tokens is not None and max_prompt_tokens_seen != expected_max_prompt_tokens:
        mismatches.append(f"max_prompt_tokens={max_prompt_tokens_seen}, expected {expected_max_prompt_tokens}")
    prompts_over_token_limit = None
    if prompt_token_lengths and max_prompt_tokens is not None:
        prompts_over_token_limit = sum(length > max_prompt_tokens for length in prompt_token_lengths)
        if prompts_over_token_limit != expected_prompts_over_token_limit:
            mismatches.append(f"prompts_over_token_limit={prompts_over_token_limit}, expected {expected_prompts_over_token_limit}; max_prompt_tokens={max_prompt_tokens_seen}, limit {max_prompt_tokens}")
    if mismatches:
        raise ValueError(f"{path} does not match profile {profile.name!r}: " + "; ".join(mismatches))
    if rows == 0:
        raise ValueError(f"{path} contains no samples")

    return ValidationSummary(
        profile=profile.name,
        path=str(path.resolve()),
        sha256=actual_sha256,
        size_bytes=size_bytes,
        rows=rows,
        unique_prompts=len(prompt_fingerprints),
        min_prompt_chars=min(prompt_char_lengths),
        max_prompt_chars=max(prompt_char_lengths),
        min_prompt_tokens=min_prompt_tokens,
        max_prompt_tokens=max_prompt_tokens_seen,
        prompts_over_token_limit=prompts_over_token_limit,
    )


def _load_tokenizer(path: Path) -> ChatTokenizer:
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("--tokenizer requires transformers in the current environment") from error
    return AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="JSONL file to validate")
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--tokenizer", type=Path, help="Local HF tokenizer/checkpoint directory")
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        help="Rendered-prompt threshold; expected over-limit rows default to zero",
    )
    parser.add_argument(
        "--expected-prompts-over-token-limit",
        type=int,
        default=0,
        help="Expected rows filtered by the token limit (default: 0)",
    )
    parser.add_argument("--expected-min-prompt-tokens", type=int)
    parser.add_argument("--expected-max-prompt-tokens", type=int)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    tokenizer = _load_tokenizer(args.tokenizer) if args.tokenizer is not None else None
    summary = validate_dataset(
        args.path,
        profile=PROFILES[args.profile],
        tokenizer=tokenizer,
        max_prompt_tokens=args.max_prompt_tokens,
        expected_prompts_over_token_limit=args.expected_prompts_over_token_limit,
        expected_min_prompt_tokens=args.expected_min_prompt_tokens,
        expected_max_prompt_tokens=args.expected_max_prompt_tokens,
    )
    print(json.dumps(asdict(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
