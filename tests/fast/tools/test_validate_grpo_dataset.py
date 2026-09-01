import hashlib
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest


_MODULE_PATH = Path(__file__).resolve().parents[3] / "tools" / "validate_grpo_dataset.py"
_SPEC = importlib.util.spec_from_file_location("validate_grpo_dataset_test_target", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_VALIDATOR = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _VALIDATOR
_SPEC.loader.exec_module(_VALIDATOR)

DAPO_MATH_17K_PROFILE = _VALIDATOR.DAPO_MATH_17K_PROFILE
DatasetProfile = _VALIDATOR.DatasetProfile
validate_dataset = _VALIDATOR.validate_dataset


def _row(content: str = "Solve this. Answer: \\boxed{x}", label: str = "x") -> dict[str, object]:
    return {"prompt": [{"role": "user", "content": content}], "label": label}


def _write_dataset(path, rows):
    payload = b"".join((json.dumps(row, separators=(",", ":")) + "\n").encode() for row in rows)
    path.write_bytes(payload)
    return DatasetProfile(
        name="fixture",
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        expected_size=len(payload),
        expected_rows=len(rows),
        prompt_markers=("Answer:", "\\boxed"),
    )


class _Tokenizer:
    chat_template = "fixture"

    def apply_chat_template(self, conversation, *, tokenize, add_generation_prompt, return_dict, tools):
        assert tokenize is True
        assert add_generation_prompt is True
        assert return_dict is False
        assert tools is None
        return list(range(len(conversation[0]["content"].split()) + 2))


def test_production_dapo_profile_pins_the_audited_payload():
    assert DAPO_MATH_17K_PROFILE.expected_sha256 == ("cc9c39c2aa19177abe9464741e121cf4cac90fd25484ef3cdf86535101e3a5b6")
    assert DAPO_MATH_17K_PROFILE.expected_size == 10_490_834
    assert DAPO_MATH_17K_PROFILE.expected_rows == 17_398


def test_validates_every_row_and_reports_tokenized_lengths(tmp_path):
    path = tmp_path / "data.jsonl"
    rows = [_row(), _row("Please solve carefully. Answer: \\boxed{y}", "y")]
    profile = _write_dataset(path, rows)

    summary = validate_dataset(
        path,
        profile=profile,
        tokenizer=_Tokenizer(),
        max_prompt_tokens=10,
        expected_min_prompt_tokens=6,
        expected_max_prompt_tokens=7,
    )

    assert summary.rows == 2
    assert summary.unique_prompts == 2
    assert summary.min_prompt_tokens == 6
    assert summary.max_prompt_tokens == 7
    assert summary.prompts_over_token_limit == 0


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ({"prompt": [{"role": "user", "content": "Answer: \\boxed{x}"}]}, "expected exactly keys"),
        ({"prompt": [], "label": "x"}, "exactly one chat message"),
        ({"prompt": [{"role": "assistant", "content": "Answer: \\boxed{x}"}], "label": "x"}, "role"),
        ({"prompt": [{"role": "user", "content": ""}], "label": "x"}, "non-empty string"),
        ({"prompt": [{"role": "user", "content": "Answer: x"}], "label": "x"}, "required marker"),
        ({"prompt": [{"role": "user", "content": "Answer: \\boxed{x}"}], "label": ""}, "label"),
    ],
)
def test_rejects_rows_that_do_not_satisfy_the_grpo_contract(tmp_path, row, message):
    path = tmp_path / "data.jsonl"
    profile = _write_dataset(path, [row])

    with pytest.raises(ValueError, match=message):
        validate_dataset(path, profile=profile)


def test_rejects_payload_identity_drift(tmp_path):
    path = tmp_path / "data.jsonl"
    profile = _write_dataset(path, [_row()])

    with pytest.raises(ValueError, match="sha256=.*expected"):
        validate_dataset(path, profile=replace(profile, expected_sha256="0" * 64))


def test_rejects_duplicate_prompts(tmp_path):
    path = tmp_path / "data.jsonl"
    profile = _write_dataset(path, [_row(), _row()])

    with pytest.raises(ValueError, match="unique_prompts=1, expected 2"):
        validate_dataset(path, profile=profile)


def test_rejects_invalid_json_instead_of_silently_skipping_it(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text("not json\n")
    profile = DatasetProfile("fixture", "0" * 64, 9, 1)

    with pytest.raises(ValueError, match="line 1: invalid UTF-8 JSON"):
        validate_dataset(path, profile=profile)


def test_rejects_a_prompt_over_the_token_limit(tmp_path):
    path = tmp_path / "data.jsonl"
    profile = _write_dataset(path, [_row("one two three four Answer: \\boxed{x}")])

    with pytest.raises(
        ValueError,
        match="prompts_over_token_limit=1, expected 0; max_prompt_tokens=8, limit 7",
    ):
        validate_dataset(path, profile=profile, tokenizer=_Tokenizer(), max_prompt_tokens=7)


def test_accepts_an_explicit_expected_filtered_prompt_count(tmp_path):
    path = tmp_path / "data.jsonl"
    profile = _write_dataset(path, [_row(), _row("one two three four Answer: \\boxed{x}")])

    summary = validate_dataset(
        path,
        profile=profile,
        tokenizer=_Tokenizer(),
        max_prompt_tokens=7,
        expected_prompts_over_token_limit=1,
    )

    assert summary.prompts_over_token_limit == 1


def test_token_limit_requires_a_tokenizer(tmp_path):
    path = tmp_path / "data.jsonl"
    profile = _write_dataset(path, [_row()])

    with pytest.raises(ValueError, match="requires a tokenizer"):
        validate_dataset(path, profile=profile, max_prompt_tokens=4096)


def test_rejects_tokenizer_semantic_drift_even_when_prompts_fit_the_limit(tmp_path):
    path = tmp_path / "data.jsonl"
    profile = _write_dataset(path, [_row()])

    with pytest.raises(ValueError, match="max_prompt_tokens=6, expected 7"):
        validate_dataset(
            path,
            profile=profile,
            tokenizer=_Tokenizer(),
            max_prompt_tokens=10,
            expected_max_prompt_tokens=7,
        )
