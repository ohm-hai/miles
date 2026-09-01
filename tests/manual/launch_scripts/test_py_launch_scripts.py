import json
import re
from collections.abc import Callable
from pathlib import Path

import pytest

from tests.fast.launch_scripts.py_harness import (
    CLEARED_ENV,
    FROZEN_HARDWARE,
    call_entrypoint,
    format_recording,
    freeze_environment,
    host_filesystem_frozen,
    import_launch_script,
    install_command_recorder,
    iter_py_launch_scripts,
    launcher_hardware_literals,
)
from tests.fast.launch_scripts.sh_harness import REPO_ROOT, assert_matches_snapshot

_SNAPSHOT_DIR = REPO_ROOT / "tests" / "snapshots" / "launch_scripts" / "py"

_SCRIPTS_IMPORTABLE_ONLY_UNDER_THE_NPU_PATCH = {"scripts/run_qwen3_4b_npu.py"}


def _glm_checkpoint(sandbox: Path, model_name: str, num_layers: int) -> dict[str, object]:
    model_dir = sandbox / "models"
    checkpoint = model_dir / model_name
    checkpoint.mkdir(parents=True)
    indexer_types = [
        "full" if layer <= 3 or (layer - 3) % 4 == 0 else "shared"
        for layer in range(1, num_layers + 1)
    ]
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
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
                "indexer_types": indexer_types,
                "intermediate_size": 12288,
                "kv_lora_rank": 512,
                "mlp_layer_types": ["dense"] * 3 + ["sparse"] * (num_layers - 3),
                "model_type": "glm_moe_dsa",
                "moe_intermediate_size": 2048,
                "n_routed_experts": 256,
                "n_shared_experts": 1,
                "num_attention_heads": 64,
                "num_experts_per_tok": 8,
                "num_hidden_layers": num_layers,
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
        )
    )
    if model_name == "GLM-5.2_5layer":
        num_weights = 1618
        num_shards = 14
        total_size = 45683868160
    else:
        num_weights = 1
        num_shards = 1
        total_size = None
    shard_names = [
        f"model-{index:05d}-of-{num_shards:05d}.safetensors"
        for index in range(1, num_shards + 1)
    ]
    weight_map = {
        f"fixture.weight.{index}": shard_names[index % num_shards]
        for index in range(num_weights)
    }
    index = {"weight_map": weight_map}
    if total_size is not None:
        index["metadata"] = {"total_size": total_size}
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps(index)
    )
    for shard_name in shard_names:
        (checkpoint / shard_name).write_bytes(b"fixture")
    return {"model_dir": str(model_dir)}


def _nemotron_checkpoint(sandbox: Path) -> dict[str, object]:
    model_dir = sandbox / "models"
    checkpoint = model_dir / "NVIDIA-Nemotron-3-Nano-4B-BF16"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "model_type": "nemotron_h",
                "auto_map": {"AutoConfig": "configuration_nemotron_h.NemotronHConfig"},
            }
        )
    )
    return {"model_dir": str(model_dir)}


_SCRIPTS_WHOSE_DEFAULTS_ARE_UNSUPPORTED: dict[str, Callable[[Path], dict[str, object]]] = {
    "scripts/amd/run_glm5_2_744b_a40b.py": lambda sandbox: _glm_checkpoint(
        sandbox, "GLM-5.2_5layer", 5
    ),
    "scripts/run_deepseek_v4.py": lambda sandbox: {"model_name": "DeepSeek-V4-Flash-FP8-4layer"},
    "scripts/run_glm5_744b_a40b.py": lambda sandbox: _glm_checkpoint(sandbox, "GLM-5", 78),
    "scripts/run_glm5_2_744b_a40b.py": lambda sandbox: _glm_checkpoint(sandbox, "GLM-5.2", 78),
    "scripts/run_inkling.py": lambda sandbox: {"model_name": "Inkling-4layer"},
    "scripts/run_nemotron_3_nano_4b_fsdp.py": _nemotron_checkpoint,
    "scripts/run_nemotron_3_ultra_550b_a55b.py": lambda sandbox: {
        "model_name": "NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16-4layer"
    },
}

# The machine each recording represents. Launchers default --hardware to whatever node they run on, so the
# suite pins one; FROZEN_HARDWARE covers the rest.
_HARDWARE_A_RECORDING_REPRESENTS = {
    "scripts/amd/run_glm5_2_744b_a40b.py": "MI355X",
    "scripts/amd/run_qwen3_30b_a3b.py": "MI355X",
    "scripts/amd/run_qwen3_4b.py": "MI355X",
    "scripts/run_deepseek_v32.py": "B200",
    "scripts/run_glm45_355b_a32b.py": "GB200",
    "scripts/run_joy_ai_llm_flash.py": "B200",
    "scripts/run_mcore_fsdp.py": "H100",
    "scripts/run_qwen3_30b_a3b.py": "H100",
    "scripts/run_qwen3_4b.py": "H100",
}

_ENTRYPOINTS_DISABLED_BY_THEIR_OWN_DEFAULTS = {
    ("scripts/run_deepseek_v4.py", "prepare_mxfp8"),
    ("scripts/run_deepseek_v4.py", "prepare_fp8"),
}

_SCRIPTS = [
    script for script in iter_py_launch_scripts() if script.rel not in _SCRIPTS_IMPORTABLE_ONLY_UNDER_THE_NPU_PATCH
]
_CASES = [(script.rel, entrypoint) for script in _SCRIPTS for entrypoint in script.entrypoints]


@pytest.fixture(params=_CASES, ids=[f"{rel}::{entrypoint}" for rel, entrypoint in _CASES])
def recorded(request, monkeypatch, tmp_path):
    rel, entrypoint = request.param
    freeze_environment(monkeypatch, hardware=_HARDWARE_A_RECORDING_REPRESENTS.get(rel, FROZEN_HARDWARE))
    recording = install_command_recorder(monkeypatch)
    module = import_launch_script(REPO_ROOT / rel)
    call_entrypoint(
        module,
        entrypoint,
        _SCRIPTS_WHOSE_DEFAULTS_ARE_UNSUPPORTED.get(rel, lambda sandbox: {})(tmp_path),
        sandbox=tmp_path,
    )
    return rel, entrypoint, recording, tmp_path


class TestEveryLauncherEntrypoint:
    def test_commands_match_snapshot(self, recorded):
        """Every launcher entrypoint must build exactly the recorded shell commands."""
        rel, entrypoint, recording, sandbox = recorded
        snapshot = _SNAPSHOT_DIR / rel / f"{entrypoint}.txt"

        assert_matches_snapshot(snapshot, format_recording(recording, sandbox=sandbox), f"{rel}::{entrypoint}")

    def test_entrypoint_issues_commands(self, recorded):
        """An entrypoint that silently does nothing is a broken launcher, not a passing test."""
        rel, entrypoint, recording, _ = recorded
        if (rel, entrypoint) in _ENTRYPOINTS_DISABLED_BY_THEIR_OWN_DEFAULTS:
            assert not recording.commands
        else:
            assert recording.commands


class TestHostFilesystemIsFrozen:
    def test_paths_outside_the_checkout_and_the_sandbox_report_absence(self, tmp_path):
        """A launcher that can see the host's checkpoints skips work, so the snapshot would follow the machine."""
        inside = tmp_path / "checkpoint.json"
        inside.write_text("{}")

        with host_filesystem_frozen(tmp_path):
            assert inside.exists()
            assert not Path("/root/models/some-checkpoint/model.safetensors.index.json").exists()

    def test_the_checkout_stays_visible(self, tmp_path):
        """A launcher resolves its own model args script out of the checkout, so hiding it breaks every launcher."""
        with host_filesystem_frozen(tmp_path):
            assert (REPO_ROOT / "pyproject.toml").exists()
            assert (REPO_ROOT / "scripts" / "models").exists()

    def test_an_unreadable_parent_reports_absence_instead_of_raising(self, tmp_path):
        """python 3.11 raises PermissionError from exists(), which is how the CPU runner's /root broke this."""
        unreadable = tmp_path / "unreadable"
        unreadable.mkdir()
        unreadable.chmod(0o000)
        try:
            with host_filesystem_frozen(tmp_path / "sandbox"):
                assert not (unreadable / "model.safetensors.index.json").exists()
        finally:
            unreadable.chmod(0o700)


class TestDiscovery:
    def test_all_py_launch_scripts_are_discovered(self):
        """Guards against the discovery glob silently going empty."""
        assert len(_SCRIPTS) > 15

    def test_every_discovered_launcher_is_covered_except_the_one_this_checkout_cannot_import(self):
        """A denylist that nobody rechecks only grows; name the survivors so the count cannot drift."""
        discovered = {script.rel for script in iter_py_launch_scripts()}

        assert discovered - {script.rel for script in _SCRIPTS} == _SCRIPTS_IMPORTABLE_ONLY_UNDER_THE_NPU_PATCH

    @pytest.mark.parametrize("rel", sorted(_SCRIPTS_IMPORTABLE_ONLY_UNDER_THE_NPU_PATCH))
    def test_the_uncovered_launcher_really_is_uncoverable_here(self, rel):
        """Once the NPU patch is upstreamed this fails, forcing the exclusion out instead of letting it rot."""
        with pytest.raises(ImportError, match="execute_train_npu"):
            import_launch_script(REPO_ROOT / rel)

    def test_every_pinned_recording_names_a_launcher_that_accepts_that_hardware(self):
        """A pin the launcher does not accept records a profile nobody can run, and a stale key pins nothing."""
        literals = launcher_hardware_literals()

        assert _HARDWARE_A_RECORDING_REPRESENTS.keys() <= literals.keys()
        for rel, hardware in _HARDWARE_A_RECORDING_REPRESENTS.items():
            assert hardware in literals[rel], rel

    def test_a_launcher_the_frozen_default_covers_is_not_pinned_as_well(self):
        """Two ways to say the same thing drift apart; only the launchers FROZEN_HARDWARE cannot serve get a pin."""
        redundant = {rel for rel, hardware in _HARDWARE_A_RECORDING_REPRESENTS.items() if hardware == FROZEN_HARDWARE}

        assert not redundant

    def test_every_environment_knob_a_model_script_reads_is_frozen(self):
        """The snapshots now pin expanded model args, so a developer's exported override would fail them."""
        knobs = set()
        for script in sorted((REPO_ROOT / "scripts" / "models").iterdir()):
            if not script.is_file():
                continue
            text = script.read_text()
            knobs |= set(re.findall(r"\$\{([A-Z][A-Z0-9_]*):-", text))
            knobs |= set(re.findall(r"environ\.get\(\s*\"([A-Z][A-Z0-9_]*)\"", text))

        assert knobs
        assert knobs <= set(CLEARED_ENV)

    def test_execute_train_config_defaults_are_not_taken_from_a_slurm_allocation(self, monkeypatch):
        """SLURM_JOB_NUM_NODES is read at import time, so a stale allocation would skew every snapshot."""
        import miles.utils.external_utils.command_utils as command_utils

        assert command_utils.ExecuteTrainConfig().num_nodes == 1
