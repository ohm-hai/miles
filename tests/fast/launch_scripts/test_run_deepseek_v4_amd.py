import pytest

from tests.fast.launch_scripts.py_harness import (
    REPO_ROOT,
    call_entrypoint,
    freeze_environment,
    import_launch_script,
    install_command_recorder,
)


@pytest.mark.parametrize(
    ("model_name", "keeps_pp1"),
    [
        ("DeepSeek-V4-Flash-FP8", True),
        ("DeepSeek-V4-Flash-FP8-4layer", False),
    ],
)
def test_only_the_full_ep8_conversion_keeps_pp1(monkeypatch, tmp_path, model_name, keeps_pp1):
    freeze_environment(monkeypatch)
    recording = install_command_recorder(monkeypatch)
    module = import_launch_script(REPO_ROOT / "scripts/amd/run_deepseek_v4.py")

    call_entrypoint(
        module,
        "prepare_spmd",
        {
            "model_name": model_name,
            "num_nodes": 1,
            "num_gpus_per_node": 8,
            "model_dir": str(tmp_path),
        },
        sandbox=tmp_path,
    )

    assert recording.commands[-1].startswith("CONVERT_KEEP_PP1=1 ") is keeps_pp1
