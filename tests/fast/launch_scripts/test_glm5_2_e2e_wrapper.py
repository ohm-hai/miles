import importlib.util
import shlex
import sys
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace

import miles
import pytest


def _stub_module(monkeypatch, name: str, **attributes) -> ModuleType:
    module = ModuleType(name)
    for attribute, value in attributes.items():
        setattr(module, attribute, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _load_wrapper(monkeypatch):
    class _ScriptArgs:
        pass

    launcher_attributes = {
        "ScriptArgs": _ScriptArgs,
        "_convert_to_fp8": lambda _args: None,
        "_execute_train": lambda _args: None,
        "_prepare_download": lambda _args: None,
        "_prepare_megatron_ckpt": lambda _args: None,
        "_validate_glm_checkpoint": lambda _args: None,
    }
    _stub_module(monkeypatch, "scripts.run_glm5_2_744b_a40b", **launcher_attributes)
    _stub_module(
        monkeypatch,
        "tests.ci.ci_register",
        register_cuda_ci=lambda **_kwargs: None,
        register_rocm_ci=lambda **_kwargs: None,
    )
    _stub_module(monkeypatch, "tests.ci.metric_history", register_ci_gate=lambda **_kwargs: None)
    utils_module = _stub_module(monkeypatch, "miles.utils")
    external_utils_module = _stub_module(monkeypatch, "miles.utils.external_utils")
    command_utils_module = _stub_module(
        monkeypatch,
        "miles.utils.external_utils.command_utils",
        exec_command_cpu=lambda _command: None,
    )
    monkeypatch.setattr(miles, "utils", utils_module, raising=False)
    utils_module.external_utils = external_utils_module
    external_utils_module.command_utils = command_utils_module
    monkeypatch.delenv("MILES_HARDWARE_PLATFORM", raising=False)

    module_path = (
        Path(__file__).resolve().parents[3]
        / "tests/e2e/megatron/model_scripts/test_glm5_2_744b_a40b_5layer_ci.py"
    )
    module_name = "tests.e2e.megatron.model_scripts._glm52_e2e_wrapper_test_target"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def _load_single_node_driver(monkeypatch):
    class _ScriptArgs:
        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)

    launcher_attributes = {
        "ScriptArgs": _ScriptArgs,
        "_execute_train": lambda _args: None,
        "_prepare_cp": lambda _args, *, skip_existing=False: None,
        "_prepare_download": lambda _args: None,
        "_prepare_megatron_ckpt": lambda _args: None,
        "_validate_glm_checkpoint": lambda _args: None,
    }
    _stub_module(monkeypatch, "scripts.amd.run_glm5_2_744b_a40b", **launcher_attributes)
    _stub_module(monkeypatch, "tests.ci.ci_register", register_rocm_ci=lambda **_kwargs: None)
    utils_module = _stub_module(monkeypatch, "miles.utils")
    external_utils_module = _stub_module(monkeypatch, "miles.utils.external_utils")
    command_utils_module = _stub_module(
        monkeypatch,
        "miles.utils.external_utils.command_utils",
        create_run_id=lambda: "unit-test-run",
        exec_command_cpu=lambda _command: None,
        exec_command_gpu=lambda _command: None,
        repo_base_dir=Path(__file__).resolve().parents[3],
    )
    monkeypatch.setattr(miles, "utils", utils_module, raising=False)
    utils_module.external_utils = external_utils_module
    external_utils_module.command_utils = command_utils_module

    module_path = (
        Path(__file__).resolve().parents[3]
        / "tests/e2e/megatron/model_scripts/test_glm5_2_amd_single_node_stages.py"
    )
    module_name = "tests.e2e.megatron.model_scripts._glm52_single_node_driver_test_target"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def _stub_preparation(monkeypatch, wrapper) -> None:
    monkeypatch.setattr(wrapper, "_prepare_download", lambda _args: None)
    monkeypatch.setattr(wrapper, "_validate_glm_checkpoint", lambda _args: None)
    monkeypatch.setattr(wrapper, "_prepare_megatron_ckpt", lambda _args: None)


def test_nvidia_prepare_does_not_require_ray_for_a_noop_local_copy(monkeypatch) -> None:
    wrapper = _load_wrapper(monkeypatch)
    calls = []
    monkeypatch.setattr(wrapper.U, "exec_command_cpu", lambda command: calls.append(command))
    _stub_preparation(monkeypatch, wrapper)

    wrapper.prepare(SimpleNamespace(output_dir="/tmp/glm52-output", fp8_rollout=False))

    assert calls == ["mkdir -p /tmp/glm52-output"]


def test_amd_prepare_stages_the_validated_node_local_bundle(monkeypatch) -> None:
    wrapper = _load_wrapper(monkeypatch)
    copied = []
    _stub_preparation(monkeypatch, wrapper)
    monkeypatch.setattr(
        wrapper,
        "_amd_prepare_cp",
        lambda args, *, skip_existing: copied.append((args, skip_existing)),
    )
    args = SimpleNamespace(output_dir="/tmp/glm52-output", fp8_rollout=False)

    wrapper.prepare(args)

    assert copied == [(args, True)]


def test_single_node_driver_locks_the_h16_and_h8_five_layer_shapes(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MILES_GLM52_TEST_ROOT", str(tmp_path))
    monkeypatch.setenv("MILES_GLM52_AMD_HARDWARE", "MI355X")
    driver = _load_single_node_driver(monkeypatch)
    monkeypatch.setattr(driver, "_local_gpu_inventory", lambda: ("AMD Instinct MI355X",) * 4)

    h16 = driver._five_layer_args("poc-h16", "rollout")
    monkeypatch.setattr(driver, "_local_gpu_inventory", lambda: ("AMD Instinct MI355X",) * 8)
    h8 = driver._five_layer_args("full-shape-h8", "grpo")

    assert (h16.num_nodes, h16.num_gpus_per_node, h16.single_node_topology) == (1, 4, "poc-h16")
    assert h16.rollout_only is True
    assert h16.num_rollout == 1
    assert "--sglang-enable-deterministic-inference" in h16.extra_args
    h16_words = shlex.split(h16.extra_args)
    capture_index = h16_words.index("--save-debug-rollout-data")
    assert h16_words[capture_index + 1] == f"{tmp_path}/rollout_data/poc-h16/{{rollout_id}}.pt"
    monkeypatch.setattr(driver, "_local_gpu_inventory", lambda: ("AMD Instinct MI355X",) * 4)
    trainer = driver._five_layer_args("poc-h16", "trainer")
    assert "--load-debug-rollout-data" in trainer.extra_args
    assert "--debug-train-only" not in trainer.extra_args
    assert (h8.num_nodes, h8.num_gpus_per_node, h8.single_node_topology) == (1, 8, "full-shape-h8")
    assert h8.allow_unvalidated_features is True
    assert h8.rollout_only is False
    assert h8.num_rollout == 2
    assert "--enable-event-analyzer" in h8.extra_args


def test_single_node_full_rollout_runs_the_unwrapped_preparation_sequence(monkeypatch) -> None:
    monkeypatch.setenv("MILES_GLM52_AMD_HARDWARE", "MI355X")
    driver = _load_single_node_driver(monkeypatch)
    monkeypatch.setattr(driver, "_local_gpu_inventory", lambda: ("AMD Instinct MI355X",) * 8)
    calls = []
    monkeypatch.setattr(driver, "_prepare_download", lambda args: calls.append(("download", args)))
    monkeypatch.setattr(driver, "_validate_glm_checkpoint", lambda args: calls.append(("validate", args)))
    monkeypatch.setattr(driver, "_prepare_megatron_ckpt", lambda args: calls.append(("convert", args)))
    monkeypatch.setattr(
        driver,
        "_prepare_cp",
        lambda args, *, skip_existing: calls.append(("copy", args, skip_existing)),
    )
    monkeypatch.setattr(driver, "_execute_train", lambda args: calls.append(("execute", args)))
    monkeypatch.setattr(
        driver,
        "_assert_rollout_capture",
        lambda path, **kwargs: calls.append(("capture", path, kwargs)),
    )

    driver._run_full_rollout("eager")

    assert [call[0] for call in calls] == ["download", "validate", "convert", "copy", "execute", "capture"]
    args = calls[0][1]
    assert (args.model_name, args.num_nodes, args.num_gpus_per_node) == ("GLM-5.2", 1, 8)
    assert args.full_model_rollout_only is True
    assert args.allow_unvalidated_features is True
    assert args.rollout_probe_mode == "eager"
    assert args.rollout_probe_capture.endswith("/rollout_data/full-eager/{rollout_id}.pt")
    assert calls[3][2] is True
    assert calls[5][2]["expected_samples"] == 1


def test_single_node_driver_detects_hardware_and_rejects_a_false_sku_claim(monkeypatch) -> None:
    driver = _load_single_node_driver(monkeypatch)
    monkeypatch.setattr(driver, "_local_gpu_inventory", lambda: ("AMD Instinct MI350X",) * 4)
    monkeypatch.delenv("MILES_GLM52_AMD_HARDWARE", raising=False)

    assert driver._hardware(4) == "MI350X"

    monkeypatch.setenv("MILES_GLM52_AMD_HARDWARE", "MI355X")
    with pytest.raises(RuntimeError, match="Requested MI355X.*homogeneous MI350X"):
        driver._hardware(4)


def test_single_node_driver_requires_the_profile_gpu_count(monkeypatch) -> None:
    driver = _load_single_node_driver(monkeypatch)
    monkeypatch.setattr(driver, "_local_gpu_inventory", lambda: ("AMD Instinct MI355X",) * 4)

    with pytest.raises(RuntimeError, match="Expected 8 visible GPUs, found 4"):
        driver._hardware(8)


def test_single_node_driver_fails_closed_on_an_external_ray_target(monkeypatch) -> None:
    driver = _load_single_node_driver(monkeypatch)
    for name in (*driver._PROXY_ENV_VARS, "MILES_SCRIPT_EXTERNAL_RAY", "RAY_ADDRESS"):
        monkeypatch.setenv(name, "unsafe-external-value")

    with pytest.raises(RuntimeError, match="refuses external Ray configuration.*unset"):
        driver._configure_single_node_env()

    assert driver.os.environ["RAY_ADDRESS"] == "unsafe-external-value"
    monkeypatch.delenv("MILES_SCRIPT_EXTERNAL_RAY")
    monkeypatch.delenv("RAY_ADDRESS")
    monkeypatch.setenv("MASTER_ADDR", "10.0.0.1")
    with pytest.raises(RuntimeError, match="refuses a non-loopback MASTER_ADDR.*unset"):
        driver._configure_single_node_env()
    monkeypatch.delenv("MASTER_ADDR")
    driver._configure_single_node_env()

    assert all(name not in driver.os.environ for name in driver._PROXY_ENV_VARS)


def test_full_rollout_alias_runs_eager_then_graph_and_prepares_once(monkeypatch) -> None:
    driver = _load_single_node_driver(monkeypatch)
    calls = []

    def run(mode, *, prepare=True):
        calls.append((mode, prepare))
        return [mode]

    monkeypatch.setattr(
        driver,
        "_parse_args",
        lambda: SimpleNamespace(profile="poc-h16", stage="full-rollout"),
    )
    monkeypatch.setattr(driver, "_run_full_rollout", run)
    monkeypatch.setattr(
        driver,
        "_assert_full_rollout_parity",
        lambda eager, graph: calls.append((eager, graph)),
    )

    driver.main()

    assert calls == [("eager", True), ("graph", False), ("eager", "graph")]


def test_full_rollout_alias_requires_token_and_bf16_logprob_parity(monkeypatch) -> None:
    driver = _load_single_node_driver(monkeypatch)
    eager = SimpleNamespace(tokens=[10, 20, 30], response_length=1, rollout_log_probs=[-0.50])
    graph = SimpleNamespace(tokens=[10, 20, 30], response_length=1, rollout_log_probs=[-0.52])

    driver._assert_full_rollout_parity(eager, graph)

    graph.tokens[-1] = 31
    with pytest.raises(RuntimeError, match="different response token IDs"):
        driver._assert_full_rollout_parity(eager, graph)
    graph.tokens[-1] = 30
    graph.rollout_log_probs[0] = -0.54
    with pytest.raises(RuntimeError, match="BF16 tolerance is 0.03"):
        driver._assert_full_rollout_parity(eager, graph)


def test_single_node_driver_validates_a_fresh_finite_rollout_capture(monkeypatch, tmp_path) -> None:
    driver = _load_single_node_driver(monkeypatch)
    capture = tmp_path / "rollout" / "0.pt"
    capture.parent.mkdir()
    capture.write_bytes(b"non-empty fixture")
    validated = []

    class _Status(Enum):
        COMPLETED = "completed"
        TRUNCATED = "truncated"

    class _Sample:
        Status = _Status
        index = 0
        status = _Status.COMPLETED
        response_length = 2
        rollout_log_probs = [-0.1, -0.2]
        reward = 1.0

        @classmethod
        def from_dict(cls, _value):
            return cls()

        def validate(self) -> None:
            validated.append(self)

    _stub_module(
        monkeypatch,
        "torch",
        load=lambda _path, *, weights_only: {"rollout_id": 0, "samples": [{}]},
    )
    _stub_module(monkeypatch, "miles.utils.types", Sample=_Sample)

    driver._assert_rollout_capture(
        str(tmp_path / "rollout" / "{rollout_id}.pt"),
        expected_samples=1,
        previous_mtime_ns=capture.stat().st_mtime_ns - 1,
    )

    assert len(validated) == 1
