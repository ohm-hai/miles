import pytest

from tests.fast.launch_scripts.model_args_harness import expand_model_args


def _flag_value(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


@pytest.mark.parametrize("model_type", ["glm5.2-744B-A40B", "glm5.2-744B-A40B_5layer"])
def test_glm5_2_uses_unscaled_rope(model_type: str) -> None:
    args = expand_model_args(model_type)

    assert _flag_value(args, "--rope-type") == "rope"
    assert _flag_value(args, "--rotary-base") == "8000000"
    assert _flag_value(args, "--rotary-scaling-factor") == "1"
