from types import SimpleNamespace

import pytest

pytest.importorskip("sglang")

from miles.ray.rollout.rollout_manager import RolloutManager

pytestmark = pytest.mark.asyncio


class _RemoteValue:
    def __init__(self, value) -> None:
        self._value = value
        self.calls = 0

    def remote(self):
        self.calls += 1

        async def result():
            return self._value

        return result()


def _manager(*, versions: list[str], expected: int = 5):
    manager = object.__new__(RolloutManager.__ray_actor_class__)
    remote_values = [_RemoteValue(version) for version in versions]
    engines = [
        SimpleNamespace(
            is_allocated=True,
            actor_handle=SimpleNamespace(get_weight_version=remote_value),
        )
        for remote_value in remote_values
    ]
    manager.servers = {
        "actor": SimpleNamespace(
            update_weights=True,
            engines=engines,
        )
    }
    manager.weight_version = expected
    return manager, remote_values


async def test_collects_every_logical_engine_version_in_order() -> None:
    manager, remote_values = _manager(versions=["5", "5", "5"])

    expected, versions = await manager.get_updatable_engine_weight_versions()

    assert expected == 5
    assert versions == ["5", "5", "5"]
    assert [remote.calls for remote in remote_values] == [1, 1, 1]


async def test_rejects_partially_allocated_engine_set() -> None:
    manager, _ = _manager(versions=["5", "5"])
    manager.servers["actor"].engines[1].is_allocated = False

    with pytest.raises(RuntimeError, match="every rollout engine allocated"):
        await manager.get_updatable_engine_weight_versions()


async def test_no_updatable_server_returns_empty_version_set() -> None:
    manager, _ = _manager(versions=[])
    manager.servers["actor"].update_weights = False

    assert await manager.get_updatable_engine_weight_versions() == (5, [])
