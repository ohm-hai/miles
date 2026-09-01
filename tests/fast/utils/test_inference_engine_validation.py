from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from miles.utils.audit_utils.inference_engine_validation import (
    _validate_engine_checksum_consistency,
    _validate_engine_weight_versions,
    maybe_validate_and_log_inference_engine_weights,
)
from miles.utils.audit_utils.event_logger.models import InferenceEngineWeightChecksumEvent

pytestmark = pytest.mark.asyncio


def _args(
    *,
    ci_test: bool = True,
    debug_train_only: bool = False,
    debug_rollout_only: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        ci_test=ci_test,
        debug_train_only=debug_train_only,
        debug_rollout_only=debug_rollout_only,
    )


def _checksum_response(checksums: list[dict[str, str]]) -> list[list[dict]]:
    return [
        [
            {
                "success": True,
                "message": "ok",
                "ranks": [
                    {
                        "checksums": checksum,
                        "parallelism_info": [{"role": "target", "rank": 0}],
                    }
                ],
            }
            for checksum in checksums
        ]
    ]


def _rollout_manager(
    *,
    expected_version: int | str | None = 3,
    engine_versions: list[int | str | None] | None = None,
    checksums: list[dict[str, str]] | None = None,
) -> MagicMock:
    manager = MagicMock()
    manager.get_updatable_engine_weight_versions.remote = AsyncMock(
        return_value=(expected_version, [3, "3"] if engine_versions is None else engine_versions)
    )
    manager.check_weights.remote = AsyncMock(
        return_value=_checksum_response([{"w": "same"}, {"w": "same"}] if checksums is None else checksums)
    )
    return manager


async def test_disabled_event_capture_issues_no_engine_requests() -> None:
    manager = _rollout_manager()

    with patch(
        "miles.utils.audit_utils.inference_engine_validation.is_event_logger_initialized",
        return_value=False,
    ):
        await maybe_validate_and_log_inference_engine_weights(
            args=_args(),
            rollout_manager=manager,
            rollout_id=2,
        )

    manager.get_updatable_engine_weight_versions.remote.assert_not_awaited()
    manager.check_weights.remote.assert_not_awaited()


async def test_event_capture_without_validation_flag_issues_no_engine_requests() -> None:
    manager = _rollout_manager()

    with patch(
        "miles.utils.audit_utils.inference_engine_validation.is_event_logger_initialized",
        return_value=True,
    ):
        await maybe_validate_and_log_inference_engine_weights(
            args=_args(ci_test=False),
            rollout_manager=manager,
            rollout_id=2,
        )

    manager.get_updatable_engine_weight_versions.remote.assert_not_awaited()
    manager.check_weights.remote.assert_not_awaited()


@pytest.mark.parametrize("debug_mode", ["debug_train_only", "debug_rollout_only"])
async def test_debug_modes_issue_no_engine_requests(debug_mode: str) -> None:
    manager = _rollout_manager()

    with patch(
        "miles.utils.audit_utils.inference_engine_validation.is_event_logger_initialized",
        return_value=True,
    ):
        await maybe_validate_and_log_inference_engine_weights(
            args=_args(**{debug_mode: True}),
            rollout_manager=manager,
            rollout_id=2,
        )

    manager.get_updatable_engine_weight_versions.remote.assert_not_awaited()
    manager.check_weights.remote.assert_not_awaited()


async def test_logs_versions_and_checksums_for_every_engine() -> None:
    manager = _rollout_manager()
    event_logger = MagicMock()

    with (
        patch(
            "miles.utils.audit_utils.inference_engine_validation.is_event_logger_initialized",
            return_value=True,
        ),
        patch(
            "miles.utils.audit_utils.inference_engine_validation.get_event_logger",
            return_value=event_logger,
        ),
    ):
        await maybe_validate_and_log_inference_engine_weights(
            args=_args(),
            rollout_manager=manager,
            rollout_id=2,
        )

    manager.get_updatable_engine_weight_versions.remote.assert_awaited_once_with()
    manager.check_weights.remote.assert_awaited_once_with("checksum")
    event_logger.log.assert_called_once_with(
        InferenceEngineWeightChecksumEvent,
        {
            "rollout_id": 2,
            "expected_weight_version": "3",
            "engine_weight_versions": ["3", "3"],
            "engine_checksums": [{"rank0/w": "same"}, {"rank0/w": "same"}],
        },
    )


async def test_version_mismatch_fails_before_expensive_checksum_request() -> None:
    manager = _rollout_manager(engine_versions=[3, 2])

    with (
        patch(
            "miles.utils.audit_utils.inference_engine_validation.is_event_logger_initialized",
            return_value=True,
        ),
        pytest.raises(RuntimeError, match="mismatched engines.*1.*2"),
    ):
        await maybe_validate_and_log_inference_engine_weights(
            args=_args(),
            rollout_manager=manager,
            rollout_id=2,
        )

    manager.check_weights.remote.assert_not_awaited()


async def test_missing_checksum_response_fails_engine_cardinality_check() -> None:
    manager = _rollout_manager(engine_versions=[3, 3], checksums=[{"w": "e0"}])

    with (
        patch(
            "miles.utils.audit_utils.inference_engine_validation.is_event_logger_initialized",
            return_value=True,
        ),
        pytest.raises(RuntimeError, match="1 checksums vs 2 versions"),
    ):
        await maybe_validate_and_log_inference_engine_weights(
            args=_args(),
            rollout_manager=manager,
            rollout_id=2,
        )


async def test_checksum_divergence_is_logged_then_fails_immediately() -> None:
    manager = _rollout_manager(checksums=[{"w": "e0"}, {"w": "different"}])
    event_logger = MagicMock()

    with (
        patch(
            "miles.utils.audit_utils.inference_engine_validation.is_event_logger_initialized",
            return_value=True,
        ),
        patch(
            "miles.utils.audit_utils.inference_engine_validation.get_event_logger",
            return_value=event_logger,
        ),
        pytest.raises(RuntimeError, match="checksum mismatch.*engine 0 differs from engines.*1"),
    ):
        await maybe_validate_and_log_inference_engine_weights(
            args=_args(),
            rollout_manager=manager,
            rollout_id=2,
        )

    event_logger.log.assert_called_once()


async def test_checksum_consistency_accepts_identical_engine_manifests() -> None:
    _validate_engine_checksum_consistency([{"rank0/w": "same"}, {"rank0/w": "same"}])


@pytest.mark.parametrize(
    ("expected_version", "engine_versions", "message"),
    [
        (None, [0], "no weight version"),
        (0, [], "no engine weight versions"),
    ],
)
async def test_incomplete_version_state_fails_loud(
    expected_version: int | None,
    engine_versions: list[int],
    message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        _validate_engine_weight_versions(
            expected_version=expected_version,
            engine_versions=engine_versions,
        )
