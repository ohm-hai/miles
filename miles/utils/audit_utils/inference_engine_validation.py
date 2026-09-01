from typing import Any

from miles.utils.audit_utils.checksum_utils import flatten_inference_engine_checksums
from miles.utils.audit_utils.event_logger.logger import get_event_logger, is_event_logger_initialized
from miles.utils.audit_utils.event_logger.models import InferenceEngineWeightChecksumEvent


async def maybe_validate_and_log_inference_engine_weights(
    *,
    args: Any,
    rollout_manager: Any,
    rollout_id: int | None,
) -> None:
    """Validate and record every rollout engine after one weight update.

    Engine-wide version/checksum reads are expensive for large models, so event
    capture plus a CI/audit flag is the explicit opt-in. Normal training returns
    before issuing any Ray or SGLang request.
    """
    if not is_event_logger_initialized():
        return
    if not any(
        getattr(args, flag, False)
        for flag in (
            "ci_test",
            "enable_event_analyzer",
            "check_weight_update_equal",
            "save_local_weight_checksum",
        )
    ):
        return
    if args.debug_train_only or args.debug_rollout_only:
        return

    expected_version, engine_versions = await rollout_manager.get_updatable_engine_weight_versions.remote()
    expected_version, engine_versions = _validate_engine_weight_versions(
        expected_version=expected_version,
        engine_versions=engine_versions,
    )

    check_weights_result = await rollout_manager.check_weights.remote("checksum")
    engine_checksums = flatten_inference_engine_checksums(check_weights_result)
    if len(engine_checksums) != len(engine_versions):
        raise RuntimeError(
            "Rollout engine audit returned a different number of checksum and version responses: "
            f"{len(engine_checksums)} checksums vs {len(engine_versions)} versions"
        )

    get_event_logger().log(
        InferenceEngineWeightChecksumEvent,
        dict(
            rollout_id=rollout_id,
            expected_weight_version=expected_version,
            engine_weight_versions=engine_versions,
            engine_checksums=engine_checksums,
        ),
    )
    _validate_engine_checksum_consistency(engine_checksums)


def _validate_engine_weight_versions(
    *,
    expected_version: int | str | None,
    engine_versions: list[int | str | None],
) -> tuple[str, list[str]]:
    if expected_version is None:
        raise RuntimeError("Rollout manager has no weight version after trainer-to-engine sync")
    if not engine_versions:
        raise RuntimeError("Rollout engine audit returned no engine weight versions")

    normalized_expected = str(expected_version)
    normalized_versions = [str(version) for version in engine_versions]
    mismatches = {
        engine_index: version
        for engine_index, version in enumerate(normalized_versions)
        if version != normalized_expected
    }
    if mismatches:
        raise RuntimeError(
            f"Rollout engine weight version mismatch: expected {normalized_expected}, "
            f"mismatched engines {mismatches}"
        )
    return normalized_expected, normalized_versions


def _validate_engine_checksum_consistency(engine_checksums: list[dict[str, str]]) -> None:
    baseline = engine_checksums[0]
    mismatches = [
        engine_index
        for engine_index, checksums in enumerate(engine_checksums[1:], start=1)
        if checksums != baseline
    ]
    if mismatches:
        raise RuntimeError(
            "Rollout engine weight checksum mismatch after trainer-to-engine sync: "
            f"engine 0 differs from engines {mismatches}"
        )
