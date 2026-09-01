"""Low-level artifact integrity checks for the AMD GLM-5.2 launcher."""

import hashlib
from collections.abc import Mapping
from pathlib import Path


def _nonempty_file_size(path: Path) -> int:
    if not path.is_file() or (size := path.stat().st_size) <= 0:
        raise FileNotFoundError(f"Required artifact file {path} is missing or empty")
    return size


def validate_exact_file(path: Path, expected: tuple[int, str], *, description: str) -> int:
    expected_size, expected_sha256 = expected
    actual_size = _nonempty_file_size(path)
    if actual_size != expected_size:
        raise RuntimeError(f"{description} {path} is {actual_size} bytes; expected {expected_size}")
    digest = hashlib.sha256()
    with open(path, "rb") as asset_file:
        for chunk in iter(lambda: asset_file.read(1024 * 1024), b""):
            digest.update(chunk)
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"{description} {path} has SHA-256 {actual_sha256}; expected {expected_sha256}"
        )
    return actual_size


def runtime_hf_inventory(
    checkpoint_dir: Path,
    expected_assets: Mapping[str, tuple[int, str]],
) -> dict[str, int]:
    inventory = {}
    for relative, expected in expected_assets.items():
        path = checkpoint_dir / relative
        inventory[relative] = validate_exact_file(path, expected, description="Runtime HF asset")
    return inventory


def _validate_torch_dist_references(release_dir: Path) -> None:
    # Import only in the internal validator: torch.distributed.checkpoint is
    # expensive and every normal launcher invocation should remain lightweight.
    from torch.distributed.checkpoint import FileSystemReader

    try:
        metadata = FileSystemReader(release_dir).read_metadata()
    except Exception as error:
        raise RuntimeError(f"Could not read torch_dist metadata from {release_dir}") from error
    storage_data = getattr(metadata, "storage_data", None)
    if not isinstance(storage_data, dict) or not storage_data:
        raise RuntimeError(f"{release_dir / '.metadata'} contains no torch_dist storage records")

    required_sizes: dict[Path, int] = {}
    for storage in storage_data.values():
        relative = Path(str(getattr(storage, "relative_path", "")))
        offset = getattr(storage, "offset", None)
        length = getattr(storage, "length", None)
        if (
            not relative.name
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.suffix != ".distcp"
            or not isinstance(offset, int)
            or not isinstance(length, int)
            or offset < 0
            or length <= 0
        ):
            raise RuntimeError(f"{release_dir / '.metadata'} contains an invalid storage record {storage!r}")
        required_sizes[relative] = max(required_sizes.get(relative, 0), offset + length)

    for relative, required_size in required_sizes.items():
        actual_size = _nonempty_file_size(release_dir / relative)
        if actual_size < required_size:
            raise RuntimeError(
                f"torch_dist shard {release_dir / relative} is truncated: "
                f"{actual_size} bytes, metadata requires {required_size}"
            )
