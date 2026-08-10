"""Validate external MeteoHub JSON Lines files against a versioned manifest."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class MeteoHubFileValidation:
    """Integrity result for one file declared in a MeteoHub manifest."""

    logical_name: str
    filename: str
    path: Path
    expected_bytes: int
    actual_bytes: int | None
    expected_records: int
    actual_records: int | None
    expected_sha256: str
    actual_sha256: str | None

    @property
    def exists(self) -> bool:
        return self.actual_bytes is not None

    @property
    def ok(self) -> bool:
        return (
            self.actual_bytes == self.expected_bytes
            and self.actual_records == self.expected_records
            and self.actual_sha256 == self.expected_sha256
        )

    @property
    def problems(self) -> tuple[str, ...]:
        if not self.exists:
            return ("missing",)
        problems: list[str] = []
        if self.actual_bytes != self.expected_bytes:
            problems.append("size")
        if self.actual_records != self.expected_records:
            problems.append("records")
        if self.actual_sha256 != self.expected_sha256:
            problems.append("sha256")
        return tuple(problems)


def _hash_and_count_lines(path: Path, chunk_size: int = 1024 * 1024) -> tuple[str, int]:
    digest = hashlib.sha256()
    newline_count = 0
    last_byte = b""
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
            newline_count += chunk.count(b"\n")
            last_byte = chunk[-1:]
    record_count = newline_count + int(bool(last_byte) and last_byte != b"\n")
    return digest.hexdigest(), record_count


def load_meteohub_manifest(path: str | Path) -> dict[str, object]:
    manifest_path = Path(path)
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), list):
        raise ValueError(f"Invalid MeteoHub manifest structure: {manifest_path}")
    return manifest


def manifest_raw_root(manifest_path: str | Path, manifest: dict[str, object]) -> Path:
    """Resolve the manifest's repository-relative external-data directory."""
    path = Path(manifest_path).resolve()
    raw_root = manifest.get("raw_root")
    if not isinstance(raw_root, str) or not raw_root:
        raise ValueError(f"Manifest does not define raw_root: {path}")
    try:
        repository_root = path.parents[2]
    except IndexError as exc:
        raise ValueError(f"Manifest must be stored below the repository root: {path}") from exc
    return repository_root / raw_root


def validate_meteohub_manifest(
    manifest_path: str | Path,
    raw_root: str | Path | None = None,
    *,
    qc_only: bool = False,
) -> list[MeteoHubFileValidation]:
    """Validate selected manifest files without loading their contents into memory."""
    path = Path(manifest_path)
    manifest = load_meteohub_manifest(path)
    root = Path(raw_root) if raw_root is not None else manifest_raw_root(path, manifest)
    results: list[MeteoHubFileValidation] = []
    entries: Iterable[object] = manifest["files"]  # type: ignore[assignment]
    for value in entries:
        if not isinstance(value, dict):
            raise ValueError(f"Invalid file entry in {path}: {value!r}")
        if qc_only and value.get("quality_controlled_only") is not True:
            continue
        filename = str(value["filename"])
        file_path = root / filename
        actual_bytes: int | None = None
        actual_records: int | None = None
        actual_sha256: str | None = None
        if file_path.is_file():
            actual_bytes = file_path.stat().st_size
            actual_sha256, actual_records = _hash_and_count_lines(file_path)
        results.append(
            MeteoHubFileValidation(
                logical_name=str(value["logical_name"]),
                filename=filename,
                path=file_path,
                expected_bytes=int(value["bytes"]),
                actual_bytes=actual_bytes,
                expected_records=int(value["jsonl_records"]),
                actual_records=actual_records,
                expected_sha256=str(value["sha256"]).lower(),
                actual_sha256=actual_sha256,
            )
        )
    return results
