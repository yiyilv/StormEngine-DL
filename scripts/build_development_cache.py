#!/usr/bin/env python3
"""Derive and verify a compact, year-bounded cache from the validated ERA5 cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stormengine_dl.data import build_cache_identity, validate_cache_identity  # noqa: E402


ARRAY_FILES = ("point_values.npy", "target_grids.npy")
STATIC_FILES = ("point_coords.npy", "point_static.npy")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, object]) -> None:
    # Explicit bytes keep identity hashes identical on Windows and Unix.
    path.write_bytes((json.dumps(value, indent=2) + "\n").encode("utf-8"))


def canonical_json_sha256(path: Path) -> str:
    value = json.loads(path.read_text(encoding="utf-8"))
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def copy_subset(source: np.ndarray, destination: Path, indices: np.ndarray, chunk: int) -> None:
    target = np.lib.format.open_memmap(
        destination,
        mode="w+",
        dtype=source.dtype,
        shape=(indices.size, *source.shape[1:]),
    )
    for start in range(0, indices.size, chunk):
        stop = min(indices.size, start + chunk)
        target[start:stop] = source[indices[start:stop]]
        if stop == indices.size or start == 0 or stop % max(chunk, 24 * 30) == 0:
            print(f"  {destination.name}: {stop:,}/{indices.size:,}", flush=True)
    target.flush()


def verify(cache: Path) -> dict[str, object]:
    provenance_path = cache / "derivation.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    for name, expected in provenance["files"].items():
        path = cache / name
        if not path.is_file() or path.stat().st_size != expected["bytes"]:
            failures.append(f"{name}: size")
        elif sha256_file(path) != expected["sha256"]:
            failures.append(f"{name}: sha256")
    if failures:
        raise ValueError(f"Development cache verification failed: {failures}")
    return provenance


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("build", "verify"))
    parser.add_argument("--source-cache", type=Path)
    parser.add_argument("--output-cache", type=Path, required=True)
    parser.add_argument("--years", nargs="+", type=int, default=[2013, 2014, 2015, 2016])
    parser.add_argument("--registry", type=Path, default=ROOT / "data/stations_registry.csv")
    parser.add_argument(
        "--source-identity",
        type=Path,
        default=ROOT / "data/manifests/v7_cache_identity_2010_2017.json",
    )
    parser.add_argument("--identity-output", type=Path)
    parser.add_argument("--chunk-hours", type=int, default=168)
    args = parser.parse_args()
    output = args.output_cache.expanduser().resolve()
    if args.mode == "verify":
        result = verify(output)
        print(json.dumps(result, indent=2))
        print(f"PASS: verified development cache at {output}")
        return 0

    if args.source_cache is None:
        parser.error("--source-cache is required in build mode")
    source = args.source_cache.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty cache: {output}")
    validate_cache_identity(source, args.registry, args.source_identity)
    source_metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    times = np.asarray(np.load(source / "times.npy", mmap_mode="r")).astype("datetime64[ns]")
    all_years = times.astype("datetime64[Y]").astype(np.int64) + 1970
    selected_years = sorted(set(args.years))
    indices = np.flatnonzero(np.isin(all_years, selected_years)).astype(np.int64)
    if indices.size == 0 or not np.array_equal(indices, np.arange(indices[0], indices[-1] + 1)):
        raise ValueError("Selected development years must form one non-empty contiguous block")
    selected_times = times[indices]
    if selected_times[-1] - selected_times[0] != np.timedelta64(indices.size - 1, "h"):
        raise ValueError("Selected development cache contains an hourly time gap")

    output.mkdir(parents=True, exist_ok=True)
    for name in STATIC_FILES:
        shutil.copyfile(source / name, output / name)
    np.save(output / "times.npy", selected_times)
    for name in ARRAY_FILES:
        copy_subset(np.load(source / name, mmap_mode="r"), output / name, indices, args.chunk_hours)

    points = np.load(output / "point_values.npy", mmap_mode="r")
    targets = np.load(output / "target_grids.npy", mmap_mode="r")
    metadata = dict(source_metadata)
    metadata.update(
        {
            "years": selected_years,
            "time_count": int(indices.size),
            "time_start": str(selected_times[0]),
            "time_end": str(selected_times[-1]),
            "point_values_shape": list(points.shape),
            "target_grids_shape": list(targets.shape),
            "total_bytes": int(points.nbytes + targets.nbytes),
            "derived_cache": True,
            "derivation_role": "architecture_development_only",
        }
    )
    write_json(output / "metadata.json", metadata)
    files = {}
    for name in (*ARRAY_FILES, *STATIC_FILES, "times.npy", "metadata.json"):
        path = output / name
        files[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    provenance: dict[str, object] = {
        "schema_version": 1,
        "role": "reusable_architecture_development_cache",
        "selected_years": selected_years,
        # Canonical JSON hashes make provenance identical across LF/CRLF
        # checkouts while retaining semantic source identity.
        "source_identity_canonical_sha256": canonical_json_sha256(args.source_identity),
        "source_metadata_canonical_sha256": canonical_json_sha256(source / "metadata.json"),
        "time_start": str(selected_times[0]),
        "time_end": str(selected_times[-1]),
        "time_count": int(indices.size),
        "files": files,
    }
    write_json(output / "derivation.json", provenance)
    verify(output)
    identity = build_cache_identity(output, args.registry)
    source_identity = json.loads(args.source_identity.read_text(encoding="utf-8"))
    registry_alternates = source_identity.get("registry_sha256_alternates", [])
    accepted_registry_hashes = {
        source_identity.get("registry_sha256"), *registry_alternates
    } - {None, identity["registry_sha256"]}
    if accepted_registry_hashes:
        identity["registry_sha256_alternates"] = sorted(accepted_registry_hashes)
    if args.identity_output is not None:
        args.identity_output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.identity_output, identity)
    print(json.dumps(identity, indent=2))
    print(f"PASS: built development cache at {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
