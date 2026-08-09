#!/usr/bin/env python3
"""Precompute normalized hourly station inputs and dense targets for fast training."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import yaml

from stormengine_dl.data import Era5SequenceDataset, NormalizationStats


def _resolve(repo_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/era5_2010_2017.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(_resolve(repo_root, args.config).read_text(encoding="utf-8"))
    data = config["data"]
    data_root = _resolve(repo_root, data["era5_root"])
    cache_value = Path(data["training_cache"])
    cache_dir = cache_value if cache_value.is_absolute() else data_root / cache_value
    metadata_path = cache_dir / "metadata.json"
    if metadata_path.exists() and not args.overwrite:
        print(f"Training cache already exists: {cache_dir}")
        print("Use --overwrite only when variables, coordinates, or normalization changed.")
        return 0
    cache_dir.mkdir(parents=True, exist_ok=True)

    all_years = sorted(
        set(data["train_years"] + data["validation_years"] + data["test_years"])
    )
    normalization_path = _resolve(repo_root, data["normalization_stats"])
    normalization = NormalizationStats.load(normalization_path)
    dataset = Era5SequenceDataset.from_station_registry(
        manifest_path=_resolve(repo_root, data["era5_manifest"]),
        data_root=data_root,
        station_registry_path=_resolve(repo_root, data["station_registry"]),
        station_profile=data["station_profile"],
        input_variables=data["input_variables"],
        target_variables=data["target_variables"],
        history_hours=int(data["history_hours"]),
        forecast_hours=int(data["forecast_hours"]),
        years=all_years,
        cache_months=1,
        normalization_path=normalization_path,
    )

    total = dataset.times.size
    stations = dataset.station_coordinates.shape[0]
    height, width = dataset.latitudes.size, dataset.longitudes.size
    points = np.lib.format.open_memmap(
        cache_dir / "point_values.npy",
        mode="w+",
        dtype=np.float32,
        shape=(total, stations, len(data["input_variables"])),
    )
    targets = np.lib.format.open_memmap(
        cache_dir / "target_grids.npy",
        mode="w+",
        dtype=np.float32,
        shape=(total, len(data["target_variables"]), height, width),
    )
    np.save(cache_dir / "times.npy", dataset.times.astype("datetime64[ns]"))
    np.save(cache_dir / "point_coords.npy", dataset.normalized_station_coordinates)
    np.save(cache_dir / "point_static.npy", dataset.station_features)

    offset = 0
    started = time.time()
    for month_index, month in enumerate(dataset.months):
        arrays = dataset._load_month(month_index)
        input_grids = np.stack([arrays[name] for name in data["input_variables"]], axis=1)
        month_points = dataset._sample_stations(input_grids)
        month_targets = np.stack([arrays[name] for name in data["target_variables"]], axis=1)
        for channel, name in enumerate(data["input_variables"]):
            month_points[:, :, channel] = normalization.normalize(
                name, month_points[:, :, channel]
            )
        for channel, name in enumerate(data["target_variables"]):
            month_targets[:, channel] = normalization.normalize(name, month_targets[:, channel])
        stop = offset + month.times.size
        points[offset:stop] = month_points
        targets[offset:stop] = month_targets
        offset = stop
        elapsed = time.time() - started
        print(
            f"[{month_index + 1:02d}/{len(dataset.months):02d}] "
            f"{month.year}-{month.month:02d} cached | {offset:,}/{total:,} hours | "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )
    points.flush()
    targets.flush()

    metadata = {
        "format_version": 1,
        "normalized": True,
        "years": all_years,
        "time_count": int(total),
        "time_start": str(dataset.times[0]),
        "time_end": str(dataset.times[-1]),
        "input_variables": list(data["input_variables"]),
        "target_variables": list(data["target_variables"]),
        "station_profile": data["station_profile"],
        "station_count": int(stations),
        "grid_shape": [int(height), int(width)],
        "normalization_sha256": _sha256(normalization_path),
        "point_values_shape": list(points.shape),
        "target_grids_shape": list(targets.shape),
        "total_bytes": int(points.nbytes + targets.nbytes),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(f"Training cache built: {cache_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
