#!/usr/bin/env python3
"""Build the aligned 151-point Open-Meteo mean-sea-level-pressure tensor."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stormengine_dl.data.open_meteo import (  # noqa: E402
    load_download_pressure_chunks,
    load_marine_points,
)
from stormengine_dl.data import load_era5_target_grid  # noqa: E402
from stormengine_dl.source_alignment import bilinear_sample_grid  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        default="data_external/open_meteo/raw/20260801_20260808/icon2i",
    )
    parser.add_argument("--registry", default="data/stations_registry.csv")
    parser.add_argument(
        "--processed",
        default="data_external/open_meteo/processed/20260801_20260808/icon2i_hourly_marine_pressure.npz",
    )
    parser.add_argument(
        "--manifest",
        default="data/manifests/open_meteo_pressure_20260801_20260808.json",
    )
    parser.add_argument("--era5t-instant")
    parser.add_argument("--era5t-accum")
    args = parser.parse_args()
    if bool(args.era5t_instant) != bool(args.era5t_accum):
        raise ValueError("--era5t-instant and --era5t-accum must be supplied together")

    raw_dir = ROOT / args.raw_dir
    points = load_marine_points(ROOT / args.registry)
    batch = load_download_pressure_chunks(raw_dir, points)
    if batch.values.shape != (169, 151, 1):
        raise ValueError(f"Unexpected pressure tensor shape: {batch.values.shape}")
    if not batch.value_mask.all():
        raise ValueError("Open-Meteo pressure_msl is not complete for all 169 x 151 cells")
    if not np.isfinite(batch.values).all():
        raise ValueError("Open-Meteo pressure tensor contains non-finite values")

    processed = ROOT / args.processed
    processed.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        processed,
        times=batch.times,
        station_ids=np.asarray(batch.station_ids),
        variable_names=np.asarray(batch.variable_names),
        values=batch.values,
        value_mask=batch.value_mask,
        observation_age=batch.observation_age,
        coordinates=batch.coordinates,
        station_static=batch.station_static,
        source_type=np.asarray(batch.source_type),
        returned_coordinates=batch.returned_coordinates,
        returned_elevation=batch.returned_elevation,
    )
    pressure = batch.values[:, :, 0]
    manifest = {
        "schema_version": 1,
        "product": "Open-Meteo Historical Forecast API",
        "model": "italia_meteo_arpae_icon_2i",
        "role": "model-derived marine mean-sea-level-pressure support",
        "source_field": "pressure_msl",
        "contract_variable": "msl",
        "unit": "hPa",
        "time_start": str(batch.times[0]),
        "time_end": str(batch.times[-1]),
        "hour_count": len(batch.times),
        "station_count": len(batch.station_ids),
        "shape": list(batch.values.shape),
        "valid_cells": int(batch.value_mask.sum()),
        "total_cells": int(batch.value_mask.size),
        "coverage_fraction": float(batch.value_mask.mean()),
        "minimum_hpa": float(pressure.min()),
        "maximum_hpa": float(pressure.max()),
        "mean_hpa": float(pressure.mean()),
        "processed_external": {
            "path": args.processed,
            "bytes": processed.stat().st_size,
            "sha256": sha256(processed),
        },
        "raw_download_manifest": str(
            Path(args.raw_dir) / "download_manifest.json"
        ).replace("\\", "/"),
        "surface_pressure_excluded": True,
        "surface_pressure_exclusion_reason": "returned model cells can have terrain elevation and are not interchangeable with mean-sea-level pressure",
        "future_leakage_rule": "input windows may use only pressure valid_time <= forecast origin",
    }
    if args.era5t_instant:
        era5t = load_era5_target_grid(
            Path(args.era5t_instant), Path(args.era5t_accum), ["msl"]
        )
        indices = era5t.indices_for(batch.times)
        sampled = bilinear_sample_grid(
            era5t.values[indices],
            era5t.latitudes,
            era5t.longitudes,
            batch.coordinates,
        )[:, :, 0]
        difference = pressure - sampled
        manifest["same_time_same_coordinate_era5t_alignment"] = {
            "reference": "ERA5T bilinearly sampled at the 151 requested coordinates",
            "count": int(difference.size),
            "bias_hpa": float(difference.mean()),
            "mae_hpa": float(np.abs(difference).mean()),
            "rmse_hpa": float(np.sqrt(np.square(difference).mean())),
            "correlation": float(
                np.corrcoef(pressure.reshape(-1), sampled.reshape(-1))[0, 1]
            ),
            "open_meteo_mean_hpa": float(pressure.mean()),
            "era5t_mean_hpa": float(sampled.mean()),
            "interpretation": "source-minus-ERA5T; diagnostic alignment, not a claim that ERA5T is error-free truth",
        }
    manifest_path = ROOT / args.manifest
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
