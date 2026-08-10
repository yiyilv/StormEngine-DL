#!/usr/bin/env python3
"""Build fixed-registry hourly tensors from official MeteoHub JSON Lines files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stormengine_dl.data.official_observations import iter_meteohub_observations  # noqa: E402
from stormengine_dl.data.operational_adapter import build_operational_tensors  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", nargs="+", type=Path)
    parser.add_argument("--registry", type=Path, default=ROOT / "data" / "stations_registry.csv")
    parser.add_argument("--from-utc", required=True, help="First hourly UTC timestamp, inclusive")
    parser.add_argument("--to-utc", required=True, help="Last hourly UTC timestamp, inclusive")
    parser.add_argument("--max-age-minutes", type=float, default=60.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = np.datetime64(args.from_utc, "h")
    stop = np.datetime64(args.to_utc, "h")
    if stop < start:
        raise ValueError("--to-utc must not precede --from-utc")
    times = np.arange(start, stop + np.timedelta64(1, "h"), np.timedelta64(1, "h"))
    batch = build_operational_tensors(
        iter_meteohub_observations(args.jsonl),
        args.registry,
        times,
        max_age_minutes=args.max_age_minutes,
    )
    if np.isnan(batch.values[batch.value_mask]).any():
        raise ValueError("Unmasked NaN reached operational values")
    target_times = np.broadcast_to(batch.times[:, None, None], batch.source_time.shape)
    invariants = {
        "finite_valid_values": bool(np.isfinite(batch.values[batch.value_mask]).all()),
        "masked_values_are_zero": bool((batch.values[~batch.value_mask] == 0).all()),
        "valid_age_within_limit": bool(
            (
                (batch.observation_age_minutes[batch.value_mask] >= 0)
                & (batch.observation_age_minutes[batch.value_mask] <= args.max_age_minutes)
            ).all()
        ),
        "masked_age_is_minus_one": bool(
            (batch.observation_age_minutes[~batch.value_mask] == -1).all()
        ),
        "no_future_source": bool(
            (batch.source_time[batch.value_mask] <= target_times[batch.value_mask]).all()
        ),
        "station_present_matches_mask": bool(
            np.array_equal(batch.station_present, batch.value_mask.any(axis=-1))
        ),
    }
    if not all(invariants.values()):
        raise ValueError(f"Operational tensor invariants failed: {invariants}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        times=batch.times,
        station_ids=np.asarray(batch.station_ids),
        variable_names=np.asarray(batch.variable_names),
        values=batch.values,
        value_mask=batch.value_mask,
        source_time=batch.source_time,
        observation_age_minutes=batch.observation_age_minutes,
        station_present=batch.station_present,
        coordinates=batch.coordinates,
        station_static=batch.station_static,
        source_type=np.asarray(batch.source_type),
    )
    metadata = {
        "format_version": 1,
        "from_utc": str(batch.times[0]),
        "to_utc": str(batch.times[-1]),
        "hour_count": int(batch.times.size),
        "station_count": len(batch.station_ids),
        "variable_names": list(batch.variable_names),
        "max_age_minutes": args.max_age_minutes,
        "diagnostics": batch.diagnostics,
        "variable_coverage": {
            name: {
                "valid_cells": int(batch.value_mask[:, :, index].sum()),
                "stations_with_any": int(batch.value_mask[:, :, index].any(axis=0).sum()),
                "coverage_fraction": float(batch.value_mask[:, :, index].mean()),
            }
            for index, name in enumerate(batch.variable_names)
        },
        "invariants": invariants,
        "source_files": [path.name for path in args.jsonl],
        "scientific_contract": {
            "pressure": "station_pressure_hpa; no silent conversion to ERA5 msl",
            "wind": "meteorological direction converted to u10/v10 before alignment",
            "precipitation": "complete non-overlapping preceding-hour coverage only",
            "leakage": "latest observation at or before target hour within maximum age",
        },
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
