#!/usr/bin/env python3
"""Create a human-readable preview of the real DPC-to-V7 conversion."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stormengine_dl.data import load_dpc_v7_input  # noqa: E402
from stormengine_dl.data.operational_adapter import load_fixed_registry  # noqa: E402


SOURCE_NAMES = ("u10", "v10", "wind_gust_max", "t2m", "tp")


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def serial_time(value: np.datetime64) -> str:
    return str(value.astype("datetime64[m]")) + "Z"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data_external/meteohub/processed/20260801_20260808/official_hourly_physical.npz",
    )
    parser.add_argument("--normalization", default="data/normalization/era5_2010_2015.json")
    parser.add_argument("--registry", default="data/stations_registry.csv")
    parser.add_argument("--output-dir", default="artifacts/v7_dpc_adapter_preview")
    parser.add_argument("--history-hours", type=int, default=12)
    args = parser.parse_args()
    if args.history_hours < 1:
        raise ValueError("history-hours must be positive")

    registry = load_fixed_registry(resolve(args.registry), include_virtual=False)
    batch = load_dpc_v7_input(
        resolve(args.input),
        resolve(args.normalization),
        expected_station_ids=registry.station_ids,
    )
    with np.load(resolve(args.input), allow_pickle=False) as source:
        source_names = tuple(str(value) for value in source["variable_names"].tolist())
        source_indices = [source_names.index(name) for name in SOURCE_NAMES]
        raw = np.stack([source["values"][:, :, index] for index in source_indices], -1)
        raw_mask = np.stack([source["value_mask"][:, :, index] for index in source_indices], -1)
        raw_age = np.stack(
            [source["observation_age_minutes"][:, :, index] for index in source_indices], -1
        )

    hourly_cells = batch.value_mask.sum(axis=(1, 2)).astype(np.int64)
    if len(hourly_cells) < args.history_hours:
        raise ValueError("DPC tensor is shorter than the requested history window")
    window_scores = np.convolve(
        hourly_cells, np.ones(args.history_hours, dtype=np.int64), mode="valid"
    )
    window_start = int(window_scores.argmax())
    window_stop = window_start + args.history_hours
    representative = window_start + int(hourly_cells[window_start:window_stop].argmax())
    representative_mask = batch.value_mask[representative]

    pattern_counts = Counter(
        "".join("1" if value else "0" for value in row) for row in representative_mask
    )
    variable_report: dict[str, dict[str, float | int]] = {}
    for channel, name in enumerate(batch.variable_names):
        selected = batch.values[:, :, channel][batch.value_mask[:, :, channel]]
        variable_report[name] = {
            "valid_cells": int(batch.value_mask[:, :, channel].sum()),
            "coverage_fraction": float(batch.value_mask[:, :, channel].mean()),
            "stations_with_any": int(batch.value_mask[:, :, channel].any(axis=0).sum()),
            "normalized_mean": float(selected.mean()),
            "normalized_std": float(selected.std()),
            "normalized_p01": float(np.quantile(selected, 0.01)),
            "normalized_p50": float(np.quantile(selected, 0.50)),
            "normalized_p99": float(np.quantile(selected, 0.99)),
            "normalized_abs_gt_5_fraction": float((np.abs(selected) > 5).mean()),
        }

    report = {
        "format_version": 1,
        "purpose": "DPC-to-V7 interface preview; not a forecast quality evaluation",
        "source": str(Path(args.input).as_posix()),
        "contract": {
            "history_hours": args.history_hours,
            "station_count": len(batch.station_ids),
            "variables": list(batch.variable_names),
            "values_shape": list(batch.values.shape),
            "mask_shape": list(batch.value_mask.shape),
            "age_units": "fraction of 60 minutes in [0,1]",
            "coordinate_domain": "lat/lon normalized to [0,1] over 39-46.5N, 12-20E",
            "normalization": "ERA5 2010-2015 training statistics",
        },
        "selected_window": {
            "start": serial_time(batch.times[window_start]),
            "end": serial_time(batch.times[window_stop - 1]),
            "valid_variable_cells": int(window_scores[window_start]),
        },
        "representative_hour": {
            "time": serial_time(batch.times[representative]),
            "stations_with_any": int(representative_mask.any(axis=1).sum()),
            "stations_with_all_five": int(representative_mask.all(axis=1).sum()),
            "valid_by_variable": {
                name: int(representative_mask[:, channel].sum())
                for channel, name in enumerate(batch.variable_names)
            },
            "mask_patterns": dict(sorted(pattern_counts.items())),
        },
        "variables": variable_report,
        "invariants": {
            "station_order_matches_registry": batch.station_ids == registry.station_ids,
            "missing_values_are_zero": bool((batch.values[~batch.value_mask] == 0).all()),
            "missing_age_is_zero": bool((batch.observation_age[~batch.value_mask] == 0).all()),
            "valid_age_in_range": bool(
                ((batch.observation_age[batch.value_mask] >= 0)
                 & (batch.observation_age[batch.value_mask] <= 1)).all()
            ),
            "finite_model_values": bool(np.isfinite(batch.values).all()),
            "coordinates_in_unit_square": bool(
                ((batch.coordinates >= 0) & (batch.coordinates <= 1)).all()
            ),
        },
    }

    output = resolve(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "preview.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        output / "representative_12h_v7_input.npz",
        times=batch.times[window_start:window_stop],
        station_ids=np.asarray(batch.station_ids),
        variable_names=np.asarray(batch.variable_names),
        values=batch.values[window_start:window_stop],
        value_mask=batch.value_mask[window_start:window_stop],
        observation_age=batch.observation_age[window_start:window_stop],
        station_present=batch.station_present[window_start:window_stop],
        coordinates=batch.coordinates,
        station_static=batch.station_static,
    )

    with (output / "coverage_by_hour.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_utc", *batch.variable_names, "stations_any", "stations_all_five"])
        for time_index, time in enumerate(batch.times):
            current = batch.value_mask[time_index]
            writer.writerow([
                serial_time(time),
                *(int(current[:, channel].sum()) for channel in range(len(batch.variable_names))),
                int(current.any(axis=1).sum()),
                int(current.all(axis=1).sum()),
            ])

    with (output / "representative_hour.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["station_id", "latitude", "longitude", "mask_pattern"]
        for name in batch.variable_names:
            fields.extend([f"{name}_raw", f"{name}_normalized", f"{name}_age_minutes"])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for station, station_id in enumerate(batch.station_ids):
            row: dict[str, object] = {
                "station_id": station_id,
                "latitude": float(registry.coordinates[station, 0]),
                "longitude": float(registry.coordinates[station, 1]),
                "mask_pattern": "".join("1" if value else "0" for value in representative_mask[station]),
            }
            for channel, name in enumerate(batch.variable_names):
                valid = bool(representative_mask[station, channel])
                row[f"{name}_raw"] = float(raw[representative, station, channel]) if valid else ""
                row[f"{name}_normalized"] = float(batch.values[representative, station, channel]) if valid else ""
                row[f"{name}_age_minutes"] = float(raw_age[representative, station, channel]) if valid else ""
            writer.writerow(row)

    sample_indices: list[int] = []
    for desired in ("11111", "11011", "00011", "00001", "00000"):
        matches = [
            index for index, row in enumerate(representative_mask)
            if "".join("1" if value else "0" for value in row) == desired
        ]
        sample_indices.extend(matches[:2])

    def cell(station: int, channel: int) -> str:
        if not representative_mask[station, channel]:
            return "missing"
        return (
            f"{raw[representative, station, channel]:.2f} -> "
            f"{batch.values[representative, station, channel]:.2f} "
            f"({raw_age[representative, station, channel]:.0f}m)"
        )

    lines = [
        "# DPC to V7 input preview",
        "",
        "This is an interface diagnostic, not a forecast-accuracy result.",
        "",
        f"- Selected 12-hour window: {report['selected_window']['start']} to {report['selected_window']['end']}",
        f"- Representative hour: {report['representative_hour']['time']}",
        f"- Stations with any V7 variable: {report['representative_hour']['stations_with_any']}/239",
        f"- Stations with all five variables: {report['representative_hour']['stations_with_all_five']}/239",
        "",
        "Each value below is `physical value -> ERA5-normalized value (observation age)`.",
        "",
        "| station | mask | u10 | v10 | gust | t2m | tp |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for station in sample_indices:
        pattern = "".join("1" if value else "0" for value in representative_mask[station])
        lines.append(
            f"| `{batch.station_ids[station]}` | `{pattern}` | "
            + " | ".join(cell(station, channel) for channel in range(len(batch.variable_names)))
            + " |"
        )
    lines.extend([
        "",
        "Mask order is `u10,v10,i10fg,t2m,tp`; `1` means observed and `0` means missing.",
        "Missing model values and ages are zero, while the variable-level mask remains false.",
    ])
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output), **report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
