#!/usr/bin/env python3
"""Build fixed-registry DPC mean-sea-level-pressure estimates with provenance."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stormengine_dl.data import load_era5_target_grid  # noqa: E402
from stormengine_dl.data.pressure_reduction import station_pressure_to_msl_hpa  # noqa: E402
from stormengine_dl.source_alignment import bilinear_sample_grid  # noqa: E402


METHOD_INVALID = 0
METHOD_TRAILING_T_RH = 1
METHOD_TRAILING_T_DRY = 2
METHOD_LOW_ELEVATION_STANDARD_T = 3
METHOD_NAMES = {
    METHOD_INVALID: "invalid",
    METHOD_TRAILING_T_RH: "trailing_temperature_and_humidity",
    METHOD_TRAILING_T_DRY: "trailing_temperature_dry_air",
    METHOD_LOW_ELEVATION_STANDARD_T: "standard_temperature_low_elevation_fallback",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_station_id(registry_station_id: str) -> str:
    return registry_station_id.removeprefix("LAND::")


def load_elevations(
    station_ids: tuple[str, ...], station_summary: Path
) -> tuple[np.ndarray, list[dict[str, str]]]:
    with station_summary.open(newline="", encoding="utf-8") as handle:
        rows = {row["station_id"]: row for row in csv.DictReader(handle)}
    elevations = np.full(len(station_ids), np.nan, np.float32)
    metadata: list[dict[str, str]] = []
    for index, registry_id in enumerate(station_ids):
        source_id = source_station_id(registry_id)
        row = rows.get(source_id)
        if row is None:
            metadata.append({"station_id": registry_id, "source_station_id": source_id})
            continue
        value = row.get("elevation_m", "")
        if value:
            elevations[index] = float(value)
        metadata.append(
            {
                "station_id": registry_id,
                "source_station_id": source_id,
                "station_name": row.get("station_name", ""),
                "network": row.get("network", ""),
                "elevation_m": value,
            }
        )
    return elevations, metadata


def build_corrected_msl(
    values: np.ndarray,
    masks: np.ndarray,
    variable_names: tuple[str, ...],
    elevations_m: np.ndarray,
    *,
    trailing_hours: int = 12,
    low_elevation_fallback_m: float = 10.0,
) -> dict[str, np.ndarray]:
    index = {name: position for position, name in enumerate(variable_names)}
    required = {"station_pressure_hpa", "t2m", "relative_humidity"}
    if not required <= set(index):
        raise ValueError(f"DPC tensor lacks pressure-reduction variables: {sorted(required-set(index))}")
    pressure = np.asarray(values[:, :, index["station_pressure_hpa"]], np.float32)
    pressure_mask = np.asarray(masks[:, :, index["station_pressure_hpa"]], bool)
    temperature = np.asarray(values[:, :, index["t2m"]], np.float32)
    temperature_mask = np.asarray(masks[:, :, index["t2m"]], bool)
    humidity = np.asarray(values[:, :, index["relative_humidity"]], np.float32)
    humidity_mask = np.asarray(masks[:, :, index["relative_humidity"]], bool)
    corrected = np.zeros_like(pressure)
    valid = np.zeros_like(pressure_mask)
    method = np.zeros_like(pressure, dtype=np.uint8)
    correction = np.zeros_like(pressure)
    temperature_count = np.zeros_like(pressure, dtype=np.uint8)
    humidity_count = np.zeros_like(pressure, dtype=np.uint8)
    used_temperature = np.full_like(pressure, np.nan)
    used_humidity = np.full_like(pressure, np.nan)

    for time_index, station_index in zip(*np.where(pressure_mask), strict=True):
        elevation = float(elevations_m[station_index])
        if not np.isfinite(elevation):
            continue
        start = max(0, int(time_index) - trailing_hours + 1)
        temperatures = temperature[start : time_index + 1, station_index]
        temperature_valid = temperature_mask[start : time_index + 1, station_index]
        humidities = humidity[start : time_index + 1, station_index]
        humidity_valid = humidity_mask[start : time_index + 1, station_index]
        if temperature_valid.any():
            temperature_c = float(temperatures[temperature_valid].mean())
            temperature_count[time_index, station_index] = int(temperature_valid.sum())
            relative_humidity = None
            if humidity_valid.any():
                relative_humidity = float(humidities[humidity_valid].mean())
                humidity_count[time_index, station_index] = int(humidity_valid.sum())
                method[time_index, station_index] = METHOD_TRAILING_T_RH
                used_humidity[time_index, station_index] = relative_humidity
            else:
                method[time_index, station_index] = METHOD_TRAILING_T_DRY
            used_temperature[time_index, station_index] = temperature_c
        elif abs(elevation) <= low_elevation_fallback_m:
            temperature_c = 15.0
            relative_humidity = None
            method[time_index, station_index] = METHOD_LOW_ELEVATION_STANDARD_T
            used_temperature[time_index, station_index] = temperature_c
        else:
            continue
        msl = station_pressure_to_msl_hpa(
            float(pressure[time_index, station_index]),
            elevation,
            temperature_c,
            relative_humidity,
        )
        corrected[time_index, station_index] = msl
        correction[time_index, station_index] = msl - pressure[time_index, station_index]
        valid[time_index, station_index] = True
    return {
        "values": corrected,
        "value_mask": valid,
        "raw_station_pressure_hpa": pressure,
        "raw_pressure_mask": pressure_mask,
        "correction_hpa": correction,
        "method_code": method,
        "temperature_sample_count": temperature_count,
        "humidity_sample_count": humidity_count,
        "used_temperature_c": used_temperature,
        "used_relative_humidity_percent": used_humidity,
    }


def error_statistics(source: np.ndarray, reference: np.ndarray) -> dict[str, float | int]:
    difference = np.asarray(source, np.float64) - np.asarray(reference, np.float64)
    return {
        "count": int(difference.size),
        "bias_hpa": float(difference.mean()),
        "mae_hpa": float(np.abs(difference).mean()),
        "rmse_hpa": float(np.sqrt(np.square(difference).mean())),
        "correlation": float(np.corrcoef(source, reference)[0, 1]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="data_external/meteohub/processed/20260801_20260808/official_hourly_physical.npz",
    )
    parser.add_argument(
        "--station-summary",
        default="data/audits/meteohub_official_20260801_20260808/station_summary.csv",
    )
    parser.add_argument(
        "--output",
        default="data_external/meteohub/processed/20260801_20260808/official_hourly_physical_msl.npz",
    )
    parser.add_argument(
        "--manifest", default="data/manifests/dpc_msl_20260801_20260808.json"
    )
    parser.add_argument("--era5t-instant", required=True)
    parser.add_argument("--era5t-accum", required=True)
    args = parser.parse_args()

    input_path = ROOT / args.input
    with np.load(input_path, allow_pickle=False) as source:
        times = np.asarray(source["times"]).astype("datetime64[ns]")
        station_ids = tuple(str(value) for value in source["station_ids"].tolist())
        variable_names = tuple(str(value) for value in source["variable_names"].tolist())
        raw_values = np.asarray(source["values"], np.float32)
        raw_mask = np.asarray(source["value_mask"], bool)
        source_time = np.asarray(source["source_time"])
        age_minutes = np.asarray(source["observation_age_minutes"], np.float32)
        coordinates = np.asarray(source["coordinates"], np.float32)
        station_static = np.asarray(source["station_static"], np.float32)
        source_type = np.asarray(source["source_type"])
    elevations, station_metadata = load_elevations(
        station_ids, ROOT / args.station_summary
    )
    corrected = build_corrected_msl(
        raw_values, raw_mask, variable_names, elevations
    )
    pressure_index = variable_names.index("station_pressure_hpa")
    valid = corrected["value_mask"]
    pressure_source_time = source_time[:, :, pressure_index]
    target_times = np.broadcast_to(times[:, None], valid.shape)
    if np.any(pressure_source_time[valid] > target_times[valid]):
        raise ValueError("Future station pressure reached corrected DPC MSL")
    if not np.isfinite(corrected["values"][valid]).all():
        raise ValueError("Corrected DPC MSL contains non-finite valid values")
    if valid.any() and (
        corrected["values"][valid].min() < 850.0
        or corrected["values"][valid].max() > 1100.0
    ):
        raise ValueError("Corrected DPC MSL is outside 850--1100 hPa")
    observation_age = np.zeros(valid.shape, np.float32)
    observation_age[valid] = age_minutes[:, :, pressure_index][valid] / 60.0
    if (observation_age[valid] < 0).any() or (observation_age[valid] > 1.0).any():
        raise ValueError("Corrected DPC MSL observation age is outside 0--1 hour")

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        times=times,
        station_ids=np.asarray(station_ids),
        variable_names=np.asarray(["msl"]),
        values=corrected["values"][:, :, None],
        value_mask=valid[:, :, None],
        observation_age=observation_age[:, :, None],
        source_time=pressure_source_time[:, :, None],
        coordinates=coordinates,
        station_static=station_static,
        source_type=np.asarray(["physical_dpc_corrected_msl"] * len(station_ids)),
        station_elevation_m=elevations,
        raw_station_pressure_hpa=corrected["raw_station_pressure_hpa"],
        raw_pressure_mask=corrected["raw_pressure_mask"],
        correction_hpa=corrected["correction_hpa"],
        correction_method_code=corrected["method_code"],
        temperature_sample_count=corrected["temperature_sample_count"],
        humidity_sample_count=corrected["humidity_sample_count"],
        used_temperature_c=corrected["used_temperature_c"],
        used_relative_humidity_percent=corrected[
            "used_relative_humidity_percent"
        ],
    )

    era5t = load_era5_target_grid(
        Path(args.era5t_instant), Path(args.era5t_accum), ["msl"]
    )
    indices = era5t.indices_for(times)
    sampled = bilinear_sample_grid(
        era5t.values[indices], era5t.latitudes, era5t.longitudes, coordinates
    )[:, :, 0]
    corrected_stats = error_statistics(corrected["values"][valid], sampled[valid])
    raw_stats = error_statistics(
        corrected["raw_station_pressure_hpa"][valid], sampled[valid]
    )
    methods = {
        METHOD_NAMES[code]: int(
            (
                corrected["raw_pressure_mask"]
                & (corrected["method_code"] == code)
            ).sum()
        )
        for code in METHOD_NAMES
    }
    pressure_stations = np.where(corrected["raw_pressure_mask"].any(axis=0))[0]
    station_rows = []
    for station_index in pressure_stations:
        station_valid = valid[:, station_index]
        item = dict(station_metadata[station_index])
        item.update(
            {
                "valid_corrected_hours": int(station_valid.sum()),
                "mean_correction_hpa": float(
                    corrected["correction_hpa"][:, station_index][station_valid].mean()
                ),
                "corrected_vs_era5t": error_statistics(
                    corrected["values"][:, station_index][station_valid],
                    sampled[:, station_index][station_valid],
                ),
            }
        )
        station_rows.append(item)

    manifest = {
        "schema_version": 1,
        "product": "DPC station pressure reduced to mean sea level",
        "source_variable": "station_pressure_hpa",
        "contract_variable": "msl",
        "unit": "hPa",
        "method": {
            "equation": "p_msl = p_station * exp(g*z/(R_d*T_v_mean_layer))",
            "temperature": "trailing observed temperature at or before valid time, up to 12 hours; mean layer adds half of a 6.5 K/km lapse-rate correction",
            "humidity": "trailing RH is used for virtual temperature when available; otherwise dry-air temperature is explicit",
            "fallback": "15 C standard temperature only where elevation <= 10 m and no temperature exists",
            "scope": "project coastal stations at elevations from 0 to 268 m; not a general high-elevation reduction",
            "method_codes": {str(code): name for code, name in METHOD_NAMES.items()},
        },
        "time_start": str(times[0]),
        "time_end": str(times[-1]),
        "hour_count": len(times),
        "registry_station_count": len(station_ids),
        "pressure_station_count": int(len(pressure_stations)),
        "raw_pressure_valid_cells": int(corrected["raw_pressure_mask"].sum()),
        "corrected_valid_cells": int(valid.sum()),
        "coverage_fraction_all_registry_cells": float(valid.mean()),
        "methods": methods,
        "elevation_range_m": [
            float(elevations[pressure_stations].min()),
            float(elevations[pressure_stations].max()),
        ],
        "correction_hpa": {
            "minimum": float(corrected["correction_hpa"][valid].min()),
            "maximum": float(corrected["correction_hpa"][valid].max()),
            "mean": float(corrected["correction_hpa"][valid].mean()),
        },
        "same_time_same_coordinate_era5t_alignment": {
            "raw_station_pressure": raw_stats,
            "corrected_msl": corrected_stats,
            "interpretation": "source-minus-ERA5T; ERA5T is a gridded diagnostic reference rather than error-free station truth",
        },
        "stations": station_rows,
        "processed_external": {
            "path": args.output,
            "bytes": output.stat().st_size,
            "sha256": sha256(output),
        },
        "provenance": {
            "input": args.input,
            "station_summary": args.station_summary,
            "raw_station_pressure_preserved": True,
            "future_observation_used": False,
        },
    }
    manifest_path = ROOT / args.manifest
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
