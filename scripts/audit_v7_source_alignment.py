#!/usr/bin/env python3
"""Audit DPC and Open-Meteo values against ERA5T at identical times and points."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))

from check_v7 import resolve  # noqa: E402
from stormengine_dl.data import load_era5_target_grid  # noqa: E402
from stormengine_dl.source_alignment import bilinear_sample_grid, paired_source_statistics  # noqa: E402

VARIABLES = ("u10", "v10", "i10fg", "t2m", "tp")
DPC_NAMES = {"u10": "u10", "v10": "v10", "i10fg": "wind_gust_max", "t2m": "t2m", "tp": "tp"}


def _select(values: np.ndarray, names: tuple[str, ...], requested: tuple[str, ...]) -> np.ndarray:
    missing = set(requested) - set(names)
    if missing: raise ValueError(f"Tensor is missing variables: {sorted(missing)}")
    return np.stack([values[:, :, names.index(name)] for name in requested], axis=-1)


def _load_dpc(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        names = tuple(str(value) for value in source["variable_names"].tolist())
        requested = tuple(DPC_NAMES[name] for name in VARIABLES)
        return {
            "times": np.asarray(source["times"]).astype("datetime64[ns]"),
            "values": _select(np.asarray(source["values"], np.float32), names, requested),
            "mask": _select(np.asarray(source["value_mask"], bool), names, requested),
            "age_minutes": _select(np.asarray(source["observation_age_minutes"], np.float32), names, requested),
            "coordinates": np.asarray(source["coordinates"], np.float32),
        }


def _load_marine(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        names = tuple(str(value) for value in source["variable_names"].tolist())
        coordinates = np.asarray(source["coordinates"], np.float32)
        returned = np.asarray(source["returned_coordinates"], np.float32) if "returned_coordinates" in source else coordinates
        return {
            "times": np.asarray(source["times"]).astype("datetime64[ns]"),
            "values": _select(np.asarray(source["values"], np.float32), names, VARIABLES),
            "mask": _select(np.asarray(source["value_mask"], bool), names, VARIABLES),
            "coordinates": coordinates,
            "returned_coordinates": returned,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dpc-input", default="data_external/meteohub/processed/20260801_20260808/official_hourly_physical.npz")
    parser.add_argument("--marine-input", default="data_external/open_meteo/processed/20260801_20260808/icon2i_hourly_marine.npz")
    parser.add_argument("--era5t-instant", required=True)
    parser.add_argument("--era5t-accum", required=True)
    parser.add_argument("--fresh-minutes", type=float, default=5.0)
    parser.add_argument("--output-dir", default="artifacts/v7_source_alignment_20260801_20260808")
    args = parser.parse_args()
    if args.fresh_minutes < 0 or args.fresh_minutes > 60: raise ValueError("fresh-minutes must be within 0-60")

    dpc = _load_dpc(resolve(args.dpc_input)); marine = _load_marine(resolve(args.marine_input))
    if not np.array_equal(dpc["times"], marine["times"]):
        raise ValueError("DPC and Open-Meteo time axes are not identical")
    if np.any(np.diff(dpc["times"]) != np.timedelta64(1, "h")):
        raise ValueError("Operational source time axis is not continuous hourly")
    era5t = load_era5_target_grid(resolve(args.era5t_instant), resolve(args.era5t_accum), VARIABLES)
    indices = era5t.indices_for(dpc["times"])
    era_values = era5t.values[indices]
    dpc_reference = bilinear_sample_grid(era_values, era5t.latitudes, era5t.longitudes, dpc["coordinates"])
    marine_reference = bilinear_sample_grid(era_values, era5t.latitudes, era5t.longitudes, marine["coordinates"])
    marine_returned_reference = bilinear_sample_grid(
        era_values, era5t.latitudes, era5t.longitudes, marine["returned_coordinates"]
    )
    dpc_all = paired_source_statistics(dpc["values"], dpc_reference, dpc["mask"], VARIABLES)
    fresh_mask = dpc["mask"] & (dpc["age_minutes"] <= args.fresh_minutes)
    dpc_fresh = paired_source_statistics(dpc["values"], dpc_reference, fresh_mask, VARIABLES)
    marine_all = paired_source_statistics(marine["values"], marine_reference, marine["mask"], VARIABLES)
    marine_returned = paired_source_statistics(
        marine["values"], marine_returned_reference, marine["mask"], VARIABLES
    )
    result = {
        "schema_version": 1,
        "purpose": "same-time same-coordinate source alignment audit; not a forecast evaluation",
        "time_start": str(dpc["times"][0]), "time_end": str(dpc["times"][-1]), "hours": len(dpc["times"]),
        "variables": list(VARIABLES),
        "station_counts": {"dpc_physical": int(dpc["values"].shape[1]), "open_meteo_marine": int(marine["values"].shape[1])},
        "comparison": {
            "dpc": "official point observations minus bilinearly sampled ERA5T at the same valid time and coordinate",
            "open_meteo": "ICON-2I model-derived values minus bilinearly sampled ERA5T at the same valid time and coordinate",
            "open_meteo_model_coordinates": "ERA5T sampled at the requested coordinates supplied to V7-B",
            "open_meteo_returned_coordinates": "ERA5T sampled at the provider-returned model-grid coordinates; comparison separates cell-selection displacement from value bias",
            "bias_sign": "positive means the operational source is higher than ERA5T",
            "fresh_dpc": f"valid DPC cells with observation age <= {args.fresh_minutes:g} minutes",
        },
        "exclusions": {"msl": "DPC station pressure is not directly comparable with ERA5T mean-sea-level pressure"},
        "limitations": [
            "DPC and Open-Meteo occupy different land/coastal and marine coordinates, so their RMSE values are not a controlled head-to-head ranking.",
            "ERA5T is a gridded reanalysis reference, not error-free in-situ truth.",
            "Open-Meteo model run time is unavailable in the stitched historical product; this audit uses valid times only.",
            "Wind gust and precipitation may retain source-specific reporting-window semantics despite hourly alignment.",
        ],
        "statistics": {
            "dpc_all_valid": dpc_all,
            "dpc_fresh": dpc_fresh,
            "open_meteo_at_model_coordinates": marine_all,
            "open_meteo_at_returned_coordinates": marine_returned,
        },
    }
    output = resolve(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    destination = output / "source_alignment.json"
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary = {
        group: {name: {key: values[name][key] for key in ("count", "bias", "rmse", "correlation")} for name in VARIABLES}
        for group, values in result["statistics"].items()
    }
    print(json.dumps({"output": str(destination), "summary": summary}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
