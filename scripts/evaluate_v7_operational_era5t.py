#!/usr/bin/env python3
"""Compare frozen V7-A and V7-B real-input forecasts against the same ERA5T grid."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))

from check_v7 import contract, load_config, make_model, resolve  # noqa: E402
from stormengine_dl.baselines import dense_grid_persistence  # noqa: E402
from stormengine_dl.data import (  # noqa: E402
    NormalizationStats,
    StaticFields,
    load_dpc_v7_input,
    load_era5_target_grid,
    load_v7_b_input,
)
from stormengine_dl.data.operational_adapter import load_fixed_registry  # noqa: E402
from stormengine_dl.models.mask_aware import require_v7_checkpoint_contract  # noqa: E402
from stormengine_dl.runtime import denormalize_channels  # noqa: E402
from stormengine_dl.training import ForecastMetricAccumulator  # noqa: E402


def _device(name: str) -> torch.device:
    selected = "cuda" if name == "auto" and torch.cuda.is_available() else ("cpu" if name == "auto" else name)
    if selected == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(selected)


def _load_frozen_model(config: dict[str, Any], checkpoint: Path, stations: int, device: torch.device) -> torch.nn.Module:
    model = make_model(config).to(device)
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    require_v7_checkpoint_contract(saved, contract(config, model, stations))
    model.load_state_dict(saved["model_state_dict"]); model.eval()
    return model


def _assert_compatible(a: dict[str, Any], b: dict[str, Any]) -> None:
    for section, key in (
        ("data", "input_variables"), ("data", "target_variables"),
        ("data", "history_hours"), ("data", "forecast_hours"),
        ("domain", "height"), ("domain", "width"),
    ):
        if a[section][key] != b[section][key]:
            raise ValueError(f"V7-A and V7-B differ at {section}.{key}")


def _improvement(candidate: dict[str, object], reference: dict[str, object]) -> dict[str, object]:
    """Return RMSE skill = 1 - candidate/reference for every comparable cell."""
    output: dict[str, object] = {"aggregate": {}, "by_lead_hour": {}}
    sections = [("aggregate", candidate["aggregate"], reference["aggregate"])]
    for lead, values in candidate["by_lead_hour"].items():
        sections.append((str(lead), values, reference["by_lead_hour"][lead]))
    for section, left, right in sections:
        result: dict[str, object] = {}
        for region in ("full", "land", "sea"):
            result[region] = {}
            for variable, metrics in left[region].items():
                candidate_rmse = float(metrics["rmse"])
                reference_rmse = float(right[region][variable]["rmse"])
                result[region][variable] = None if reference_rmse == 0 else 1.0 - candidate_rmse / reference_rmse
        if section == "aggregate": output["aggregate"] = result
        else: output["by_lead_hour"][section] = result
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v7-a-config", default="configs/v7_a.yaml")
    parser.add_argument("--v7-b-config", default="configs/v7_b.yaml")
    parser.add_argument("--v7-a-checkpoint", default="artifacts/v7_a_2010_2017/best.pt")
    parser.add_argument("--v7-b-checkpoint", default="artifacts/v7_b_2010_2017/best.pt")
    parser.add_argument("--dpc-input", default="data_external/meteohub/processed/20260801_20260808/official_hourly_physical.npz")
    parser.add_argument("--marine-input", default="data_external/open_meteo/processed/20260801_20260808/icon2i_hourly_marine.npz")
    parser.add_argument("--era5t-instant", required=True)
    parser.add_argument("--era5t-accum", required=True)
    parser.add_argument("--normalization", default="data/normalization/era5_2010_2015.json")
    parser.add_argument("--output-dir", default="artifacts/v7_operational_era5t_20260801_20260808")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-windows", type=int)
    args = parser.parse_args()
    if args.batch_size < 1: raise ValueError("batch-size must be positive")

    config_a = load_config(resolve(args.v7_a_config)); config_b = load_config(resolve(args.v7_b_config))
    _assert_compatible(config_a, config_b)
    data = config_a["data"]; targets = list(data["target_variables"])
    history, forecast = int(data["history_hours"]), int(data["forecast_hours"])
    device = _device(args.device)

    registry_a = load_fixed_registry(resolve(data["station_registry"]), include_virtual=False)
    registry_b = load_fixed_registry(resolve(config_b["data"]["station_registry"]), include_virtual=True)
    normalization_path = resolve(args.normalization)
    input_a = load_dpc_v7_input(resolve(args.dpc_input), normalization_path, expected_station_ids=registry_a.station_ids)
    input_b = load_v7_b_input(resolve(args.dpc_input), resolve(args.marine_input), normalization_path, expected_station_ids=registry_b.station_ids)
    times = input_a.times.astype("datetime64[ns]")
    if not np.array_equal(times, input_b.times.astype("datetime64[ns]")):
        raise ValueError("V7-A and V7-B operational inputs do not share the same time axis")
    if np.any(np.diff(times) != np.timedelta64(1, "h")):
        raise ValueError("Operational input time axis is not continuous hourly")
    count = len(times) - history - forecast + 1
    if count < 1: raise ValueError("Operational input does not contain one complete forecast window")
    if args.max_windows is not None: count = min(count, args.max_windows)
    starts = np.arange(count, dtype=np.int64)

    target_grid = load_era5_target_grid(resolve(args.era5t_instant), resolve(args.era5t_accum), targets)
    future_times = np.stack([times[start + history:start + history + forecast] for start in starts])
    target_indices = target_grid.indices_for(future_times.reshape(-1)).reshape(count, forecast)
    analysis_indices = target_grid.indices_for(times[starts + history - 1])

    static_a_data = StaticFields.load(resolve(data["static_fields"])); static_b_data = StaticFields.load(resolve(config_b["data"]["static_fields"]))
    for name, actual, expected in (
        ("latitudes", target_grid.latitudes, static_a_data.latitudes),
        ("longitudes", target_grid.longitudes, static_a_data.longitudes),
    ):
        if not np.allclose(actual, expected): raise ValueError(f"ERA5T and model static {name} differ")
    if not np.array_equal(static_a_data.land_sea_mask, static_b_data.land_sea_mask):
        raise ValueError("V7-A and V7-B land-sea masks differ")

    model_a = _load_frozen_model(config_a, resolve(args.v7_a_checkpoint), len(input_a.station_ids), device)
    model_b = _load_frozen_model(config_b, resolve(args.v7_b_checkpoint), len(input_b.station_ids), device)
    static_a = static_a_data.as_tensor().unsqueeze(0).to(device); static_b = static_b_data.as_tensor().unsqueeze(0).to(device)
    coords_a = torch.from_numpy(input_a.coordinates).to(device); coords_b = torch.from_numpy(input_b.coordinates).to(device)
    point_static_a = torch.from_numpy(input_a.station_static).to(device); point_static_b = torch.from_numpy(input_b.station_static).to(device)
    normalization = NormalizationStats.load(normalization_path)
    metrics = {name: ForecastMetricAccumulator(tuple(targets), forecast) for name in ("v7_a", "v7_b", "dense_era5t_persistence")}
    land = torch.from_numpy(static_a_data.land_sea_mask)
    started = time.perf_counter()
    with torch.no_grad():
        for offset in range(0, count, args.batch_size):
            batch_starts = starts[offset:offset + args.batch_size]; size = len(batch_starts)
            predictions: dict[str, torch.Tensor] = {}
            for name, batch, model, coords, point_static, static in (
                ("v7_a", input_a, model_a, coords_a, point_static_a, static_a),
                ("v7_b", input_b, model_b, coords_b, point_static_b, static_b),
            ):
                values = torch.from_numpy(np.stack([batch.values[s:s + history] for s in batch_starts])).to(device)
                mask = torch.from_numpy(np.stack([batch.value_mask[s:s + history] for s in batch_starts])).to(device)
                age = torch.from_numpy(np.stack([batch.observation_age[s:s + history] for s in batch_starts])).to(device)
                output = model(
                    values, coords[None].expand(size, -1, -1), mask, forecast,
                    observation_age=age, static_fields=static.expand(size, -1, -1, -1),
                    point_static=point_static[None].expand(size, -1, -1),
                )
                if not torch.isfinite(output).all(): raise RuntimeError(f"{name} produced non-finite output")
                predictions[name] = denormalize_channels(output, targets, normalization)
            truth = torch.from_numpy(target_grid.values[target_indices[offset:offset + size]])
            current = torch.from_numpy(target_grid.values[analysis_indices[offset:offset + size]])
            persistence = dense_grid_persistence(current, forecast)
            metrics["v7_a"].update(predictions["v7_a"], truth, land)
            metrics["v7_b"].update(predictions["v7_b"], truth, land)
            metrics["dense_era5t_persistence"].update(persistence, truth, land)
            print(f"windows {offset + size}/{count}", flush=True)

    computed = {name: accumulator.compute() for name, accumulator in metrics.items()}
    result = {
        "schema_version": 1,
        "purpose": "same-window operational-input evaluation against ERA5T grid truth",
        "interpretation": {
            "v7_a": "239 physical DPC/MeteoHub stations",
            "v7_b": "the same 239 physical stations plus 151 model-derived Open-Meteo marine points",
            "target": "independent same-period ERA5T gridded analysis in physical units",
            "comparison_scope": "model-system ablation; it measures the value of the complete marine-support pathway, not Open-Meteo as in-situ truth",
            "dense_era5t_persistence": "last complete ERA5T grid repeated for all leads; stronger-information diagnostic baseline",
        },
        "time_range": {"input_start": str(times[0]), "input_end": str(times[-1]), "first_analysis": str(times[history - 1]), "last_analysis": str(times[count + history - 2])},
        "history_hours": history, "forecast_hours": forecast, "windows": count,
        "station_counts": {"v7_a_physical": len(input_a.station_ids), "v7_b_total": len(input_b.station_ids), "v7_b_marine": input_b.marine_station_count},
        "variables": targets, "elapsed_seconds": time.perf_counter() - started,
        "metrics": computed,
        "rmse_skill": {
            "v7_b_over_v7_a": _improvement(computed["v7_b"], computed["v7_a"]),
            "v7_a_over_dense_persistence": _improvement(computed["v7_a"], computed["dense_era5t_persistence"]),
            "v7_b_over_dense_persistence": _improvement(computed["v7_b"], computed["dense_era5t_persistence"]),
            "definition": "1 - candidate_RMSE / reference_RMSE; positive means the candidate is better",
        },
    }
    output = resolve(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    destination = output / "metrics.json"; destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(destination), "windows": count, "device": str(device)}, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
