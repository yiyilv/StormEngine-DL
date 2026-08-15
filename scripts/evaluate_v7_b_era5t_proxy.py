#!/usr/bin/env python3
"""Evaluate frozen V7-B with idealized same-week ERA5T point inputs."""

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
from stormengine_dl.data import NormalizationStats, StaticFields, load_era5_target_grid  # noqa: E402
from stormengine_dl.data.operational_adapter import load_fixed_registry  # noqa: E402
from stormengine_dl.models.mask_aware import require_v7_checkpoint_contract  # noqa: E402
from stormengine_dl.runtime import denormalize_channels  # noqa: E402
from stormengine_dl.source_alignment import bilinear_sample_grid, counterfactual_mse_penalty  # noqa: E402
from stormengine_dl.training import ForecastMetricAccumulator  # noqa: E402


def _device(name: str) -> torch.device:
    selected = "cuda" if name == "auto" and torch.cuda.is_available() else ("cpu" if name == "auto" else name)
    if selected == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(selected)


def _load_model(config: dict[str, Any], checkpoint: Path, stations: int, device: torch.device) -> torch.nn.Module:
    model = make_model(config).to(device)
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    require_v7_checkpoint_contract(saved, contract(config, model, stations))
    model.load_state_dict(saved["model_state_dict"]); model.eval(); return model


def _normalise_points(values: np.ndarray, names: list[str], stats: NormalizationStats) -> np.ndarray:
    result = np.asarray(values, np.float32).copy()
    for channel, name in enumerate(names):
        item = stats.variables[name]; result[:, :, channel] = (result[:, :, channel] - item.mean) / item.std
    return result


def _normalise_coordinates(coordinates: np.ndarray, domain: dict[str, Any]) -> np.ndarray:
    result = np.asarray(coordinates, np.float32).copy()
    result[:, 0] = (result[:, 0] - float(domain["lat_min"])) / (float(domain["lat_max"]) - float(domain["lat_min"]))
    result[:, 1] = (result[:, 1] - float(domain["lon_min"])) / (float(domain["lon_max"]) - float(domain["lon_min"]))
    if (result < 0).any() or (result > 1).any(): raise ValueError("Registry coordinates fall outside the V7 domain")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/v7_b.yaml")
    parser.add_argument("--checkpoint", default="artifacts/v7_b_2010_2017/best.pt")
    parser.add_argument("--real-metrics", default="results/v7_operational_era5t_20260801_20260808/metrics.json")
    parser.add_argument("--era5t-instant", required=True); parser.add_argument("--era5t-accum", required=True)
    parser.add_argument("--normalization", default="data/normalization/era5_2010_2015.json")
    parser.add_argument("--output-dir", default="artifacts/v7_b_era5t_proxy_20260801_20260808")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.batch_size < 1: raise ValueError("batch-size must be positive")

    config = load_config(resolve(args.config)); data = config["data"]; domain = config["domain"]
    inputs, targets = list(data["input_variables"]), list(data["target_variables"])
    history, forecast = int(data["history_hours"]), int(data["forecast_hours"])
    real_report = json.loads(resolve(args.real_metrics).read_text(encoding="utf-8"))
    if real_report["history_hours"] != history or real_report["forecast_hours"] != forecast:
        raise ValueError("Published operational metrics use a different forecast contract")
    start = np.datetime64(real_report["time_range"]["input_start"], "ns")
    end = np.datetime64(real_report["time_range"]["input_end"], "ns")
    times = np.arange(start, end + np.timedelta64(1, "h"), np.timedelta64(1, "h")).astype("datetime64[ns]")
    count = len(times) - history - forecast + 1
    if count != int(real_report["windows"]): raise ValueError("Proxy and operational window counts differ")
    starts = np.arange(count, dtype=np.int64)

    names = list(dict.fromkeys(inputs + targets))
    era5t = load_era5_target_grid(resolve(args.era5t_instant), resolve(args.era5t_accum), names)
    time_indices = era5t.indices_for(times); fields = era5t.values[time_indices]
    registry = load_fixed_registry(resolve(data["station_registry"]), include_virtual=True)
    if len(registry.station_ids) != int(data["station_count"]): raise ValueError("V7-B station count differs from registry")
    sampled = bilinear_sample_grid(fields, era5t.latitudes, era5t.longitudes, registry.coordinates)
    input_channels = [names.index(name) for name in inputs]; target_channels = [names.index(name) for name in targets]
    normalization = NormalizationStats.load(resolve(args.normalization))
    point_values = _normalise_points(sampled[:, :, input_channels], inputs, normalization)
    coordinates = _normalise_coordinates(registry.coordinates, domain)
    target_values = fields[:, target_channels]

    static_data = StaticFields.load(resolve(data["static_fields"]))
    if not np.allclose(era5t.latitudes, static_data.latitudes) or not np.allclose(era5t.longitudes, static_data.longitudes):
        raise ValueError("ERA5T and V7-B static grid coordinates differ")
    device = _device(args.device); model = _load_model(config, resolve(args.checkpoint), len(registry.station_ids), device)
    coords = torch.from_numpy(coordinates).to(device); point_static = torch.from_numpy(registry.station_static).to(device)
    static = static_data.as_tensor().unsqueeze(0).to(device); mask = torch.ones((history, len(registry.station_ids), len(inputs)), dtype=torch.bool, device=device)
    age = torch.zeros((history, len(registry.station_ids), len(inputs)), dtype=torch.float32, device=device)
    accumulator = ForecastMetricAccumulator(tuple(targets), forecast); land = torch.from_numpy(static_data.land_sea_mask)
    started = time.perf_counter()
    with torch.no_grad():
        for offset in range(0, count, args.batch_size):
            batch_starts = starts[offset:offset + args.batch_size]; size = len(batch_starts)
            values = torch.from_numpy(np.stack([point_values[s:s + history] for s in batch_starts])).to(device)
            output = model(
                values, coords[None].expand(size, -1, -1), mask[None].expand(size, -1, -1, -1), forecast,
                observation_age=age[None].expand(size, -1, -1, -1), static_fields=static.expand(size, -1, -1, -1),
                point_static=point_static[None].expand(size, -1, -1),
            )
            if not torch.isfinite(output).all(): raise RuntimeError("V7-B produced non-finite proxy-input output")
            prediction = denormalize_channels(output, targets, normalization)
            truth = torch.from_numpy(np.stack([target_values[s + history:s + history + forecast] for s in batch_starts]))
            accumulator.update(prediction, truth, land); print(f"windows {offset + size}/{count}", flush=True)

    proxy_metrics = accumulator.compute(); operational_metrics = real_report["metrics"]["v7_b"]
    comparison = counterfactual_mse_penalty(operational_metrics, proxy_metrics)
    result = {
        "schema_version": 1,
        "purpose": "V7-B same-week ideal ERA5T point-input counterfactual",
        "interpretation": {
            "operational": "published V7-B metrics from real DPC plus Open-Meteo inputs",
            "proxy": "the same frozen V7-B with complete ERA5T values sampled at all 390 input coordinates; mask=true and age=0",
            "controlled_factors": "same model, checkpoint, coordinates, 12-hour histories, 6-hour targets, 152 windows, target grids, and metrics",
            "excess_mse": "operational MSE minus ERA5T-proxy MSE; positive indicates an operational-input penalty",
            "scope": "combined deployment input-system effect, including source mismatch, missingness, age, and point/grid representativeness; not an additive causal decomposition",
        },
        "time_start": str(times[0]), "time_end": str(times[-1]), "windows": count,
        "history_hours": history, "forecast_hours": forecast, "station_count": len(registry.station_ids),
        "variables": targets, "device": str(device), "elapsed_seconds": time.perf_counter() - started,
        "metrics": {"operational_real_input": operational_metrics, "era5t_proxy_input": proxy_metrics},
        "counterfactual_mse_penalty": comparison,
    }
    output = resolve(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    destination = output / "metrics.json"; destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary = {
        region: {
            name: {key: comparison["aggregate"][region][name][key] for key in ("operational_rmse", "era5t_proxy_rmse", "excess_mse", "excess_fraction_of_operational_mse")}
            for name in targets
        } for region in ("full", "land", "sea")
    }
    print(json.dumps({"output": str(destination), "summary": summary}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
