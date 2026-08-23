#!/usr/bin/env python3
"""No-truth V9.2 stress test using marine inputs with all DPC stations offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from check_v7 import load_config, resolve  # noqa: E402
from train_v9_output_form import forward, make_model  # noqa: E402
from stormengine_dl.data import NormalizationStats, StaticFields  # noqa: E402
from stormengine_dl.data.operational_adapter import load_fixed_registry  # noqa: E402
from stormengine_dl.event_metrics import PhysicalSixHourEventAccumulator  # noqa: E402
from stormengine_dl.runtime import denormalize_channels  # noqa: E402


TARGETS = ("msl", "u10", "v10", "t2m", "tp")
INPUTS = ("msl", "u10", "v10", "i10fg", "t2m", "tp")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(config: dict, path: Path, device: torch.device) -> torch.nn.Module:
    model = make_model(config, "autoregressive", "field")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval()


def predicted_event_summary(
    prediction: torch.Tensor, sea_mask: torch.Tensor, thresholds: dict[str, float]
) -> dict[str, object]:
    sea = sea_mask.bool().flatten()
    u10 = prediction[:, :, TARGETS.index("u10")].flatten(start_dim=2)[:, :, sea]
    v10 = prediction[:, :, TARGETS.index("v10")].flatten(start_dim=2)[:, :, sea]
    tp = prediction[:, :, TARGETS.index("tp")].flatten(start_dim=2)[:, :, sea]
    maximum_wind = torch.hypot(u10, v10).amax(dim=1)
    six_hour_rain = tp.clamp_min(0).sum(dim=1)
    events = {
        "rain_6h_10mm": six_hour_rain > thresholds["rain_6h_mm"],
        "heavy_rain_6h_30mm": six_hour_rain > thresholds["storm_rain_6h_mm"],
        "extreme_rain_6h_50mm": six_hour_rain > thresholds["extreme_rain_6h_mm"],
        "strong_wind_6h_15ms": maximum_wind > thresholds["strong_wind_ms"],
        "extreme_wind_6h_20ms": maximum_wind > thresholds["extreme_wind_ms"],
    }
    events["storm_any_6h"] = events["heavy_rain_6h_30mm"] | events["strong_wind_6h_15ms"]
    events["compound_storm_6h"] = events["heavy_rain_6h_30mm"] & events["strong_wind_6h_15ms"]
    events["extreme_weather_6h"] = events["extreme_rain_6h_50mm"] | events["extreme_wind_6h_20ms"]
    return {
        "maximum_predicted_sea_wind_ms": float(maximum_wind.max()),
        "maximum_predicted_sea_6h_precipitation_mm": float(six_hour_rain.max()),
        "events": {
            name: {
                "forecast_cases": int(mask.any(dim=1).sum()),
                "sea_grid_cells": int(mask.sum()),
            }
            for name, mask in events.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/v9_2_event_aware_final16.yaml")
    parser.add_argument(
        "--marine-input",
        default="data_external/open_meteo/processed/20260816_20260819/icon2i_hourly_marine.npz",
    )
    parser.add_argument(
        "--marine-msl",
        default="data_external/open_meteo/processed/20260816_20260819/icon2i_hourly_marine_pressure.npz",
    )
    parser.add_argument(
        "--checkpoint", default="artifacts/v9_2_event_aware_final16/seed_42/final.pt"
    )
    parser.add_argument(
        "--source-checkpoint",
        default="artifacts/v9_1_pressure_ablation/pressure_6var/pressure_6var/seed_42/best.pt",
    )
    parser.add_argument(
        "--output-dir", default="artifacts/v9_2_marine_only_stress_20260816_20260819"
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    config = load_config(resolve(args.config))
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    stats = NormalizationStats.load(resolve(config["data"]["normalization_stats"]))
    registry = load_fixed_registry(resolve(config["data"]["station_registry"]), include_virtual=True)
    physical_count = sum(source == "physical_land" for source in registry.source_type)
    if physical_count != 239 or len(registry.station_ids) - physical_count != 151:
        raise ValueError("Expected the frozen 239 physical + 151 marine registry")

    with np.load(resolve(args.marine_input), allow_pickle=False) as marine:
        times = np.asarray(marine["times"]).astype("datetime64[ns]")
        marine_ids = tuple(str(value) for value in marine["station_ids"].tolist())
        marine_names = tuple(str(value) for value in marine["variable_names"].tolist())
        marine_values = np.asarray(marine["values"], np.float32)
        marine_mask = np.asarray(marine["value_mask"], bool)
        marine_age = np.asarray(marine["observation_age"], np.float32)
    with np.load(resolve(args.marine_msl), allow_pickle=False) as pressure:
        pressure_times = np.asarray(pressure["times"]).astype("datetime64[ns]")
        pressure_ids = tuple(str(value) for value in pressure["station_ids"].tolist())
        pressure_names = tuple(str(value) for value in pressure["variable_names"].tolist())
        pressure_values = np.asarray(pressure["values"], np.float32)
        pressure_mask = np.asarray(pressure["value_mask"], bool)
        pressure_age = np.asarray(pressure["observation_age"], np.float32)
    expected_marine = registry.station_ids[physical_count:]
    if marine_ids != expected_marine or pressure_ids != expected_marine:
        raise ValueError("Marine station order differs from the frozen registry")
    if marine_names != INPUTS[1:] or pressure_names != ("msl",):
        raise ValueError("Marine variable order differs from the six-variable V9 contract")
    if not np.array_equal(times, pressure_times):
        raise ValueError("Marine base and pressure time axes differ")

    hours, stations, channels = len(times), len(registry.station_ids), len(INPUTS)
    values = np.zeros((hours, stations, channels), np.float32)
    mask = np.zeros_like(values, bool)
    age = np.zeros_like(values, np.float32)
    raw = np.concatenate((pressure_values, marine_values), axis=-1)
    raw_mask = np.concatenate((pressure_mask, marine_mask), axis=-1)
    raw_age = np.concatenate((pressure_age, marine_age), axis=-1)
    for channel, name in enumerate(INPUTS):
        statistic = stats.variables[name]
        normalized = (raw[:, :, channel] - statistic.mean) / statistic.std
        values[:, physical_count:, channel] = normalized
    mask[:, physical_count:] = raw_mask
    age[:, physical_count:] = raw_age
    values[~mask] = 0

    coordinates = registry.coordinates.copy()
    coordinates[:, 0] = (coordinates[:, 0] - 39.0) / (46.5 - 39.0)
    coordinates[:, 1] = (coordinates[:, 1] - 12.0) / (20.0 - 12.0)
    if not np.isfinite(values).all() or (coordinates < 0).any() or (coordinates > 1).any():
        raise ValueError("Assembled marine-only stress input is invalid")

    history = int(config["data"]["history_hours"])
    forecast = int(config["data"]["forecast_hours"])
    count = hours - history - forecast + 1
    if count <= 0:
        raise ValueError("The input period is too short for one forecast window")
    starts = np.arange(count, dtype=np.int64)
    static_data = StaticFields.load(resolve(config["data"]["static_fields"]))
    static = static_data.as_tensor().unsqueeze(0).to(device)
    sea_mask = torch.from_numpy(static_data.land_sea_mask < 0.5)
    point_coords = torch.from_numpy(coordinates).to(device)
    point_static = torch.from_numpy(registry.station_static).to(device)
    paths = {
        "v9_2_final16": resolve(args.checkpoint),
        "v9_1_frozen": resolve(args.source_checkpoint),
    }
    models = {name: load_model(config, path, device) for name, path in paths.items()}
    predictions: dict[str, list[torch.Tensor]] = {name: [] for name in models}
    started = time.perf_counter()
    with torch.no_grad():
        for offset in range(0, count, args.batch_size):
            batch_starts = starts[offset : offset + args.batch_size]
            size = len(batch_starts)
            batch = {
                "point_values": torch.from_numpy(
                    np.stack([values[start : start + history] for start in batch_starts])
                ).to(device),
                "value_mask": torch.from_numpy(
                    np.stack([mask[start : start + history] for start in batch_starts])
                ).to(device),
                "observation_age": torch.from_numpy(
                    np.stack([age[start : start + history] for start in batch_starts])
                ).to(device),
                "point_coords": point_coords[None].expand(size, -1, -1),
                "point_static": point_static[None].expand(size, -1, -1),
                "target": torch.empty(size, forecast, len(TARGETS), 31, 33, device=device),
            }
            for name, model in models.items():
                normalized = forward(model, batch, static)
                if not torch.isfinite(normalized).all():
                    raise RuntimeError(f"{name} produced non-finite output")
                predictions[name].append(
                    denormalize_channels(normalized, list(TARGETS), stats).cpu()
                )
            print(f"marine-only stress {min(offset + size, count)}/{count}", flush=True)

    merged = {name: torch.cat(blocks) for name, blocks in predictions.items()}
    thresholds = dict(PhysicalSixHourEventAccumulator.DEFAULT_THRESHOLDS)
    thresholds.update(config["training"]["event_aware"].get("thresholds", {}))
    summaries = {
        name: predicted_event_summary(prediction, sea_mask, thresholds)
        for name, prediction in merged.items()
    }
    final, source = merged["v9_2_final16"], merged["v9_1_frozen"]
    difference = final - source
    summaries["v9_2_minus_v9_1"] = {
        variable: {
            "full_grid_rmse_difference": float(
                difference[:, :, channel].square().mean().sqrt()
            ),
            "mean_signed_difference": float(difference[:, :, channel].mean()),
        }
        for channel, variable in enumerate(TARGETS)
    }
    result = {
        "schema_version": 1,
        "scientific_status": "no_truth_marine_only_operational_stress_test",
        "accuracy_claim": False,
        "contract": {
            "period": [str(times[0]), str(times[-1])],
            "forecast_windows": count,
            "history_hours": history,
            "forecast_hours": forecast,
            "physical_dpc_stations": physical_count,
            "physical_dpc_valid_fraction": 0.0,
            "marine_open_meteo_stations": len(expected_marine),
            "marine_open_meteo_valid_fraction": float(raw_mask.mean()),
            "interpretation": "All physical stations are deliberately offline; Open-Meteo is input support, not target truth.",
        },
        "input_stress": {
            "maximum_open_meteo_gust_ms": float(
                raw[:, :, INPUTS.index("i10fg")].max()
            ),
            "maximum_open_meteo_hourly_precipitation_mm": float(
                raw[:, :, INPUTS.index("tp")].max()
            ),
            "hours_with_any_gust_at_least_20ms": int(
                (raw[:, :, INPUTS.index("i10fg")] >= 20.0).any(axis=1).sum()
            ),
            "hours_with_any_precipitation_at_least_5mm": int(
                (raw[:, :, INPUTS.index("tp")] >= 5.0).any(axis=1).sum()
            ),
            "warning": "These model-derived inputs identify stress intensity; they are not verification truth.",
        },
        "checkpoints": {
            name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()
        },
        "predicted_response": summaries,
        "elapsed_seconds": time.perf_counter() - started,
    }
    output = resolve(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    target = output / "stress_summary.json"
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Marine-only stress test complete: {target}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
