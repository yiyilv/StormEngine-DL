#!/usr/bin/env python3
"""Independent 2026 ERA5T evaluation of V9.2 Final16 with real inputs."""

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
from stormengine_dl.baselines import dense_grid_persistence  # noqa: E402
from stormengine_dl.data import (  # noqa: E402
    NormalizationStats,
    StaticFields,
    load_era5_target_grid,
    load_v7_b_input,
)
from stormengine_dl.data.operational_adapter import load_fixed_registry  # noqa: E402
from stormengine_dl.event_metrics import PhysicalSixHourEventAccumulator  # noqa: E402
from stormengine_dl.runtime import denormalize_channels  # noqa: E402
from stormengine_dl.training import ForecastMetricAccumulator  # noqa: E402


TARGETS = ("msl", "u10", "v10", "t2m", "tp")
INPUTS = ("msl", "u10", "v10", "i10fg", "t2m", "tp")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pressure(
    physical_path: Path,
    marine_path: Path,
    *,
    times: np.ndarray,
    station_ids: tuple[str, ...],
    physical_count: int,
    stats: NormalizationStats,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    blocks = []
    metadata: dict[str, object] = {}
    for name, path, expected_ids in (
        ("physical_dpc", physical_path, station_ids[:physical_count]),
        ("marine_open_meteo", marine_path, station_ids[physical_count:]),
    ):
        with np.load(path, allow_pickle=False) as source:
            source_times = np.asarray(source["times"]).astype("datetime64[ns]")
            source_ids = tuple(str(value) for value in source["station_ids"].tolist())
            variables = tuple(str(value) for value in source["variable_names"].tolist())
            if not np.array_equal(source_times, times) or source_ids != expected_ids:
                raise ValueError(f"{name} pressure axes do not match the 390-point input")
            if variables != ("msl",):
                raise ValueError(f"{name} pressure variable must be exactly msl")
            values = np.asarray(source["values"], np.float32)
            mask = np.asarray(source["value_mask"], bool)
            age = np.asarray(source["observation_age"], np.float32)
        if values.shape != mask.shape or values.shape != age.shape:
            raise ValueError(f"{name} pressure tensor shapes differ")
        if not np.isfinite(values[mask]).all() or not np.isfinite(age[mask]).all():
            raise ValueError(f"{name} valid pressure contains non-finite values")
        if ((age[mask] < 0) | (age[mask] > 1)).any():
            raise ValueError(f"{name} pressure age must be normalized to 0--1 hour")
        blocks.append((values, mask, age))
        metadata[name] = {
            "path": str(path),
            "stations": len(expected_ids),
            "valid_fraction": float(mask.mean()),
            "valid_cells": int(mask.sum()),
        }
    values = np.concatenate([block[0] for block in blocks], axis=1)
    mask = np.concatenate([block[1] for block in blocks], axis=1)
    age = np.concatenate([block[2] for block in blocks], axis=1)
    stat = stats.variables["msl"]
    values = (values - stat.mean) / stat.std
    values[~mask] = 0.0
    return values, mask, age, metadata


def load_model(config: dict, checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    model = make_model(config, "autoregressive", "field")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval(), checkpoint.get("model_contract", {})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v9_2_event_aware_final16.yaml")
    parser.add_argument(
        "--dpc-input",
        default="data_external/meteohub/processed/20260801_20260808/official_hourly_physical.npz",
    )
    parser.add_argument(
        "--dpc-msl",
        default="data_external/meteohub/processed/20260801_20260808/official_hourly_physical_msl.npz",
    )
    parser.add_argument(
        "--marine-input",
        default="data_external/open_meteo/processed/20260801_20260808/icon2i_hourly_marine.npz",
    )
    parser.add_argument(
        "--marine-msl",
        default="data_external/open_meteo/processed/20260801_20260808/icon2i_hourly_marine_pressure.npz",
    )
    parser.add_argument("--era5t-instant", required=True)
    parser.add_argument("--era5t-accum", required=True)
    parser.add_argument(
        "--checkpoint", default="artifacts/v9_2_event_aware_final16/seed_42/final.pt"
    )
    parser.add_argument(
        "--source-checkpoint",
        default="artifacts/v9_1_pressure_ablation/pressure_6var/pressure_6var/seed_42/best.pt",
    )
    parser.add_argument(
        "--output-dir", default="artifacts/v9_2_final16_operational_era5t_20260801_20260808"
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-windows", type=int)
    args = parser.parse_args()

    config = load_config(resolve(args.config))
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    normalization = NormalizationStats.load(resolve(config["data"]["normalization_stats"]))
    registry = load_fixed_registry(resolve(config["data"]["station_registry"]), include_virtual=True)
    base = load_v7_b_input(
        resolve(args.dpc_input),
        resolve(args.marine_input),
        resolve(config["data"]["normalization_stats"]),
        expected_station_ids=registry.station_ids,
    )
    pressure_values, pressure_mask, pressure_age, pressure_metadata = load_pressure(
        resolve(args.dpc_msl),
        resolve(args.marine_msl),
        times=base.times.astype("datetime64[ns]"),
        station_ids=base.station_ids,
        physical_count=base.physical_station_count,
        stats=normalization,
    )
    values = np.concatenate((pressure_values, base.values), axis=-1)
    mask = np.concatenate((pressure_mask, base.value_mask), axis=-1)
    age = np.concatenate((pressure_age, base.observation_age), axis=-1)
    if values.shape[-1] != len(INPUTS) or not np.isfinite(values).all():
        raise ValueError("assembled six-variable operational input is invalid")

    times = base.times.astype("datetime64[ns]")
    history = int(config["data"]["history_hours"])
    forecast_hours = int(config["data"]["forecast_hours"])
    count = len(times) - history - forecast_hours + 1
    if args.max_windows is not None:
        count = min(count, args.max_windows)
    starts = np.arange(count, dtype=np.int64)
    target_grid = load_era5_target_grid(
        resolve(args.era5t_instant), resolve(args.era5t_accum), list(TARGETS)
    )
    future_times = np.stack(
        [times[start + history : start + history + forecast_hours] for start in starts]
    )
    target_indices = target_grid.indices_for(future_times.reshape(-1)).reshape(count, forecast_hours)
    analysis_indices = target_grid.indices_for(times[starts + history - 1])

    static_data = StaticFields.load(resolve(config["data"]["static_fields"]))
    if not np.allclose(target_grid.latitudes, static_data.latitudes) or not np.allclose(
        target_grid.longitudes, static_data.longitudes
    ):
        raise ValueError("ERA5T and frozen V9.2 grids differ")
    static = static_data.as_tensor().unsqueeze(0).to(device)
    land_mask = torch.from_numpy(static_data.land_sea_mask)
    sea_mask = land_mask < 0.5
    coordinates = torch.from_numpy(base.coordinates).to(device)
    point_static = torch.from_numpy(base.station_static).to(device)

    final_path, source_path = resolve(args.checkpoint), resolve(args.source_checkpoint)
    final_model, final_contract = load_model(config, final_path, device)
    source_model, source_contract = load_model(config, source_path, device)
    names = ("v9_2_final16", "v9_1_frozen", "dense_era5t_persistence")
    field_metrics = {
        name: ForecastMetricAccumulator(TARGETS, forecast_hours) for name in names
    }
    thresholds = config["training"]["event_aware"]["thresholds"]
    event_metrics = {
        name: PhysicalSixHourEventAccumulator(
            forecast_hours, thresholds=thresholds, region_name="sea"
        )
        for name in names
    }
    started = time.perf_counter()
    processed = 0
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
                "point_coords": coordinates[None].expand(size, -1, -1),
                "point_static": point_static[None].expand(size, -1, -1),
                "target": torch.empty(size, forecast_hours, len(TARGETS), 31, 33, device=device),
            }
            final_normalized = forward(final_model, batch, static)
            source_normalized = forward(source_model, batch, static)
            if not torch.isfinite(final_normalized).all() or not torch.isfinite(source_normalized).all():
                raise RuntimeError("operational model produced non-finite output")
            truth = torch.from_numpy(target_grid.values[target_indices[offset : offset + size]])
            current = torch.from_numpy(target_grid.values[analysis_indices[offset : offset + size]])
            predictions = {
                "v9_2_final16": denormalize_channels(final_normalized, list(TARGETS), normalization),
                "v9_1_frozen": denormalize_channels(source_normalized, list(TARGETS), normalization),
                "dense_era5t_persistence": dense_grid_persistence(current, forecast_hours),
            }
            for name, prediction in predictions.items():
                field_metrics[name].update(prediction, truth, land_mask)
                event_metrics[name].update(prediction, truth, sea_mask, TARGETS)
            processed += size
            print(f"2026 operational evaluation {processed}/{count}", flush=True)

    bounded = args.max_windows is not None and args.max_windows < len(times) - history - forecast_hours + 1
    coverage = {
        "overall_valid_fraction": float(mask.mean()),
        "physical_valid_fraction": float(mask[:, : base.physical_station_count].mean()),
        "marine_valid_fraction": float(mask[:, base.physical_station_count :].mean()),
        "by_variable": {
            variable: {
                "valid_fraction": float(mask[:, :, channel].mean()),
                "valid_cells": int(mask[:, :, channel].sum()),
            }
            for channel, variable in enumerate(INPUTS)
        },
        "pressure_sources": pressure_metadata,
    }
    result = {
        "schema_version": 1,
        "scientific_status": (
            "bounded_2026_operational_pipeline_check"
            if bounded else "independent_2026_era5t_operational_evaluation"
        ),
        "contract": {
            "input_period": [str(times[0]), str(times[-1])],
            "forecast_windows": count,
            "history_hours": history,
            "forecast_hours": forecast_hours,
            "input_variables": list(INPUTS),
            "target_variables": list(TARGETS),
            "target_reference": "ERA5T reanalysis, not direct station truth",
            "years_in_final_training": list(range(2010, 2026)),
            "2026_excluded_from_training": True,
        },
        "checkpoints": {
            "v9_2_final16": {"path": str(final_path), "sha256": sha256(final_path), "contract": final_contract},
            "v9_1_frozen": {"path": str(source_path), "sha256": sha256(source_path), "contract": source_contract},
        },
        "input_coverage": coverage,
        "field_metrics": {name: accumulator.compute() for name, accumulator in field_metrics.items()},
        "physical_six_hour_event_metrics": {
            name: accumulator.compute() for name, accumulator in event_metrics.items()
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    output = resolve(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    target = output / "metrics.json"
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"V9.2 Final16 2026 evaluation complete: {target}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
