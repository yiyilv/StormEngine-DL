#!/usr/bin/env python3
"""Diagnose frozen V9-A with real DPC + Open-Meteo inputs against ERA5T."""

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
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_v9_2024_confirmation as common  # noqa: E402
from check_v7 import load_config, resolve  # noqa: E402
from stormengine_dl.baselines import dense_grid_persistence  # noqa: E402
from stormengine_dl.data import (  # noqa: E402
    NormalizationStats,
    StaticFields,
    load_era5_target_grid,
    load_v7_b_input,
)
from stormengine_dl.data.operational_adapter import load_fixed_registry  # noqa: E402
from stormengine_dl.runtime import denormalize_channels  # noqa: E402
from stormengine_dl.source_alignment import (  # noqa: E402
    bilinear_sample_grid,
    counterfactual_mse_penalty,
)
from stormengine_dl.training import ForecastMetricAccumulator  # noqa: E402
from train_v8_stage3 import reconstruction_forward  # noqa: E402


TARGETS = ("msl", "u10", "v10", "t2m", "tp")


def device_from_name(name: str) -> torch.device:
    selected = "cuda" if name == "auto" and torch.cuda.is_available() else (
        "cpu" if name == "auto" else name
    )
    if selected == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(selected)


def source_coverage(
    mask: np.ndarray,
    age: np.ndarray,
    variable_names: tuple[str, ...],
    physical_count: int,
) -> dict[str, Any]:
    """Summarise actual valid cells and ages for physical and marine sources."""
    result: dict[str, Any] = {}
    slices = {
        "physical_dpc": slice(0, physical_count),
        "marine_open_meteo": slice(physical_count, mask.shape[1]),
    }
    for source, station_slice in slices.items():
        source_mask = mask[:, station_slice]
        source_age = age[:, station_slice]
        variables: dict[str, Any] = {}
        for channel, variable in enumerate(variable_names):
            valid = source_mask[:, :, channel]
            ages = source_age[:, :, channel][valid]
            variables[variable] = {
                "valid_cells": int(valid.sum()),
                "total_cells": int(valid.size),
                "valid_fraction": float(valid.mean()),
                "mean_age_hours": float(ages.mean()) if ages.size else None,
                "p95_age_hours": float(np.percentile(ages, 95)) if ages.size else None,
            }
        result[source] = {
            "station_count": int(source_mask.shape[1]),
            "overall_valid_fraction": float(source_mask.mean()),
            "variables": variables,
        }
    return result


def availability_relationship(records: list[dict[str, float]]) -> dict[str, Any]:
    """Relate physical-input availability to per-window sea RMSE."""
    coverage = np.asarray([row["physical_valid_fraction"] for row in records], np.float64)
    result: dict[str, Any] = {
        "windows": int(len(records)),
        "physical_valid_fraction_min": float(coverage.min()),
        "physical_valid_fraction_mean": float(coverage.mean()),
        "physical_valid_fraction_max": float(coverage.max()),
        "interpretation": "negative correlation means more valid DPC cells coincide with lower error; association is not causation",
        "models": {},
    }
    for model in ("v9_a", "v7_b"):
        errors = np.asarray([row[f"{model}_sea_rmse"] for row in records], np.float64)
        correlation = None
        if coverage.std() > 0 and errors.std() > 0:
            correlation = float(np.corrcoef(coverage, errors)[0, 1])
        order = np.argsort(coverage)
        groups = [group for group in np.array_split(order, min(3, len(order))) if len(group)]
        result["models"][model] = {
            "pearson_correlation": correlation,
            "coverage_tertiles": [
                {
                    "windows": int(len(group)),
                    "coverage_mean": float(coverage[group].mean()),
                    "sea_rmse_mean": float(errors[group].mean()),
                }
                for group in groups
            ],
        }
    return result


def sea_rmse_per_window(
    prediction: torch.Tensor, truth: torch.Tensor, land_sea_mask: torch.Tensor
) -> np.ndarray:
    sea = land_sea_mask.to(prediction.device) < 0.5
    squared = (prediction - truth).square()[..., sea]
    return squared.mean(dim=(1, 2, 3)).sqrt().detach().cpu().numpy()


def normalize_points(
    values: np.ndarray, variable_names: list[str], stats: NormalizationStats
) -> np.ndarray:
    result = np.asarray(values, np.float32).copy()
    for channel, variable in enumerate(variable_names):
        item = stats.variables[variable]
        result[:, :, channel] = (result[:, :, channel] - item.mean) / item.std
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-config", default="configs/v9_2025_final_test.yaml")
    parser.add_argument(
        "--dpc-input",
        default="data_external/meteohub/processed/20260801_20260808/official_hourly_physical.npz",
    )
    parser.add_argument(
        "--marine-input",
        default="data_external/open_meteo/processed/20260801_20260808/icon2i_hourly_marine.npz",
    )
    parser.add_argument("--era5t-instant", required=True)
    parser.add_argument("--era5t-accum", required=True)
    parser.add_argument(
        "--output-dir", default="artifacts/v9_operational_era5t_20260801_20260808"
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-windows", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    protocol = common.read_yaml(resolve(args.protocol_config))
    data_config = load_config(resolve(protocol["data_config"]))
    variables = tuple(data_config["data"]["target_variables"])
    input_variables = list(data_config["data"]["input_variables"])
    if variables != TARGETS:
        raise ValueError(f"Unexpected target contract: {variables}")
    history = int(data_config["data"]["history_hours"])
    forecast = int(data_config["data"]["forecast_hours"])
    selected_device = device_from_name(args.device)

    registry = load_fixed_registry(
        resolve(data_config["data"]["station_registry"]), include_virtual=True
    )
    normalization_path = resolve(protocol["normalization_stats"])
    input_batch = load_v7_b_input(
        resolve(args.dpc_input),
        resolve(args.marine_input),
        normalization_path,
        expected_station_ids=registry.station_ids,
    )
    times = input_batch.times.astype("datetime64[ns]")
    if np.any(np.diff(times) != np.timedelta64(1, "h")):
        raise ValueError("Operational input time axis is not continuous hourly")
    count = len(times) - history - forecast + 1
    if count < 1:
        raise ValueError("Operational input does not contain a complete forecast window")
    if args.max_windows is not None:
        count = min(count, args.max_windows)
    starts = np.arange(count, dtype=np.int64)

    all_variables = list(dict.fromkeys(input_variables + list(variables)))
    target_grid = load_era5_target_grid(
        resolve(args.era5t_instant), resolve(args.era5t_accum), all_variables
    )
    future_times = np.stack(
        [times[start + history : start + history + forecast] for start in starts]
    )
    target_indices = target_grid.indices_for(future_times.reshape(-1)).reshape(
        count, forecast
    )
    analysis_indices = target_grid.indices_for(times[starts + history - 1])
    normalization = NormalizationStats.load(normalization_path)
    time_indices = target_grid.indices_for(times)
    same_week_fields = target_grid.values[time_indices]
    proxy_sampled = bilinear_sample_grid(
        same_week_fields,
        target_grid.latitudes,
        target_grid.longitudes,
        registry.coordinates,
    )
    input_channels = [all_variables.index(name) for name in input_variables]
    target_channels = [all_variables.index(name) for name in variables]
    proxy_point_values = normalize_points(
        proxy_sampled[:, :, input_channels], input_variables, normalization
    )

    static_data = StaticFields.load(resolve(data_config["data"]["static_fields"]))
    if not np.allclose(target_grid.latitudes, static_data.latitudes):
        raise ValueError("ERA5T latitude grid differs from the frozen model grid")
    if not np.allclose(target_grid.longitudes, static_data.longitudes):
        raise ValueError("ERA5T longitude grid differs from the frozen model grid")
    land_mask = torch.from_numpy(static_data.land_sea_mask)
    static = static_data.as_tensor().unsqueeze(0).to(selected_device)
    coordinates = torch.from_numpy(input_batch.coordinates).to(selected_device)
    point_static = torch.from_numpy(input_batch.station_static).to(selected_device)
    models, checkpoint_metadata = common.load_models(
        protocol, data_config, selected_device, len(input_batch.station_ids)
    )
    forecast_names = (
        "v9_a",
        "v7_b",
        "v9_a_era5t_proxy",
        "v7_b_era5t_proxy",
        "v9_reconstruction_persistence",
        "v7_reconstruction_persistence",
        "dense_era5t_persistence",
    )
    metrics = {
        name: ForecastMetricAccumulator(variables, forecast) for name in forecast_names
    }
    reconstruction = {
        name: ForecastMetricAccumulator(variables, 1)
        for name in ("v9_a", "v7_b", "v9_a_era5t_proxy", "v7_b_era5t_proxy")
    }
    window_records: list[dict[str, float]] = []
    started = time.perf_counter()
    with torch.no_grad():
        for offset in range(0, count, args.batch_size):
            batch_starts = starts[offset : offset + args.batch_size]
            size = len(batch_starts)
            values = torch.from_numpy(
                np.stack([input_batch.values[s : s + history] for s in batch_starts])
            ).to(selected_device)
            mask = torch.from_numpy(
                np.stack([input_batch.value_mask[s : s + history] for s in batch_starts])
            ).to(selected_device)
            age = torch.from_numpy(
                np.stack([input_batch.observation_age[s : s + history] for s in batch_starts])
            ).to(selected_device)
            coords = coordinates[None].expand(size, -1, -1)
            points = point_static[None].expand(size, -1, -1)
            static_batch = static.expand(size, -1, -1, -1)

            v9_normalized, v9_current_normalized = models[
                "v9_a"
            ].forward_with_reconstruction(
                values,
                coords,
                mask,
                forecast,
                observation_age=age,
                static_fields=static_batch,
                point_static=points,
            )
            v7_normalized = models["v7_b"](
                values,
                coords,
                mask,
                forecast,
                observation_age=age,
                static_fields=static_batch,
                point_static=points,
            )
            proxy_values = torch.from_numpy(
                np.stack([proxy_point_values[s : s + history] for s in batch_starts])
            ).to(selected_device)
            proxy_mask = torch.ones_like(mask)
            proxy_age = torch.zeros_like(age)
            v9_proxy_normalized, v9_proxy_current_normalized = models[
                "v9_a"
            ].forward_with_reconstruction(
                proxy_values,
                coords,
                proxy_mask,
                forecast,
                observation_age=proxy_age,
                static_fields=static_batch,
                point_static=points,
            )
            v7_proxy_normalized = models["v7_b"](
                proxy_values,
                coords,
                proxy_mask,
                forecast,
                observation_age=proxy_age,
                static_fields=static_batch,
                point_static=points,
            )
            current_batch = {
                "point_values": values[:, -1:],
                "value_mask": mask[:, -1:],
                "observation_age": age[:, -1:],
                "point_coords": coords,
                "point_static": points,
                "target": torch.empty(
                    (
                        size,
                        1,
                        len(variables),
                        static_data.land_sea_mask.shape[0],
                        static_data.land_sea_mask.shape[1],
                    ),
                    device=selected_device,
                ),
            }
            v7_current_normalized = reconstruction_forward(
                models["v7_b"], current_batch, static
            )[:, 0]
            proxy_current_batch = {
                **current_batch,
                "point_values": proxy_values[:, -1:],
                "value_mask": proxy_mask[:, -1:],
                "observation_age": proxy_age[:, -1:],
            }
            v7_proxy_current_normalized = reconstruction_forward(
                models["v7_b"], proxy_current_batch, static
            )[:, 0]
            normalized_outputs = (
                v9_normalized,
                v7_normalized,
                v9_current_normalized,
                v7_current_normalized,
                v9_proxy_normalized,
                v7_proxy_normalized,
                v9_proxy_current_normalized,
                v7_proxy_current_normalized,
            )
            if not all(torch.isfinite(item).all() for item in normalized_outputs):
                raise RuntimeError("Frozen model produced a non-finite operational output")

            v9 = denormalize_channels(v9_normalized, variables, normalization)
            v7 = denormalize_channels(v7_normalized, variables, normalization)
            v9_current = denormalize_channels(
                v9_current_normalized[:, None], variables, normalization
            )[:, 0]
            v7_current = denormalize_channels(
                v7_current_normalized[:, None], variables, normalization
            )[:, 0]
            v9_proxy = denormalize_channels(
                v9_proxy_normalized, variables, normalization
            )
            v7_proxy = denormalize_channels(
                v7_proxy_normalized, variables, normalization
            )
            v9_proxy_current = denormalize_channels(
                v9_proxy_current_normalized[:, None], variables, normalization
            )[:, 0]
            v7_proxy_current = denormalize_channels(
                v7_proxy_current_normalized[:, None], variables, normalization
            )[:, 0]
            truth = torch.from_numpy(
                target_grid.values[target_indices[offset : offset + size]][
                    :, :, target_channels
                ]
            )
            dense_current = torch.from_numpy(
                target_grid.values[analysis_indices[offset : offset + size]][
                    :, target_channels
                ]
            )
            predictions = {
                "v9_a": v9.cpu(),
                "v7_b": v7.cpu(),
                "v9_a_era5t_proxy": v9_proxy.cpu(),
                "v7_b_era5t_proxy": v7_proxy.cpu(),
                "v9_reconstruction_persistence": dense_grid_persistence(
                    v9_current.cpu(), forecast
                ),
                "v7_reconstruction_persistence": dense_grid_persistence(
                    v7_current.cpu(), forecast
                ),
                "dense_era5t_persistence": dense_grid_persistence(dense_current, forecast),
            }
            for name, prediction in predictions.items():
                metrics[name].update(prediction, truth, land_mask)
            reconstruction["v9_a"].update(
                v9_current.cpu()[:, None], dense_current[:, None], land_mask
            )
            reconstruction["v7_b"].update(
                v7_current.cpu()[:, None], dense_current[:, None], land_mask
            )
            reconstruction["v9_a_era5t_proxy"].update(
                v9_proxy_current.cpu()[:, None], dense_current[:, None], land_mask
            )
            reconstruction["v7_b_era5t_proxy"].update(
                v7_proxy_current.cpu()[:, None], dense_current[:, None], land_mask
            )

            v9_window_rmse = sea_rmse_per_window(predictions["v9_a"], truth, land_mask)
            v7_window_rmse = sea_rmse_per_window(predictions["v7_b"], truth, land_mask)
            physical_mask = mask[:, :, : input_batch.physical_station_count]
            physical_fraction = physical_mask.float().mean(dim=(1, 2, 3)).cpu().numpy()
            for index, start in enumerate(batch_starts):
                window_records.append(
                    {
                        "start": int(start),
                        "analysis_time": str(times[start + history - 1].astype("datetime64[m]")),
                        "physical_valid_fraction": float(physical_fraction[index]),
                        "v9_a_sea_rmse": float(v9_window_rmse[index]),
                        "v7_b_sea_rmse": float(v7_window_rmse[index]),
                    }
                )
            print(f"windows {offset + size}/{count}", flush=True)

    computed = {name: accumulator.compute() for name, accumulator in metrics.items()}
    reconstructed = {
        name: accumulator.compute() for name, accumulator in reconstruction.items()
    }
    skills = {
        "v9_a_vs_v7_b": common.skill(computed["v9_a"], computed["v7_b"]),
        "v9_a_era5t_proxy_vs_v7_b_era5t_proxy": common.skill(
            computed["v9_a_era5t_proxy"], computed["v7_b_era5t_proxy"]
        ),
        "v9_a_vs_v9_reconstruction_persistence": common.skill(
            computed["v9_a"], computed["v9_reconstruction_persistence"]
        ),
        "v9_a_vs_v7_reconstruction_persistence": common.skill(
            computed["v9_a"], computed["v7_reconstruction_persistence"]
        ),
        "v9_a_vs_dense_era5t_persistence": common.skill(
            computed["v9_a"], computed["dense_era5t_persistence"]
        ),
    }
    result = {
        "schema_version": 1,
        "scientific_status": "post_freeze_real_input_diagnosis",
        "purpose": "frozen V9-A diagnosis with real DPC plus Open-Meteo inputs and same-period ERA5T targets",
        "limitations": [
            "ERA5T is a gridded reference rather than error-free in-situ truth.",
            "The seven-day period is a compatibility and domain-shift diagnosis, not a new training or model-selection set.",
            "Availability/error correlations are descriptive and do not establish causality.",
            "Dense ERA5T persistence uses stronger information unavailable to the operational model.",
        ],
        "contract": {
            "input_start": str(times[0]),
            "input_end": str(times[-1]),
            "first_analysis": str(times[history - 1]),
            "last_analysis": str(times[count + history - 2]),
            "history_hours": history,
            "forecast_hours": forecast,
            "windows": count,
            "station_count": len(input_batch.station_ids),
            "physical_stations": input_batch.physical_station_count,
            "marine_stations": input_batch.marine_station_count,
            "variables": list(variables),
            "checkpoint_frozen_before_diagnosis": True,
            "no_training": True,
        },
        "checkpoint_metadata": checkpoint_metadata,
        "input_coverage": source_coverage(
            input_batch.value_mask,
            input_batch.observation_age,
            input_batch.variable_names,
            input_batch.physical_station_count,
        ),
        "metrics": computed,
        "current_reconstruction_metrics": reconstructed,
        "rmse_skills": skills,
        "operational_input_penalty": {
            "v9_a": counterfactual_mse_penalty(
                computed["v9_a"], computed["v9_a_era5t_proxy"]
            ),
            "v7_b": counterfactual_mse_penalty(
                computed["v7_b"], computed["v7_b_era5t_proxy"]
            ),
            "definition": "operational-input MSE minus complete ERA5T point-input MSE; positive indicates domain/missingness penalty",
        },
        "availability_relationship": availability_relationship(window_records),
        "window_records": window_records,
        "elapsed_seconds": time.perf_counter() - started,
    }
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "metrics.json"
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(destination),
                "windows": count,
                "device": str(selected_device),
                "all_finite": True,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
