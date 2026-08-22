#!/usr/bin/env python3
"""Add frozen event verification to the already completed V9-A 2025 test."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_v9_2024_confirmation as v9_common  # noqa: E402
import evaluate_v9_2025_final_test as frozen  # noqa: E402
from check_v7 import load_config, make_dataset, move, resolve  # noqa: E402
from evaluate_v8_2016_benchmark import (  # noqa: E402
    derive_event_thresholds,
    make_event_accumulators,
    update_events,
)
from stormengine_dl.baselines import dense_grid_persistence  # noqa: E402
from stormengine_dl.data import NormalizationStats, StaticFields  # noqa: E402
from stormengine_dl.event_metrics import PhysicalSixHourEventAccumulator  # noqa: E402
from stormengine_dl.runtime import denormalize_channels  # noqa: E402
from stormengine_dl.training import ForecastMetricAccumulator  # noqa: E402


def assert_replay_matches(
    replay: dict[str, Any], original: dict[str, Any], *, tolerance: float = 1e-6
) -> None:
    for model, values in replay.items():
        expected = original[model]
        for region in ("full", "land", "sea"):
            for variable in v9_common.TARGETS:
                for metric in ("mae", "rmse"):
                    actual_value = float(values["aggregate"][region][variable][metric])
                    expected_value = float(expected["aggregate"][region][variable][metric])
                    if abs(actual_value - expected_value) > tolerance:
                        raise ValueError(
                            f"Frozen 2025 replay mismatch for {model}/{region}/{variable}/{metric}: "
                            f"{actual_value} vs {expected_value}"
                        )


def publish(source: Path, destination: Path, result: dict[str, Any]) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / "events.json"
    shutil.copyfile(source, output)
    physical = result.get("physical_six_hour_event_metrics", {})
    percentile = result.get("event_metrics", {})
    title = (
        "# Frozen 2025 original physical-event extension"
        if physical and not percentile
        else "# V9-A frozen 2025 event-metric extension"
    )
    lines = [
        title,
        "",
        "This is a post-freeze metric extension of the existing one-time 2025 test. "
        "It reuses the unchanged checkpoints, windows, prediction pipeline, and baselines. "
        "It does not permit model or threshold changes.",
    ]
    model_names = (
        "v7_b",
        "v9_a",
        "sparse_reconstruction_persistence",
        "dense_era5_persistence",
    )
    if percentile:
        lines.extend(
            [
                "",
                "## Training-distribution event thresholds",
                "",
                "Thresholds are derived only from 2010--2015.",
                "",
                "| model | event | POD | FAR | CSI | event RMSE | peak bias |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for model in model_names:
            for event, value in percentile[model].items():
                aggregate = value["aggregate"]

                def fmt(name: str) -> str:
                    item = aggregate[name]
                    return "NA" if item is None else f"{float(item):.4f}"

                lines.append(
                    f"| {model} | {event} | {fmt('pod')} | {fmt('far')} | {fmt('csi')} "
                    f"| {fmt('event_conditioned_rmse')} | {fmt('peak_intensity_bias')} |"
                )
    if physical:
        lines.extend(
            [
                "",
                "## Original six-hour physical event definitions",
                "",
                "Hourly precipitation is clipped at zero and summed over +1--+6 h. "
                "Wind speed is derived from u10/v10 and maximized over the same window. "
                "Both grid-cell localization and whole-forecast-case detection are reported.",
                "",
                "| model | event | grid POD | grid FAR | grid CSI | case POD | case FAR | case CSI | tp6h RMSE | wind-max RMSE |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for model in model_names:
            for event, value in physical[model]["events"].items():
                grid = value["grid_cell"]
                case = value["forecast_case"]
                components = value["components"]

                def number(item: float | int | None) -> str:
                    return "NA" if item is None else f"{float(item):.4f}"

                tp_rmse = components.get("tp_6h_mm", {}).get("event_conditioned_rmse")
                wind_rmse = components.get("max_wind_speed_ms", {}).get(
                    "event_conditioned_rmse"
                )
                lines.append(
                    f"| {model} | {event} | {number(grid['pod'])} | {number(grid['far'])} "
                    f"| {number(grid['csi'])} | {number(case['pod'])} | {number(case['far'])} "
                    f"| {number(case['csi'])} | {number(tp_rmse)} | {number(wind_rmse)} |"
                )
    readme = destination / "README.md"
    readme.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    files = [
        {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256_lf": v9_common.normalized_lf_sha256(path),
        }
        for path in (output, readme)
    ]
    v9_common.write_json(
        destination / "manifest.json",
        {
            "schema_version": 1,
            "scientific_status": result["scientific_status"],
            "test_years_read": [2025],
            "no_training_or_tuning": True,
            "files": files,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v9_2025_event_extension.yaml")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--max-batches", type=int)
    args = parser.parse_args()

    config = v9_common.read_yaml(resolve(args.config))
    benchmark_path = resolve(config["frozen_benchmark"])
    if v9_common.normalized_lf_sha256(benchmark_path) != str(
        config["frozen_benchmark_sha256"]
    ):
        raise ValueError("Frozen V9 2025 benchmark SHA-256 mismatch")
    original = json.loads(benchmark_path.read_text(encoding="utf-8"))
    if original.get("scientific_status") != "one_time_2025_final_test":
        raise ValueError("Event extension requires the frozen one-time 2025 result")
    final_config = v9_common.read_yaml(resolve(config["final_test_config"]))
    frozen.validate_protocol(final_config)
    data_config = load_config(resolve(final_config["data_config"]))
    data_config["data"]["window_stride_hours"] = int(final_config["window_stride_hours"])
    dataset = make_dataset(data_config, [2025], augment=False)
    variables = list(data_config["data"]["target_variables"])
    if variables != list(v9_common.TARGETS) or len(dataset.station_ids) != 390:
        raise ValueError("Frozen event extension requires five targets and 390 stations")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    models, checkpoint_metadata = v9_common.load_models(final_config, data_config, device, 390)
    static_data = StaticFields.load(resolve(data_config["data"]["static_fields"]))
    static = static_data.as_tensor().unsqueeze(0).to(device)
    land_mask = torch.from_numpy(static_data.land_sea_mask)
    sea_mask = land_mask < 0.5
    normalization = NormalizationStats.load(resolve(final_config["normalization_stats"]))
    forecast_hours = int(data_config["data"]["forecast_hours"])
    history_hours = int(data_config["data"]["history_hours"])
    include_percentile = bool(config.get("include_percentile_events", True))
    thresholds = (
        derive_event_thresholds(
            dataset,
            variables,
            normalization,
            static_data.land_sea_mask,
            list(config["threshold_years"]),
            config["events"],
        )
        if include_percentile
        else None
    )
    names = tuple(v9_common.FORECAST_NAMES)
    metric_accumulators = {
        name: ForecastMetricAccumulator(tuple(variables), forecast_hours) for name in names
    }
    event_accumulators = (
        {name: make_event_accumulators(forecast_hours, thresholds) for name in names}
        if thresholds is not None
        else {}
    )
    physical_spec = config.get("physical_six_hour_events")
    physical_accumulators = (
        {
            name: PhysicalSixHourEventAccumulator(
                forecast_hours,
                thresholds=physical_spec.get("thresholds"),
                region_name=str(physical_spec.get("region", "sea")),
            )
            for name in names
        }
        if physical_spec
        else {}
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["evaluation"]["batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    expected_batches = min(len(loader), args.max_batches) if args.max_batches else len(loader)
    processed = 0
    started = time.perf_counter()
    with torch.no_grad():
        for batch_index, raw in enumerate(loader):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            batch = move(raw, device)
            target = denormalize_channels(batch["target"], variables, normalization)
            v9_normalized, _ = v9_common.v9_forward_components(models["v9_a"], batch, static)
            v7_normalized = v9_common.v7_forward(models["v7_b"], batch, static)
            predictions = {
                "v9_a": denormalize_channels(v9_normalized, variables, normalization),
                "v7_b": denormalize_channels(v7_normalized, variables, normalization),
            }
            current_batch = {
                "point_values": batch["point_values"][:, -1:],
                "value_mask": batch["value_mask"][:, -1:],
                "observation_age": batch["observation_age"][:, -1:],
                "point_coords": batch["point_coords"],
                "point_static": batch["point_static"],
                "target": batch["target"][:, :1],
            }
            reconstructed = v9_common.reconstruction_forward(
                models["v7_b"], current_batch, static
            )[:, 0]
            reconstructed = denormalize_channels(
                reconstructed[:, None], variables, normalization
            )[:, 0]
            predictions["sparse_reconstruction_persistence"] = dense_grid_persistence(
                reconstructed, forecast_hours
            )
            current_indices = raw["start_index"].numpy() + history_hours - 1
            current_normalized = torch.from_numpy(
                np.asarray(dataset.target_grids[current_indices], dtype=np.float32).copy()
            ).to(device)
            dense_current = denormalize_channels(
                current_normalized[:, None], variables, normalization
            )[:, 0]
            predictions["dense_era5_persistence"] = dense_grid_persistence(
                dense_current, forecast_hours
            )
            for name, prediction in predictions.items():
                metric_accumulators[name].update(prediction, target, land_mask)
                if event_accumulators:
                    update_events(
                        event_accumulators[name], prediction, target, sea_mask, variables
                    )
                if physical_accumulators:
                    physical_accumulators[name].update(
                        prediction, target, sea_mask, variables
                    )
            processed += int(target.shape[0])
            completed = batch_index + 1
            every = int(config["evaluation"]["progress_every_batches"])
            if every and (completed % every == 0 or completed == expected_batches):
                print(f"  V9 2025 events {completed}/{expected_batches}", flush=True)

    metrics = {name: value.compute() for name, value in metric_accumulators.items()}
    bounded = args.max_batches is not None and args.max_batches < len(loader)
    if not bounded:
        assert_replay_matches(metrics, original["metrics"])
    event_metrics = {
        name: {event: accumulator.compute() for event, accumulator in values.items()}
        for name, values in event_accumulators.items()
    }
    physical_event_metrics = {
        name: accumulator.compute() for name, accumulator in physical_accumulators.items()
    }
    full_status = str(
        config.get("scientific_status", "frozen_2025_event_metric_extension")
    )
    result = {
        "schema_version": 2 if physical_event_metrics else 1,
        "scientific_status": (
            "bounded_event_pipeline_check" if bounded else full_status
        ),
        "contract": {
            "test_years_read": [2025],
            "threshold_years": list(config.get("threshold_years", [])),
            "samples": processed,
            "batches": expected_batches,
            "checkpoint_and_predictions_unchanged": True,
            "no_training_or_post_test_tuning": True,
        },
        "frozen_benchmark_sha256": str(config["frozen_benchmark_sha256"]),
        "checkpoint_metadata": checkpoint_metadata,
        "event_thresholds": thresholds,
        "metric_replay": metrics,
        "event_metrics": event_metrics,
        "physical_six_hour_event_metrics": physical_event_metrics,
        "elapsed_seconds": time.perf_counter() - started,
    }
    output_dir = resolve(config["evaluation"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "events.json"
    v9_common.write_json(output, result)
    if not bounded:
        publish(output, resolve(config["evaluation"]["publish_dir"]), result)
    print(f"Event extension complete: {output}", flush=True)
    dataset.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
