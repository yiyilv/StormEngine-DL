#!/usr/bin/env python3
"""Evaluate a frozen StormEngine checkpoint without changing model selection."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from stormengine_dl.data import NormalizationStats, StaticFields
from stormengine_dl.runtime import (
    denormalize_channels,
    forecast,
    make_dataset,
    make_model,
    move_batch,
    require_training_cache,
    resolve_path,
    select_device,
)
from stormengine_dl.training import ForecastMetricAccumulator


def _example_indices(sample_count: int, requested: int) -> set[int]:
    if requested <= 0 or sample_count <= 0:
        return set()
    count = min(requested, sample_count)
    return set(np.linspace(0, sample_count - 1, num=count, dtype=np.int64).tolist())


def _forecast_times(dataset: Any, start_index: int, history_hours: int, steps: int) -> np.ndarray:
    source = dataset.all_times if hasattr(dataset, "all_times") else dataset.times
    first = np.asarray(source)[start_index] + np.timedelta64(history_hours, "h")
    return first + np.arange(steps) * np.timedelta64(1, "h")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/era5_2010_2017.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu", "mps"), default="auto")
    parser.add_argument("--output-dir", help="Defaults to <checkpoint-dir>/evaluation_<split>")
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--save-examples", type=int, default=3)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config_path = resolve_path(repo_root, args.config)
    checkpoint_path = resolve_path(repo_root, args.checkpoint)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data, training = config["data"], config["training"]
    require_training_cache(repo_root, data, config_path)

    device = select_device(args.device)
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    years_field = "validation_years" if args.split == "validation" else "test_years"
    years = list(data[years_field])
    dataset = make_dataset(repo_root, data, years, dropout=0.0)
    batch_size = int(training["batch_size"])
    num_workers = int(training.get("num_workers", 0))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = make_model(config).to(device)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    static = StaticFields.load(resolve_path(repo_root, data["static_fields"]))
    static_fields = static.as_tensor().unsqueeze(0).to(device)
    normalization = NormalizationStats.load(resolve_path(repo_root, data["normalization_stats"]))
    variables = list(data["target_variables"])
    forecast_hours = int(data["forecast_hours"])
    accumulator = ForecastMetricAccumulator(tuple(variables), forecast_hours)
    land_mask = torch.from_numpy(static.land_sea_mask)

    output_dir = (
        resolve_path(repo_root, args.output_dir)
        if args.output_dir
        else checkpoint_path.parent / f"evaluation_{args.split}"
    )
    examples_dir = output_dir / "examples"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_examples > 0:
        examples_dir.mkdir(parents=True, exist_ok=True)

    expected_batches = min(len(loader), args.max_batches) if args.max_batches else len(loader)
    available_samples = min(len(dataset), expected_batches * batch_size)
    requested_examples = _example_indices(available_samples, args.save_examples)
    saved_examples: list[str] = []
    processed_samples = 0
    started = time.time()

    with torch.no_grad():
        for batch_index, raw_batch in enumerate(loader):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            batch = move_batch(raw_batch, device)
            prediction = forecast(model, batch, static_fields)
            prediction_physical = denormalize_channels(prediction, variables, normalization)
            target_physical = denormalize_channels(batch["target"], variables, normalization)
            accumulator.update(prediction_physical, target_physical, land_mask)

            current_batch = prediction_physical.shape[0]
            for local_index in range(current_batch):
                dataset_index = processed_samples + local_index
                if dataset_index not in requested_examples:
                    continue
                start_index = int(raw_batch["start_index"][local_index])
                example_path = examples_dir / f"example_{dataset_index:06d}.npz"
                np.savez_compressed(
                    example_path,
                    prediction=prediction_physical[local_index].numpy(),
                    target=target_physical[local_index].numpy(),
                    latitudes=static.latitudes,
                    longitudes=static.longitudes,
                    variables=np.asarray(variables),
                    forecast_times=_forecast_times(
                        dataset, start_index, int(data["history_hours"]), forecast_hours
                    ).astype("datetime64[ns]"),
                    dataset_index=np.asarray(dataset_index),
                    start_index=np.asarray(start_index),
                )
                saved_examples.append(str(example_path.relative_to(output_dir)))
            processed_samples += current_batch

            completed = batch_index + 1
            if args.progress_every > 0 and (
                completed % args.progress_every == 0 or completed == expected_batches
            ):
                elapsed = time.time() - started
                eta = elapsed / completed * max(0, expected_batches - completed)
                print(
                    f"  {args.split}: {completed:,}/{expected_batches:,} "
                    f"({100 * completed / expected_batches:5.1f}%) "
                    f"elapsed={elapsed / 60:.1f}m ETA={eta / 60:.1f}m",
                    flush=True,
                )

    metric_values = accumulator.compute()
    result = {
        "schema_version": 1,
        "split": args.split,
        "years": years,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(saved["epoch"]) + 1,
        "best_validation_loss": float(saved["best_val_loss"]),
        "samples_evaluated": processed_samples,
        "batches_evaluated": min(expected_batches, len(loader)),
        "variables": variables,
        "forecast_hours": forecast_hours,
        "metrics": metric_values,
        "examples": saved_examples,
    }
    result_path = output_dir / "metrics_by_lead.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("\nAggregate denormalized metrics")
    aggregate = metric_values["aggregate"]
    for region in ("full", "land", "sea"):
        values = " ".join(
            f"{name}:MAE={aggregate[region][name]['mae']:.4f},RMSE={aggregate[region][name]['rmse']:.4f}"
            for name in variables
        )
        print(f"  {region}: {values}")
    print(f"Evaluation: {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
