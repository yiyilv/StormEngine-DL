#!/usr/bin/env python3
"""Evaluate dense persistence on the Processor development validation windows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from stormengine_dl.data import StaticFields  # noqa: E402
from stormengine_dl.training import ForecastMetricAccumulator, sea_weight_map, weighted_mse  # noqa: E402
from train_dense_processor import (  # noqa: E402
    denormalize_channels,
    load_config,
    load_normalization,
    make_dense_dataset,
    resolve,
)


def evaluate(config: dict[str, Any], batch_size: int | None) -> dict[str, Any]:
    data, training = config["data"], config["training"]
    development = config["processor_development"]
    batch_size = int(batch_size or development["batch_size"])
    validation = make_dense_dataset(config, data["validation_years"])
    loader = DataLoader(validation, batch_size=batch_size, shuffle=False, num_workers=0)
    static = StaticFields.load(resolve(data["static_fields"])).as_tensor().unsqueeze(0)
    weights = sea_weight_map(static[0, 0], float(training["sea_weight"]))
    variables = list(data["target_variables"])
    normalization = load_normalization()
    metrics = ForecastMetricAccumulator(tuple(variables), int(data["forecast_hours"]))
    batch_mean_loss = 0.0
    sample_weighted_loss = 0.0
    batches = 0
    samples = 0
    with torch.no_grad():
        for raw in loader:
            history = raw["history"]
            target = raw["target"]
            prediction = history[:, -1:].expand(-1, target.shape[1], -1, -1, -1)
            current_batch = int(target.shape[0])
            loss = float(weighted_mse(prediction, target, weights))
            batch_mean_loss += loss
            sample_weighted_loss += loss * current_batch
            batches += 1
            samples += current_batch
            metrics.update(
                denormalize_channels(prediction, variables, normalization),
                denormalize_channels(target, variables, normalization),
                static[0, 0],
            )
    result = {
        "schema_version": 1,
        "scientific_status": "processor_development_validation_baseline",
        "baseline": "dense_persistence",
        "definition": "repeat the final history grid at each of the six future lead hours",
        "training_required": False,
        "protocol_train_years": list(data["train_years"]),
        "validation_years": list(data["validation_years"]),
        "test_years_read": [],
        "history_hours": int(data["history_hours"]),
        "forecast_hours": int(data["forecast_hours"]),
        "window_stride_hours": int(development["window_stride_hours"]),
        "validation_samples": samples,
        "validation_batches": batches,
        "evaluation_batch_size": batch_size,
        "variables": variables,
        "sea_weight": float(training["sea_weight"]),
        # The primary value exactly mirrors train_dense_processor.run_epoch,
        # which averages per-batch losses. The sample-weighted value is also
        # retained to expose the tiny final-partial-batch difference.
        "normalized_sea_weighted_validation_loss": batch_mean_loss / max(1, batches),
        "sample_weighted_normalized_validation_loss": sample_weighted_loss / max(1, samples),
        "validation_metrics": metrics.compute(),
    }
    validation.close()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v8_processor_dev3y_convgru.yaml")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--output", default="artifacts/v8_processor_dev3y_persistence_2016.json"
    )
    args = parser.parse_args()
    if args.batch_size is not None and args.batch_size < 1:
        raise ValueError("batch size must be positive")
    result = evaluate(load_config(resolve(args.config)), args.batch_size)
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    print(f"Persistence result: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
