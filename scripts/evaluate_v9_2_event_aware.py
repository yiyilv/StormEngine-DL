#!/usr/bin/env python3
"""Compare a V9.2 candidate with its frozen V9.1 source on 2018 validation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from check_v7 import load_config, make_dataset, move, resolve  # noqa: E402
from train_v9_output_form import forward, make_model, require_development_protocol  # noqa: E402
from stormengine_dl.baselines import dense_grid_persistence  # noqa: E402
from stormengine_dl.data import NormalizationStats, StaticFields  # noqa: E402
from stormengine_dl.event_metrics import PhysicalSixHourEventAccumulator  # noqa: E402
from stormengine_dl.runtime import denormalize_channels  # noqa: E402
from stormengine_dl.training import ForecastMetricAccumulator  # noqa: E402


def load_weights(model: torch.nn.Module, path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return checkpoint.get("model_contract", {})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v9_2_event_aware.yaml")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--max-batches", type=int)
    args = parser.parse_args()

    config = load_config(resolve(args.config))
    require_development_protocol(config)
    if config["data"]["validation_years"] != [2018] or config["data"]["test_years"]:
        raise ValueError("V9.2 evaluation is restricted to 2018 validation only")
    dataset = make_dataset(config, [2018], augment=False)
    variables = list(config["data"]["target_variables"])
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    source_path = resolve(config["development"]["source_checkpoint"])
    candidate_path = resolve(args.candidate)
    source = make_model(config, "autoregressive", "field")
    candidate = make_model(config, "autoregressive", "field")
    contracts = {
        "v9_1_source": load_weights(source, source_path),
        "v9_2_candidate": load_weights(candidate, candidate_path),
    }
    source = source.to(device).eval()
    candidate = candidate.to(device).eval()
    static_data = StaticFields.load(resolve(config["data"]["static_fields"]))
    static = static_data.as_tensor().unsqueeze(0).to(device)
    land_mask = torch.from_numpy(static_data.land_sea_mask)
    sea_mask = land_mask < 0.5
    normalization = NormalizationStats.load(resolve(config["data"]["normalization_stats"]))
    forecast_hours = int(config["data"]["forecast_hours"])
    thresholds = config["training"]["event_aware"]["thresholds"]
    names = ("v9_1_source", "v9_2_candidate", "dense_era5_persistence")
    field = {
        name: ForecastMetricAccumulator(tuple(variables), forecast_hours) for name in names
    }
    events = {
        name: PhysicalSixHourEventAccumulator(
            forecast_hours, thresholds=thresholds, region_name="sea"
        )
        for name in names
    }
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    expected = min(len(loader), args.max_batches) if args.max_batches else len(loader)
    processed = 0
    started = time.perf_counter()
    with torch.no_grad():
        for batch_index, raw in enumerate(loader):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            batch = move(raw, device)
            target = denormalize_channels(batch["target"], variables, normalization)
            source_prediction = denormalize_channels(
                forward(source, batch, static), variables, normalization
            )
            candidate_prediction = denormalize_channels(
                forward(candidate, batch, static), variables, normalization
            )
            current_indices = raw["start_index"].numpy() + int(config["data"]["history_hours"]) - 1
            current = torch.from_numpy(
                dataset.target_grids[current_indices].copy()
            ).to(device)
            current = denormalize_channels(current[:, None], variables, normalization)[:, 0]
            predictions = {
                "v9_1_source": source_prediction,
                "v9_2_candidate": candidate_prediction,
                "dense_era5_persistence": dense_grid_persistence(current, forecast_hours),
            }
            for name, prediction in predictions.items():
                field[name].update(prediction, target, land_mask)
                events[name].update(prediction, target, sea_mask, variables)
            processed += int(target.shape[0])
            if (batch_index + 1) % 25 == 0 or batch_index + 1 == expected:
                print(f"V9.2 validation {batch_index + 1}/{expected}", flush=True)

    result = {
        "schema_version": 1,
        "scientific_status": (
            "bounded_2018_pipeline_check"
            if args.max_batches is not None and args.max_batches < len(loader)
            else "v9_2_2018_development_evaluation"
        ),
        "contract": {
            "years_read": [2018],
            "2019_read": False,
            "2019_already_exposed": True,
            "samples": processed,
            "candidate_checkpoint": str(candidate_path),
            "source_checkpoint": str(source_path),
        },
        "checkpoint_contracts": contracts,
        "field_metrics": {name: accumulator.compute() for name, accumulator in field.items()},
        "physical_six_hour_event_metrics": {
            name: accumulator.compute() for name, accumulator in events.items()
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    suffix = "" if candidate_path.stem == "best" else f"_{candidate_path.stem}"
    output = candidate_path.parent / f"development_evaluation_2018{suffix}.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"V9.2 development evaluation complete: {output}", flush=True)
    dataset.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
