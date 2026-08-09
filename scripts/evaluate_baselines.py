#!/usr/bin/env python3
"""Evaluate dense and input-fair persistence baselines on StormEngine windows."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from stormengine_dl.baselines import (
    build_idw_neighbors,
    dense_grid_persistence,
    geographic_station_coordinates,
    rmse_skill_scores,
    sparse_idw_persistence,
)
from stormengine_dl.data import CachedEra5SequenceDataset, NormalizationStats, StaticFields
from stormengine_dl.runtime import (
    denormalize_channels,
    make_dataset,
    require_training_cache,
    resolve_path,
)
from stormengine_dl.training import ForecastMetricAccumulator


def _denormalize_points(
    values: torch.Tensor, variables: list[str], normalization: NormalizationStats
) -> torch.Tensor:
    result = values.detach().float().cpu().clone()
    for channel, variable in enumerate(variables):
        stat = normalization.variables[variable]
        result[:, :, channel] = result[:, :, channel] * stat.std + stat.mean
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/era5_2010_2017.yaml")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--output-dir", help="Defaults to artifacts/baselines_<split>")
    parser.add_argument("--v6-metrics", help="Optional V6 metrics_by_lead.json for skill scores")
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config_path = resolve_path(repo_root, args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data = config["data"]
    require_training_cache(repo_root, data, config_path)
    years_field = "validation_years" if args.split == "validation" else "test_years"
    years = list(data[years_field])
    dataset = make_dataset(repo_root, data, years, dropout=0.0)
    if not isinstance(dataset, CachedEra5SequenceDataset):
        raise RuntimeError("baseline evaluation requires the normalized hourly training cache")

    static = StaticFields.load(resolve_path(repo_root, data["static_fields"]))
    normalization = NormalizationStats.load(resolve_path(repo_root, data["normalization_stats"]))
    input_variables = list(data["input_variables"])
    target_variables = list(data["target_variables"])
    forecast_hours = int(data["forecast_hours"])
    history_hours = int(data["history_hours"])
    height, width = len(static.latitudes), len(static.longitudes)
    station_coordinates = geographic_station_coordinates(
        np.asarray(dataset.normalized_station_coordinates), static.latitudes, static.longitudes
    )
    neighbor_indices, neighbor_weights = build_idw_neighbors(
        station_coordinates,
        static.latitudes,
        static.longitudes,
        neighbors=args.neighbors,
        power=args.power,
    )

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    accumulators = {
        "dense_grid_persistence": ForecastMetricAccumulator(
            tuple(target_variables), forecast_hours
        ),
        "sparse_idw_persistence": ForecastMetricAccumulator(
            tuple(target_variables), forecast_hours
        ),
    }
    land_mask = torch.from_numpy(static.land_sea_mask)
    expected_batches = min(len(loader), args.max_batches) if args.max_batches else len(loader)
    processed_samples = 0
    started = time.time()

    for batch_index, batch in enumerate(loader):
        if args.max_batches is not None and batch_index >= args.max_batches:
            break
        target = denormalize_channels(batch["target"], target_variables, normalization)
        start_indices = batch["start_index"].numpy().astype(np.int64)
        current_indices = start_indices + history_hours - 1
        current_dense_normalized = torch.from_numpy(
            np.asarray(dataset.target_grids[current_indices], dtype=np.float32)
        )
        current_dense = denormalize_channels(
            current_dense_normalized[:, None], target_variables, normalization
        )[:, 0]
        dense_prediction = dense_grid_persistence(current_dense, forecast_hours)

        current_points = _denormalize_points(
            batch["point_values"][:, -1], input_variables, normalization
        )
        sparse_prediction = sparse_idw_persistence(
            current_points,
            input_variables,
            target_variables,
            neighbor_indices,
            neighbor_weights,
            height,
            width,
            forecast_hours,
        )
        accumulators["dense_grid_persistence"].update(dense_prediction, target, land_mask)
        accumulators["sparse_idw_persistence"].update(sparse_prediction, target, land_mask)
        processed_samples += target.shape[0]

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

    baseline_results = {
        name: {"metrics": accumulator.compute()} for name, accumulator in accumulators.items()
    }
    baseline_results["dense_grid_persistence"]["input_assumption"] = (
        "last full ERA5 target grid; strong reference with more information than V6"
    )
    baseline_results["sparse_idw_persistence"].update(
        {
            "input_assumption": "same last-hour sparse point values available to V6",
            "interpolation": {
                "method": "metric-aware inverse-distance weighting",
                "neighbors": args.neighbors,
                "power": args.power,
            },
            "precipitation_strategy": "zero hourly precipitation because tp is absent from V6 input",
        }
    )
    result: dict[str, object] = {
        "schema_version": 1,
        "split": args.split,
        "years": years,
        "samples_evaluated": processed_samples,
        "batches_evaluated": expected_batches,
        "variables": target_variables,
        "forecast_hours": forecast_hours,
        "baselines": baseline_results,
    }

    if args.v6_metrics:
        v6_path = resolve_path(repo_root, args.v6_metrics)
        v6 = json.loads(v6_path.read_text(encoding="utf-8"))
        if v6.get("split") != args.split or list(v6.get("years", [])) != years:
            raise ValueError("V6 metrics split/years do not match this baseline evaluation")
        if int(v6.get("samples_evaluated", -1)) != processed_samples:
            raise ValueError(
                "V6 metrics sample count does not match this baseline evaluation; "
                "do not compute skill from a bounded smoke run"
            )
        result["v6_metrics_source"] = str(Path(args.v6_metrics))
        result["v6_skill"] = {
            name: rmse_skill_scores(v6["metrics"], values["metrics"])
            for name, values in baseline_results.items()
        }

    output_dir = (
        resolve_path(repo_root, args.output_dir)
        if args.output_dir
        else repo_root / "artifacts" / f"baselines_{args.split}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "baseline_metrics.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("\nAggregate denormalized RMSE")
    for name, values in baseline_results.items():
        print(f"  {name}")
        aggregate = values["metrics"]["aggregate"]
        for region in ("full", "land", "sea"):
            summary = " ".join(
                f"{variable}={aggregate[region][variable]['rmse']:.4f}"
                for variable in target_variables
            )
            print(f"    {region}: {summary}")
    print(f"Baseline evaluation: {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
