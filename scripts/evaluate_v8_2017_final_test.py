#!/usr/bin/env python3
"""Run the acknowledged, one-time frozen V8/V7-B comparison on 2017."""

from __future__ import annotations

import argparse
import hashlib
import json
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

from check_v7 import forward, load_config, make_dataset, move, resolve  # noqa: E402
from evaluate_v8_2016_benchmark import (  # noqa: E402
    BASELINE_NAMES,
    TARGETS,
    derive_event_thresholds,
    load_models,
    make_event_accumulators,
    read_yaml,
    skill,
    update_events,
    write_json,
)
from stormengine_dl.baselines import dense_grid_persistence  # noqa: E402
from stormengine_dl.data import NormalizationStats, StaticFields  # noqa: E402
from stormengine_dl.runtime import denormalize_channels  # noqa: E402
from stormengine_dl.training import ForecastMetricAccumulator  # noqa: E402
from train_v8_stage3 import reconstruction_forward, sha256_file  # noqa: E402


def validate_unlock_evidence(config: dict[str, Any]) -> dict[str, Any]:
    spec = config["unlock_evidence"]
    path = resolve(spec["path"])
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen 2016 unlock evidence: {path}")
    # The published manifest defines the hash over Git-ready LF bytes. Normalize
    # a Windows CRLF checkout before verification without altering the file.
    actual_hash = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if actual_hash != str(spec["sha256"]):
        raise ValueError(f"2016 unlock-evidence SHA-256 mismatch: {actual_hash}")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("scientific_status") != spec["required_scientific_status"]:
        raise ValueError("The 2016 benchmark is not the frozen validation result")
    passed = bool(result.get("acceptance", {}).get("both_seeds_passed"))
    if passed is not bool(spec["required_both_seeds_passed"]):
        raise ValueError("The pre-registered 2016 gate did not unlock the final test")
    if result.get("contract", {}).get("test_years_read") != []:
        raise ValueError("Unlock evidence unexpectedly reports a prior test-year read")
    return {
        "path": str(path),
        "sha256": actual_hash,
        "scientific_status": result["scientific_status"],
        "both_stage3a_seeds_passed": passed,
    }


def final_decision(skills: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    spec = config["decision"]
    candidate = str(spec["candidate"])
    reference = str(spec["reference"])
    comparison = skills[f"{candidate}_vs_{reference}"]
    sea_skills = {
        variable: float(comparison["aggregate"]["sea"][variable]["skill"])
        for variable in TARGETS
    }
    wind_wins = {
        variable: sum(
            float(comparison["by_lead_hour"][str(lead)]["sea"][variable]["skill"]) > 0
            for lead in range(1, 7)
        )
        for variable in ("u10", "v10")
    }
    total_wind_wins = sum(wind_wins.values())
    mean_sea_skill = sum(sea_skills.values()) / len(sea_skills)
    minimum_wind_wins = int(spec["minimum_positive_sea_wind_component_leads"])
    supports_replacement = mean_sea_skill > 0 and total_wind_wins >= minimum_wind_wins
    return {
        "rule_predeclared_before_2017": True,
        "candidate": candidate,
        "reference": reference,
        "mean_sea_rmse_skill": mean_sea_skill,
        "sea_rmse_skill_by_variable": sea_skills,
        "positive_sea_wind_component_leads": wind_wins,
        "positive_sea_wind_component_leads_total": total_wind_wins,
        "minimum_required_positive_sea_wind_component_leads": minimum_wind_wins,
        "supports_v8_replacement_claim": supports_replacement,
        "interpretation": (
            "The locked test evaluates the pre-declared V8 replacement claim. "
            "It is not followed by test-driven hyperparameter tuning. Event metrics "
            "and dense persistence remain scientific diagnostics rather than hidden gates."
        ),
    }


def publish_result(source: Path, destination: Path, result: dict[str, Any]) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    benchmark_path = destination / "benchmark.json"
    benchmark_path.write_bytes(source.read_bytes())
    decision = result["decision"]
    metrics = result["metrics"]
    models = ("v7_b", "v8_stage3a_seed42")
    lines = [
        "# V8 one-time 2017 final test",
        "",
        "This is the locked 2017 comparison. Models and thresholds were frozen before "
        "the test was read; no post-test tuning is permitted.",
        "",
        f"- Samples: `{result['contract']['samples']}`",
        f"- Window stride: `{result['contract']['window_stride_hours']} h`",
        f"- V8 replacement claim supported: `{decision['supports_v8_replacement_claim']}`",
        f"- Mean sea RMSE skill of V8 vs V7-B: `{100 * decision['mean_sea_rmse_skill']:.2f}%`",
        f"- Positive sea-wind component/leads: `{decision['positive_sea_wind_component_leads_total']}/12`",
        "",
        "| model | msl sea RMSE | u10 sea RMSE | v10 sea RMSE | t2m sea RMSE | tp sea RMSE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in models:
        sea = metrics[name]["aggregate"]["sea"]
        lines.append(
            f"| {name} | {sea['msl']['rmse']:.4f} | {sea['u10']['rmse']:.4f} "
            f"| {sea['v10']['rmse']:.4f} | {sea['t2m']['rmse']:.4f} "
            f"| {sea['tp']['rmse']:.4f} |"
        )
    lines.extend(
        [
            "",
            "`benchmark.json` contains full/land/sea MAE and RMSE, +1--+6 h "
            "metrics, persistence skills, and frozen-threshold event verification.",
            "",
        ]
    )
    readme_path = destination / "README.md"
    readme_path.write_bytes(("\n".join(lines)).encode("utf-8"))
    files = []
    for path in (benchmark_path, readme_path):
        files.append(
            {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    write_json(
        destination / "manifest.json",
        {
            "schema_version": 1,
            "hash_scope": "Git-ready LF bytes generated by the final-test evaluator",
            "test_years_read": [2017],
            "one_time_test": True,
            "files": files,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v8_2017_final_test.yaml")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--publish-dir")
    parser.add_argument("--acknowledge-one-time-test", action="store_true")
    args = parser.parse_args()
    if not args.acknowledge_one_time_test:
        raise ValueError("Final test is locked; pass --acknowledge-one-time-test explicitly")

    config = read_yaml(resolve(args.config))
    unlock = validate_unlock_evidence(config)
    test_years = list(config["test_years"])
    if test_years != [2017]:
        raise ValueError("The one-time final test split must be exactly [2017]")
    threshold_years = list(config["threshold_years"])
    if threshold_years != list(range(2010, 2016)):
        raise ValueError("Final event thresholds must stay frozen to 2010--2015")

    output = resolve(args.output_dir or config["evaluation"]["output_dir"])
    result_path = output / "benchmark.json"
    if result_path.exists() and args.max_batches is None:
        raise FileExistsError(
            f"Refusing to rerun the one-time final test over existing result: {result_path}"
        )

    data_config = load_config(resolve(config["data_config"]))
    if list(data_config["data"]["test_years"]) != [2017]:
        raise ValueError("The frozen model data contract no longer declares 2017 as test")
    data_config["data"]["window_stride_hours"] = int(config["window_stride_hours"])
    dataset = make_dataset(data_config, test_years, augment=False)
    variables = list(data_config["data"]["target_variables"])
    if variables != list(TARGETS) or len(dataset.station_ids) != 390:
        raise ValueError("Final test requires the frozen five-target, 390-station contract")

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    evaluated_models = tuple(config["models"])
    sparse_name = str(config["baselines"]["sparse_reconstruction_checkpoint"])
    load_names = tuple(dict.fromkeys((*evaluated_models, sparse_name)))
    models, checkpoint_metadata = load_models(
        config, device, len(dataset.station_ids), model_names=load_names
    )
    static_data = StaticFields.load(resolve(data_config["data"]["static_fields"]))
    static = static_data.as_tensor().unsqueeze(0).to(device)
    land_mask = torch.from_numpy(static_data.land_sea_mask)
    sea_mask = land_mask < 0.5
    normalization = NormalizationStats.load(resolve("data/normalization/era5_2010_2015.json"))
    forecast_hours = int(data_config["data"]["forecast_hours"])
    history_hours = int(data_config["data"]["history_hours"])
    loader = DataLoader(
        dataset,
        batch_size=int(config["evaluation"]["batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    thresholds = derive_event_thresholds(
        dataset,
        variables,
        normalization,
        static_data.land_sea_mask,
        threshold_years,
        config["events"],
    )
    prediction_names = (*evaluated_models, *BASELINE_NAMES)
    metric_accumulators = {
        name: ForecastMetricAccumulator(tuple(variables), forecast_hours)
        for name in prediction_names
    }
    event_accumulators = {
        name: make_event_accumulators(forecast_hours, thresholds)
        for name in prediction_names
    }
    expected_batches = min(len(loader), args.max_batches) if args.max_batches else len(loader)
    processed = 0
    started = time.perf_counter()
    with torch.no_grad():
        for batch_index, raw in enumerate(loader):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            batch = move(raw, device)
            target = denormalize_channels(batch["target"], variables, normalization)
            predictions = {
                name: denormalize_channels(forward(models[name], batch, static), variables, normalization)
                for name in evaluated_models
            }
            current_batch = {
                "point_values": batch["point_values"][:, -1:],
                "value_mask": batch["value_mask"][:, -1:],
                "observation_age": batch["observation_age"][:, -1:],
                "point_coords": batch["point_coords"],
                "point_static": batch["point_static"],
                "target": batch["target"][:, :1],
            }
            reconstructed = reconstruction_forward(models[sparse_name], current_batch, static)[:, 0]
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
            current_dense = denormalize_channels(
                current_normalized[:, None], variables, normalization
            )[:, 0]
            predictions["dense_era5_persistence"] = dense_grid_persistence(
                current_dense, forecast_hours
            )
            for name, prediction in predictions.items():
                metric_accumulators[name].update(prediction, target, land_mask)
                update_events(event_accumulators[name], prediction, target, sea_mask, variables)
            processed += int(target.shape[0])
            completed = batch_index + 1
            every = int(config["evaluation"]["progress_every_batches"])
            if every > 0 and (completed % every == 0 or completed == expected_batches):
                elapsed = time.perf_counter() - started
                eta = elapsed / completed * max(0, expected_batches - completed)
                print(
                    f"  one-time 2017 test {completed:,}/{expected_batches:,} "
                    f"elapsed={elapsed / 60:.1f}m ETA={eta / 60:.1f}m",
                    flush=True,
                )

    metrics = {name: accumulator.compute() for name, accumulator in metric_accumulators.items()}
    event_metrics = {
        name: {event: accumulator.compute() for event, accumulator in values.items()}
        for name, values in event_accumulators.items()
    }
    skills: dict[str, Any] = {}
    for candidate in evaluated_models:
        for baseline in BASELINE_NAMES:
            skills[f"{candidate}_vs_{baseline}"] = skill(metrics[candidate], metrics[baseline])
    skills["v8_stage3a_seed42_vs_v7_b"] = skill(
        metrics["v8_stage3a_seed42"], metrics["v7_b"]
    )
    bounded = args.max_batches is not None and args.max_batches < len(loader)
    result = {
        "schema_version": 1,
        "scientific_status": "bounded_pipeline_check" if bounded else "one_time_2017_final_test",
        "contract": {
            "test_years_read": [2017],
            "threshold_years": threshold_years,
            "window_stride_hours": int(config["window_stride_hours"]),
            "history_hours": history_hours,
            "forecast_hours": forecast_hours,
            "station_count": len(dataset.station_ids),
            "target_variables": variables,
            "samples": processed,
            "batches": expected_batches,
            "models_frozen_before_test": list(evaluated_models),
            "baseline_definitions": config["baselines"],
            "no_post_test_tuning": True,
        },
        "unlock_evidence": unlock,
        "checkpoint_metadata": checkpoint_metadata,
        "event_thresholds": thresholds,
        "metrics": metrics,
        "rmse_skills": skills,
        "event_metrics": event_metrics,
        "decision": final_decision(skills, config),
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(result_path, result)
    if args.publish_dir or config["evaluation"].get("publish_dir"):
        if bounded:
            raise ValueError("A bounded check cannot be published as the final test")
        destination = resolve(args.publish_dir or config["evaluation"]["publish_dir"])
        publish_result(result_path, destination, result)
    print(
        f"Final test complete: status={result['scientific_status']} "
        f"supports_v8={result['decision']['supports_v8_replacement_claim']} "
        f"output={result_path}",
        flush=True,
    )
    dataset.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
