#!/usr/bin/env python3
"""Run the acknowledged one-time 2025 final test of frozen V9-A."""

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

import evaluate_v9_2024_confirmation as common  # noqa: E402
from check_v7 import load_config, make_dataset, move, resolve  # noqa: E402
from stormengine_dl.baselines import dense_grid_persistence  # noqa: E402
from stormengine_dl.data import NormalizationStats, StaticFields  # noqa: E402
from stormengine_dl.runtime import denormalize_channels  # noqa: E402
from stormengine_dl.training import ForecastMetricAccumulator  # noqa: E402


def validate_protocol(config: dict[str, Any]) -> None:
    expected = {
        "preflight_years": [2023],
        "confirmation_years": [2024],
        "final_test_years": [2025],
    }
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"V9 final-test chronology is not frozen: {mismatches}")
    decision = config["decision"]
    if int(decision["minimum_positive_sea_wind_component_leads"]) != 7:
        raise ValueError("V9 final-test wind gate must remain 7/12")
    if float(decision["maximum_mean_sea_reconstruction_degradation_percent"]) != 3.0:
        raise ValueError("V9 final-test reconstruction gate must remain 3 percent")


def validate_unlock_evidence(config: dict[str, Any]) -> dict[str, Any]:
    spec = config["unlock_evidence"]
    path = resolve(spec["path"])
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen 2024 unlock evidence: {path}")
    digest = common.normalized_lf_sha256(path)
    if digest != str(spec["sha256_lf"]):
        raise ValueError(f"2024 unlock-evidence SHA-256 mismatch: {digest}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("scientific_status") != spec["required_scientific_status"]:
        raise ValueError("Unlock evidence is not the one-time 2024 confirmation")
    if bool(value.get("decision", {}).get("passed")) is not bool(spec["required_passed"]):
        raise ValueError("The frozen 2024 confirmation did not unlock 2025")
    contract = value.get("contract", {})
    if contract.get("confirmation_years_read") != spec["required_confirmation_years_read"]:
        raise ValueError("Unlock evidence has an unexpected confirmation-year record")
    if contract.get("test_years_read") != spec["required_test_years_read"]:
        raise ValueError("Unlock evidence reports a prior final-test read")
    return {
        "path": common.portable(path),
        "sha256_lf": digest,
        "scientific_status": value["scientific_status"],
        "confirmation_passed": True,
    }


def final_decision(
    skills: dict[str, Any],
    reconstruction_metrics: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    result = common.confirmation_decision(skills, reconstruction_metrics, config)
    passed = bool(result.pop("passed"))
    result.pop("next_action")
    result["rule_frozen_before_2025"] = True
    result["supports_v9_replacement_claim"] = passed
    result["no_post_test_tuning"] = True
    result["interpretation"] = (
        "This locked 2025 result is the final V9 replacement test. It is reported as-is "
        "and cannot trigger model, threshold, or checkpoint changes."
    )
    return result


def publish_result(source: Path, destination: Path, result: dict[str, Any]) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    benchmark = destination / "benchmark.json"
    shutil.copyfile(source, benchmark)
    decision = result["decision"]
    metrics = result["metrics"]
    lines = [
        "# V9-A one-time 2025 final test",
        "",
        "The candidate, checkpoint, metrics, and decision gates were frozen before 2025 was read.",
        "No post-test tuning is permitted.",
        "",
        f"- V9 replacement claim supported: `{decision['supports_v9_replacement_claim']}`",
        f"- Mean sea RMSE skill vs V7-B: `{100 * decision['mean_sea_rmse_skill']:.2f}%`",
        f"- Positive sea wind component/leads: `{decision['positive_sea_wind_component_leads_total']}/12`",
        f"- Mean sea reconstruction degradation: `{100 * decision['mean_sea_reconstruction_degradation']:.2f}%`",
        f"- Samples: `{result['contract']['samples']}`",
        "",
        "| model | msl sea RMSE | u10 sea RMSE | v10 sea RMSE | t2m sea RMSE | tp sea RMSE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("v7_b", "v9_a"):
        sea = metrics[name]["aggregate"]["sea"]
        lines.append(
            f"| {name} | {sea['msl']['rmse']:.4f} | {sea['u10']['rmse']:.4f} "
            f"| {sea['v10']['rmse']:.4f} | {sea['t2m']['rmse']:.4f} | {sea['tp']['rmse']:.4f} |"
        )
    (destination / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    files = []
    for path in (benchmark, destination / "README.md"):
        files.append(
            {"path": path.name, "bytes": path.stat().st_size, "sha256": common.sha256_file(path)}
        )
    common.write_json(
        destination / "manifest.json",
        {
            "schema_version": 1,
            "confirmation_years_read": [2024],
            "test_years_read": [2025],
            "one_time_final_test": True,
            "files": files,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v9_2025_final_test.yaml")
    parser.add_argument("--mode", choices=("preflight", "final"), default="preflight")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--acknowledge-one-time-2025", action="store_true")
    args = parser.parse_args()

    config = common.read_yaml(resolve(args.config))
    validate_protocol(config)
    unlock = validate_unlock_evidence(config)
    if args.mode == "final" and not args.acknowledge_one_time_2025:
        raise ValueError("2025 is locked; pass --acknowledge-one-time-2025 explicitly")
    years = list(config["preflight_years"] if args.mode == "preflight" else config["final_test_years"])
    output = resolve(config["evaluation"]["output_dir"])
    result_path = output / "benchmark.json"
    marker = output / "FINAL_TEST_STARTED.json"
    if args.mode == "final" and (marker.exists() or result_path.exists()):
        raise FileExistsError("Refusing to repeat the one-time 2025 final test")

    data_config = load_config(resolve(config["data_config"]))
    data_config["data"]["window_stride_hours"] = int(config["window_stride_hours"])
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    models, checkpoint_metadata = common.load_models(config, data_config, device, 390)
    if args.mode == "final":
        common.write_json(
            marker,
            {
                "protocol": config["protocol"],
                "confirmation_years_read": [2024],
                "test_years_read": [2025],
            },
        )
    dataset = make_dataset(data_config, years, augment=False)
    if len(dataset.station_ids) != 390:
        raise ValueError("V9 final test requires the frozen 390-station contract")
    variables = list(data_config["data"]["target_variables"])
    if variables != list(common.TARGETS):
        raise ValueError(f"V9 final-test target order must be {common.TARGETS}")
    static_data = StaticFields.load(resolve(data_config["data"]["static_fields"]))
    static = static_data.as_tensor().unsqueeze(0).to(device)
    land_mask = torch.from_numpy(static_data.land_sea_mask)
    normalization = NormalizationStats.load(resolve(config["normalization_stats"]))
    forecast_hours = int(data_config["data"]["forecast_hours"])
    history_hours = int(data_config["data"]["history_hours"])
    loader = DataLoader(
        dataset,
        batch_size=int(config["evaluation"]["batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    first_raw = next(iter(loader))
    first = move(first_raw, device)
    with torch.no_grad():
        v9_prediction, v9_current = common.v9_forward_components(models["v9_a"], first, static)
        v7_prediction = common.v7_forward(models["v7_b"], first, static)
    expected_shape = [int(first["target"].shape[0]), 6, 5, 31, 33]
    if list(v9_prediction.shape) != expected_shape or list(v7_prediction.shape) != expected_shape:
        raise ValueError("Frozen final-test model output shape mismatch")
    preflight = {
        "schema_version": 1,
        "mode": args.mode,
        "device": str(device),
        "years_instantiated": years,
        "confirmation_years_read": [2024],
        "test_years_read": [] if args.mode == "preflight" else [2025],
        "samples": len(dataset),
        "batches": len(loader),
        "station_count": len(dataset.station_ids),
        "variables": variables,
        "v9_output_shape": list(v9_prediction.shape),
        "v7_output_shape": list(v7_prediction.shape),
        "v9_current_reconstruction_shape": list(v9_current.shape),
        "unlock_evidence": unlock,
        "checkpoint_metadata": checkpoint_metadata,
    }
    output.mkdir(parents=True, exist_ok=True)
    common.write_json(
        output / ("preflight.json" if args.mode == "preflight" else "final_preflight.json"),
        preflight,
    )
    print(
        f"Preflight OK: years={years} samples={len(dataset):,} batches={len(loader):,} "
        f"test_years_read={preflight['test_years_read']}",
        flush=True,
    )
    if args.mode == "preflight":
        dataset.close()
        return 0

    metrics_accumulators = {
        name: ForecastMetricAccumulator(tuple(variables), forecast_hours)
        for name in common.FORECAST_NAMES
    }
    reconstruction_accumulators = {
        name: ForecastMetricAccumulator(tuple(variables), 1)
        for name in common.RECONSTRUCTION_NAMES
    }
    processed = 0
    started = time.perf_counter()
    with torch.no_grad():
        for batch_index, raw in enumerate(loader):
            batch = move(raw, device)
            target = denormalize_channels(batch["target"], variables, normalization)
            v9_normalized, v9_current_normalized = common.v9_forward_components(
                models["v9_a"], batch, static
            )
            v7_normalized = common.v7_forward(models["v7_b"], batch, static)
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
            v7_current_normalized = common.reconstruction_forward(
                models["v7_b"], current_batch, static
            )[:, 0]
            v7_current = denormalize_channels(
                v7_current_normalized[:, None], variables, normalization
            )[:, 0]
            v9_current = denormalize_channels(
                v9_current_normalized[:, None], variables, normalization
            )[:, 0]
            predictions["sparse_reconstruction_persistence"] = dense_grid_persistence(
                v7_current, forecast_hours
            )
            current_indices = raw["start_index"].numpy() + history_hours - 1
            dense_normalized = torch.from_numpy(
                np.asarray(dataset.target_grids[current_indices], dtype=np.float32).copy()
            ).to(device)
            dense_current = denormalize_channels(
                dense_normalized[:, None], variables, normalization
            )[:, 0]
            predictions["dense_era5_persistence"] = dense_grid_persistence(
                dense_current, forecast_hours
            )
            for name, prediction in predictions.items():
                metrics_accumulators[name].update(prediction, target, land_mask)
            current_target = denormalize_channels(
                batch["current_target"][:, None], variables, normalization
            )
            reconstruction_accumulators["v9_a"].update(
                v9_current[:, None], current_target, land_mask
            )
            reconstruction_accumulators["v7_b"].update(
                v7_current[:, None], current_target, land_mask
            )
            processed += int(target.shape[0])
            completed = batch_index + 1
            every = int(config["evaluation"]["progress_every_batches"])
            if every > 0 and (completed % every == 0 or completed == len(loader)):
                elapsed = time.perf_counter() - started
                eta = elapsed / completed * max(0, len(loader) - completed)
                print(
                    f"  one-time 2025 final test {completed:,}/{len(loader):,} "
                    f"elapsed={elapsed / 60:.1f}m ETA={eta / 60:.1f}m",
                    flush=True,
                )

    metrics = {name: accumulator.compute() for name, accumulator in metrics_accumulators.items()}
    reconstruction_metrics = {
        name: accumulator.compute() for name, accumulator in reconstruction_accumulators.items()
    }
    skills = {
        "v9_a_vs_v7_b": common.skill(metrics["v9_a"], metrics["v7_b"]),
        "v9_a_vs_sparse_reconstruction_persistence": common.skill(
            metrics["v9_a"], metrics["sparse_reconstruction_persistence"]
        ),
        "v9_a_vs_dense_era5_persistence": common.skill(
            metrics["v9_a"], metrics["dense_era5_persistence"]
        ),
    }
    decision = final_decision(skills, reconstruction_metrics, config)
    result = {
        "schema_version": 1,
        "scientific_status": "one_time_2025_final_test",
        "contract": {
            "confirmation_years_read": [2024],
            "test_years_read": [2025],
            "window_stride_hours": int(config["window_stride_hours"]),
            "history_hours": history_hours,
            "forecast_hours": forecast_hours,
            "station_count": len(dataset.station_ids),
            "target_variables": variables,
            "samples": processed,
            "batches": len(loader),
            "no_post_test_tuning": True,
        },
        "unlock_evidence": unlock,
        "checkpoint_metadata": checkpoint_metadata,
        "baseline_definitions": config["baselines"],
        "metrics": metrics,
        "reconstruction_metrics": reconstruction_metrics,
        "rmse_skills": skills,
        "decision": decision,
        "elapsed_seconds": time.perf_counter() - started,
    }
    common.write_json(result_path, result)
    publish_result(result_path, resolve(config["evaluation"]["publish_dir"]), result)
    print(
        f"2025 final test complete: supports_v9={decision['supports_v9_replacement_claim']} "
        f"output={result_path}",
        flush=True,
    )
    dataset.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

