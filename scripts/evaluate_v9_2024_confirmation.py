#!/usr/bin/env python3
"""Evaluate the frozen V9-A candidate once on 2024 while keeping 2025 locked."""

from __future__ import annotations

import argparse
import hashlib
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

from check_v7 import (  # noqa: E402
    contract as v7_contract,
    forward as v7_forward,
    load_config,
    make_dataset,
    make_model as make_v7_model,
    move,
    resolve,
)
from evaluate_v8_2016_benchmark import read_yaml, skill, write_json  # noqa: E402
from stormengine_dl.baselines import dense_grid_persistence  # noqa: E402
from stormengine_dl.data import NormalizationStats, StaticFields  # noqa: E402
from stormengine_dl.models.mask_aware import require_v7_checkpoint_contract  # noqa: E402
from stormengine_dl.runtime import denormalize_channels  # noqa: E402
from stormengine_dl.training import ForecastMetricAccumulator  # noqa: E402
from train_v8_stage3 import reconstruction_forward, sha256_file  # noqa: E402
from train_v9_output_form import (  # noqa: E402
    forward_components as v9_forward_components,
    make_model as make_v9_model,
)


TARGETS = ("msl", "u10", "v10", "t2m", "tp")
FORECAST_NAMES = (
    "v9_a",
    "v7_b",
    "sparse_reconstruction_persistence",
    "dense_era5_persistence",
)
RECONSTRUCTION_NAMES = ("v9_a", "v7_b")


def normalized_lf_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def portable(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def portable_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: portable_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable_value(item) for item in value]
    if isinstance(value, str):
        path = Path(value)
        if path.is_absolute():
            try:
                return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
            except (OSError, ValueError):
                return value
    return value


def validate_protocol(config: dict[str, Any]) -> None:
    expected = {
        "preflight_years": [2023],
        "confirmation_years": [2024],
        "locked_test_years": [2025],
    }
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"V9 confirmation chronology is not frozen: {mismatches}")
    decision = config["decision"]
    if int(decision["minimum_positive_sea_wind_component_leads"]) != 7:
        raise ValueError("V9 confirmation wind gate must remain 7/12")
    if float(decision["maximum_mean_sea_reconstruction_degradation_percent"]) != 3.0:
        raise ValueError("V9 reconstruction gate must remain 3 percent")


def validate_freeze_evidence(config: dict[str, Any]) -> dict[str, Any]:
    spec = config["freeze_evidence"]
    path = resolve(spec["path"])
    if not path.is_file():
        raise FileNotFoundError(f"Missing V9-A freeze evidence: {path}")
    digest = normalized_lf_sha256(path)
    if digest != str(spec["sha256_lf"]):
        raise ValueError(f"V9-A freeze-evidence SHA-256 mismatch: {digest}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value["selected_for_frozen_2024_confirmation"] != spec["required_candidate"]:
        raise ValueError("Freeze evidence does not select the declared V9-A candidate")
    if bool(value["confirmation_year_read"]) is not bool(
        spec["required_confirmation_year_read"]
    ):
        raise ValueError("Freeze evidence reports an unexpected 2024 read")
    if bool(value["locked_test_year_read"]) is not bool(
        spec["required_locked_test_year_read"]
    ):
        raise ValueError("Freeze evidence reports an unexpected 2025 read")
    return {"path": portable(path), "sha256_lf": digest, "candidate": spec["required_candidate"]}


def checkpoint_path(spec: dict[str, Any]) -> tuple[Path, str]:
    path = resolve(spec["path"])
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen checkpoint: {path}")
    digest = sha256_file(path)
    if digest != str(spec["sha256"]):
        raise ValueError(f"Checkpoint SHA-256 mismatch for {path}: {digest}")
    return path, digest


def validate_v9_checkpoint(saved: dict[str, Any], spec: dict[str, Any]) -> None:
    contract = saved.get("model_contract")
    if not isinstance(contract, dict):
        raise ValueError("V9-A checkpoint is missing its model contract")
    expected = {
        "temporal_mode": spec["temporal_mode"],
        "output_mode": spec["output_mode"],
        "learning_rate": float(spec["learning_rate"]),
        "reconstruction_loss_weight": float(spec["reconstruction_loss_weight"]),
        "station_count": 390,
        "train_years": [2020, 2021, 2022],
        "validation_years": [2023],
        "confirmation_years": [2024],
        "locked_test_years": [2025],
    }
    mismatches = {
        key: {"expected": value, "actual": contract.get(key)}
        for key, value in expected.items()
        if contract.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Frozen V9-A checkpoint contract mismatch: {mismatches}")


def load_models(
    config: dict[str, Any],
    data_config: dict[str, Any],
    device: torch.device,
    station_count: int,
) -> tuple[dict[str, torch.nn.Module], dict[str, Any]]:
    v9_spec = config["checkpoints"]["v9_a"]
    v9_path, v9_digest = checkpoint_path(v9_spec)
    v9_saved = torch.load(v9_path, map_location=device, weights_only=False)
    validate_v9_checkpoint(v9_saved, v9_spec)
    v9_model = make_v9_model(
        data_config, str(v9_spec["temporal_mode"]), str(v9_spec["output_mode"])
    ).to(device)
    v9_model.load_state_dict(v9_saved["model_state_dict"], strict=True)
    v9_model.eval()

    v7_spec = config["checkpoints"]["v7_b"]
    v7_path, v7_digest = checkpoint_path(v7_spec)
    v7_saved = torch.load(v7_path, map_location=device, weights_only=False)
    v7_config = load_config(resolve(config["v7_b_config"]))
    v7_model = make_v7_model(v7_config).to(device)
    require_v7_checkpoint_contract(
        v7_saved, v7_contract(v7_config, v7_model, station_count)
    )
    v7_model.load_state_dict(v7_saved["model_state_dict"], strict=True)
    v7_model.eval()
    return {"v9_a": v9_model, "v7_b": v7_model}, {
        "v9_a": {
            "path": portable(v9_path),
            "sha256": v9_digest,
            "checkpoint_contract": portable_value(v9_saved["model_contract"]),
        },
        "v7_b": {
            "path": portable(v7_path),
            "sha256": v7_digest,
            "checkpoint_contract": portable_value(v7_saved.get("model_contract")),
        },
    }


def confirmation_decision(
    skills: dict[str, Any],
    reconstruction_metrics: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    spec = config["decision"]
    comparison = skills["v9_a_vs_v7_b"]
    sea_skills = {
        variable: float(comparison["aggregate"]["sea"][variable]["skill"])
        for variable in TARGETS
    }
    mean_sea_skill = sum(sea_skills.values()) / len(sea_skills)
    wind_wins = {
        variable: sum(
            float(comparison["by_lead_hour"][str(lead)]["sea"][variable]["skill"]) > 0
            for lead in range(1, 7)
        )
        for variable in ("u10", "v10")
    }
    total_wind_wins = sum(wind_wins.values())
    reconstruction_degradation = {}
    for variable in TARGETS:
        candidate = float(reconstruction_metrics["v9_a"]["aggregate"]["sea"][variable]["rmse"])
        reference = float(reconstruction_metrics["v7_b"]["aggregate"]["sea"][variable]["rmse"])
        if reference <= 0:
            raise ValueError("V7-B reconstruction RMSE must be positive")
        reconstruction_degradation[variable] = candidate / reference - 1.0
    mean_reconstruction_degradation = sum(reconstruction_degradation.values()) / len(TARGETS)
    minimum_skill = float(spec["minimum_mean_sea_rmse_skill"])
    minimum_wind_wins = int(spec["minimum_positive_sea_wind_component_leads"])
    maximum_reconstruction = (
        float(spec["maximum_mean_sea_reconstruction_degradation_percent"]) / 100.0
    )
    passed = bool(
        mean_sea_skill > minimum_skill
        and total_wind_wins >= minimum_wind_wins
        and mean_reconstruction_degradation <= maximum_reconstruction
    )
    return {
        "rule_frozen_before_2024": True,
        "candidate": spec["candidate"],
        "reference": spec["reference"],
        "mean_sea_rmse_skill": mean_sea_skill,
        "minimum_mean_sea_rmse_skill_exclusive": minimum_skill,
        "sea_rmse_skill_by_variable": sea_skills,
        "positive_sea_wind_component_leads": wind_wins,
        "positive_sea_wind_component_leads_total": total_wind_wins,
        "minimum_required_positive_sea_wind_component_leads": minimum_wind_wins,
        "sea_reconstruction_degradation_by_variable": reconstruction_degradation,
        "mean_sea_reconstruction_degradation": mean_reconstruction_degradation,
        "maximum_allowed_mean_sea_reconstruction_degradation": maximum_reconstruction,
        "passed": passed,
        "next_action": "unlock_one_time_2025_final_test" if passed else "stop_v9_keep_2025_locked",
    }


def publish_result(source: Path, destination: Path, result: dict[str, Any]) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    benchmark = destination / "benchmark.json"
    shutil.copyfile(source, benchmark)
    decision = result["decision"]
    metrics = result["metrics"]
    lines = [
        "# V9-A one-time 2024 confirmation",
        "",
        "The candidate, checkpoint, metrics, and gates were frozen before 2024 was read. ",
        "The locked 2025 test remains unread unless this confirmation passes.",
        "",
        f"- Confirmation passed: `{decision['passed']}`",
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
        files.append({"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_json(
        destination / "manifest.json",
        {"schema_version": 1, "confirmation_years_read": [2024], "test_years_read": [], "files": files},
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v9_2024_confirmation.yaml")
    parser.add_argument("--mode", choices=("preflight", "confirm"), default="preflight")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--acknowledge-one-time-2024", action="store_true")
    args = parser.parse_args()

    config = read_yaml(resolve(args.config))
    validate_protocol(config)
    freeze_evidence = validate_freeze_evidence(config)
    if args.mode == "confirm" and not args.acknowledge_one_time_2024:
        raise ValueError("2024 is locked; pass --acknowledge-one-time-2024 explicitly")
    years = list(
        config["preflight_years"] if args.mode == "preflight" else config["confirmation_years"]
    )
    output = resolve(config["evaluation"]["output_dir"])
    result_path = output / "benchmark.json"
    marker = output / "CONFIRMATION_STARTED.json"
    if args.mode == "confirm":
        if marker.exists() or result_path.exists():
            raise FileExistsError("Refusing to repeat the one-time 2024 confirmation")

    data_config = load_config(resolve(config["data_config"]))
    data_config["data"]["window_stride_hours"] = int(config["window_stride_hours"])
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    # Validate both frozen checkpoints before creating the irreversible 2024
    # start marker. Dataset construction below is the first confirmation-year read.
    models, checkpoint_metadata = load_models(config, data_config, device, 390)
    if args.mode == "confirm":
        write_json(
            marker,
            {"protocol": config["protocol"], "confirmation_years_read": [2024], "test_years_read": []},
        )
    dataset = make_dataset(data_config, years, augment=False)
    if len(dataset.station_ids) != 390:
        raise ValueError("V9 confirmation requires the frozen 390-station contract")
    variables = list(data_config["data"]["target_variables"])
    if variables != list(TARGETS):
        raise ValueError(f"V9 confirmation target order must be {TARGETS}")
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
        v9_prediction, v9_current = v9_forward_components(models["v9_a"], first, static)
        v7_prediction = v7_forward(models["v7_b"], first, static)
    expected_shape = [int(first["target"].shape[0]), 6, 5, 31, 33]
    if list(v9_prediction.shape) != expected_shape or list(v7_prediction.shape) != expected_shape:
        raise ValueError("Frozen model output shape mismatch")
    preflight = {
        "schema_version": 1,
        "mode": args.mode,
        "device": str(device),
        "years_instantiated": years,
        "confirmation_years_read": [] if args.mode == "preflight" else [2024],
        "test_years_read": [],
        "samples": len(dataset),
        "batches": len(loader),
        "station_count": len(dataset.station_ids),
        "variables": variables,
        "v9_output_shape": list(v9_prediction.shape),
        "v7_output_shape": list(v7_prediction.shape),
        "v9_current_reconstruction_shape": list(v9_current.shape),
        "freeze_evidence": freeze_evidence,
        "checkpoint_metadata": checkpoint_metadata,
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / ("preflight.json" if args.mode == "preflight" else "confirmation_preflight.json"), preflight)
    print(
        f"Preflight OK: years={years} samples={len(dataset):,} batches={len(loader):,} test_years_read=[]",
        flush=True,
    )
    if args.mode == "preflight":
        dataset.close()
        return 0

    metric_accumulators = {
        name: ForecastMetricAccumulator(tuple(variables), forecast_hours)
        for name in FORECAST_NAMES
    }
    reconstruction_accumulators = {
        name: ForecastMetricAccumulator(tuple(variables), 1)
        for name in RECONSTRUCTION_NAMES
    }
    processed = 0
    started = time.perf_counter()
    with torch.no_grad():
        for batch_index, raw in enumerate(loader):
            batch = move(raw, device)
            target = denormalize_channels(batch["target"], variables, normalization)
            v9_normalized, v9_current_normalized = v9_forward_components(models["v9_a"], batch, static)
            v7_normalized = v7_forward(models["v7_b"], batch, static)
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
            v7_current_normalized = reconstruction_forward(models["v7_b"], current_batch, static)[:, 0]
            v7_current = denormalize_channels(v7_current_normalized[:, None], variables, normalization)[:, 0]
            v9_current = denormalize_channels(v9_current_normalized[:, None], variables, normalization)[:, 0]
            predictions["sparse_reconstruction_persistence"] = dense_grid_persistence(v7_current, forecast_hours)
            current_indices = raw["start_index"].numpy() + history_hours - 1
            dense_normalized = torch.from_numpy(
                np.asarray(dataset.target_grids[current_indices], dtype=np.float32).copy()
            ).to(device)
            dense_current = denormalize_channels(dense_normalized[:, None], variables, normalization)[:, 0]
            predictions["dense_era5_persistence"] = dense_grid_persistence(dense_current, forecast_hours)
            for name, prediction in predictions.items():
                metric_accumulators[name].update(prediction, target, land_mask)
            current_target = denormalize_channels(batch["current_target"][:, None], variables, normalization)
            reconstruction_accumulators["v9_a"].update(v9_current[:, None], current_target, land_mask)
            reconstruction_accumulators["v7_b"].update(v7_current[:, None], current_target, land_mask)
            processed += int(target.shape[0])
            completed = batch_index + 1
            every = int(config["evaluation"]["progress_every_batches"])
            if every > 0 and (completed % every == 0 or completed == len(loader)):
                elapsed = time.perf_counter() - started
                eta = elapsed / completed * max(0, len(loader) - completed)
                print(
                    f"  one-time 2024 confirmation {completed:,}/{len(loader):,} "
                    f"elapsed={elapsed / 60:.1f}m ETA={eta / 60:.1f}m",
                    flush=True,
                )

    metrics = {name: accumulator.compute() for name, accumulator in metric_accumulators.items()}
    reconstruction_metrics = {
        name: accumulator.compute() for name, accumulator in reconstruction_accumulators.items()
    }
    skills = {
        "v9_a_vs_v7_b": skill(metrics["v9_a"], metrics["v7_b"]),
        "v9_a_vs_sparse_reconstruction_persistence": skill(
            metrics["v9_a"], metrics["sparse_reconstruction_persistence"]
        ),
        "v9_a_vs_dense_era5_persistence": skill(metrics["v9_a"], metrics["dense_era5_persistence"]),
    }
    decision = confirmation_decision(skills, reconstruction_metrics, config)
    result = {
        "schema_version": 1,
        "scientific_status": "one_time_2024_confirmation",
        "contract": {
            "confirmation_years_read": [2024],
            "test_years_read": [],
            "window_stride_hours": int(config["window_stride_hours"]),
            "history_hours": history_hours,
            "forecast_hours": forecast_hours,
            "station_count": len(dataset.station_ids),
            "target_variables": variables,
            "samples": processed,
            "batches": len(loader),
            "no_post_confirmation_tuning": True,
        },
        "freeze_evidence": freeze_evidence,
        "checkpoint_metadata": checkpoint_metadata,
        "baseline_definitions": config["baselines"],
        "metrics": metrics,
        "reconstruction_metrics": reconstruction_metrics,
        "rmse_skills": skills,
        "decision": decision,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(result_path, result)
    publish_result(result_path, resolve(config["evaluation"]["publish_dir"]), result)
    print(
        f"2024 confirmation complete: passed={decision['passed']} "
        f"next={decision['next_action']} output={result_path}",
        flush=True,
    )
    dataset.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
