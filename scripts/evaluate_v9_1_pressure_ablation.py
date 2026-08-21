#!/usr/bin/env python3
"""Evaluate the frozen five-vs-six-variable V9.1 pressure ablation."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_v9_2024_confirmation as v9_common  # noqa: E402
from check_v7 import load_config, make_dataset, move, resolve  # noqa: E402
from evaluate_v8_2016_benchmark import (  # noqa: E402
    derive_event_thresholds,
    make_event_accumulators,
    update_events,
)
from stormengine_dl.data import NormalizationStats, StaticFields  # noqa: E402
from stormengine_dl.models.v9 import StormEngineV9ForecastModel  # noqa: E402
from stormengine_dl.runtime import denormalize_channels  # noqa: E402
from stormengine_dl.training import ForecastMetricAccumulator  # noqa: E402
from train_v9_output_form import forward_components, make_model, require_development_protocol  # noqa: E402


MODEL_NAMES = (
    "control_seed42",
    "pressure_seed42",
    "control_seed43",
    "pressure_seed43",
)


def checkpoint(path: Path, config: dict[str, Any], device: torch.device) -> tuple[StormEngineV9ForecastModel, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing V9.1 checkpoint: {path}")
    saved = torch.load(path, map_location=device, weights_only=False)
    model = make_model(config, "autoregressive", "field").to(device)
    expected_inputs = list(config["data"]["input_variables"])
    contract = saved.get("model_contract", {})
    if contract.get("development_protocol") != "v9.1-pressure-ablation-2015-2019":
        raise ValueError(f"Checkpoint has the wrong V9.1 protocol: {path}")
    if contract.get("input_variables") != expected_inputs:
        raise ValueError(f"Checkpoint input contract does not match {expected_inputs}: {path}")
    if contract.get("train_years") != [2015, 2016, 2017]:
        raise ValueError("V9.1 checkpoint did not train on the frozen 2015--2017 split")
    if contract.get("validation_years") != [2018] or contract.get("locked_test_years") != [2019]:
        raise ValueError("V9.1 checkpoint has the wrong validation/test contract")
    model.load_state_dict(saved["model_state_dict"], strict=True)
    model.eval()
    return model, {
        "path": v9_common.portable(path),
        "sha256": v9_common.sha256_file(path),
        "checkpoint_contract": contract,
        "best_validation_loss": float(saved["best_validation_loss"]),
        "best_epoch": int(min(saved["history"], key=lambda row: row["validation_loss"])["epoch"]),
    }


def acceptance(
    skills: dict[str, Any], events: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    rows = []
    for seed in (42, 43):
        key = f"pressure_seed{seed}_vs_control_seed{seed}"
        skill = skills[key]
        full_msl = float(skill["aggregate"]["full"]["msl"]["skill"])
        sea_msl = float(skill["aggregate"]["sea"]["msl"]["skill"])
        non_msl = [
            float(skill["aggregate"]["sea"][variable]["skill"])
            for variable in ("u10", "v10", "t2m", "tp")
        ]
        control_event = events[f"control_seed{seed}"]["low_msl_q05"]["aggregate"]
        pressure_event = events[f"pressure_seed{seed}"]["low_msl_q05"]["aggregate"]
        control_rmse = control_event["event_conditioned_rmse"]
        pressure_rmse = pressure_event["event_conditioned_rmse"]
        event_improved = (
            control_rmse is not None
            and pressure_rmse is not None
            and float(pressure_rmse) < float(control_rmse)
        )
        rows.append(
            {
                "seed": seed,
                "full_msl_rmse_skill": full_msl,
                "sea_msl_rmse_skill": sea_msl,
                "mean_non_msl_sea_rmse_skill": sum(non_msl) / len(non_msl),
                "low_msl_event_rmse_control": control_rmse,
                "low_msl_event_rmse_pressure": pressure_rmse,
                "low_msl_event_rmse_improved": event_improved,
                "low_msl_csi_control": control_event["csi"],
                "low_msl_csi_pressure": pressure_event["csi"],
            }
        )
    rules = config["acceptance"]
    required_true = (
        "require_positive_full_msl_skill_each_seed",
        "require_positive_sea_msl_skill_each_seed",
        "require_lower_low_msl_event_rmse_each_seed",
    )
    if any(rules.get(name) is not True for name in required_true):
        raise ValueError("The frozen V9.1 pressure acceptance rules were changed")
    passed = all(row["full_msl_rmse_skill"] > 0 for row in rows)
    passed &= all(row["sea_msl_rmse_skill"] > 0 for row in rows)
    mean_non_msl = sum(row["mean_non_msl_sea_rmse_skill"] for row in rows) / len(rows)
    passed &= mean_non_msl >= -float(rules["maximum_mean_non_msl_sea_rmse_degradation"])
    passed &= all(bool(row["low_msl_event_rmse_improved"]) for row in rows)
    return {
        "rules_frozen_before_2018": True,
        "seed_results": rows,
        "two_seed_mean_full_msl_rmse_skill": sum(row["full_msl_rmse_skill"] for row in rows) / 2,
        "two_seed_mean_sea_msl_rmse_skill": sum(row["sea_msl_rmse_skill"] for row in rows) / 2,
        "two_seed_mean_non_msl_sea_rmse_skill": mean_non_msl,
        "passed": bool(passed),
        "next_action": "unlock_one_time_2019" if passed else "retain_five_variable_control",
    }


def validate_test_unlock(config: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    path = resolve(config["evaluation"]["validation_publish_dir"]) / "benchmark.json"
    if not path.is_file():
        raise FileNotFoundError("The frozen 2018 validation result does not exist")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("scientific_status") != "frozen_2018_pressure_validation":
        raise ValueError("2019 is not unlocked by a full frozen 2018 validation")
    if not bool(value.get("acceptance", {}).get("passed")):
        raise ValueError("V9.1 pressure did not pass the 2018 acceptance gate")
    expected = value.get("checkpoint_metadata", {})
    for name in MODEL_NAMES:
        if expected.get(name, {}).get("sha256") != metadata[name]["sha256"]:
            raise ValueError(f"Checkpoint {name} changed after 2018 validation")
    return {
        "path": v9_common.portable(path),
        "sha256": v9_common.sha256_file(path),
        "validation_passed": True,
    }


def publish(source: Path, destination: Path, result: dict[str, Any]) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / "benchmark.json"
    shutil.copyfile(source, output)
    decision = result["acceptance"]
    lines = [
        f"# V9.1 pressure ablation — {result['scientific_status']}",
        "",
        f"- Years read: `{result['contract']['evaluation_years']}`",
        f"- Two-seed mean full MSL skill: `{100 * decision['two_seed_mean_full_msl_rmse_skill']:.2f}%`",
        f"- Two-seed mean sea MSL skill: `{100 * decision['two_seed_mean_sea_msl_rmse_skill']:.2f}%`",
        f"- Passed: `{decision['passed']}`",
        "",
        "The only controlled input difference is the deployment-compatible MSL channel. "
        "Event thresholds come exclusively from 2015--2017.",
    ]
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
            "evaluation_years": result["contract"]["evaluation_years"],
            "test_years_read": result["contract"]["test_years_read"],
            "files": files,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("validation", "test"))
    parser.add_argument("--config", default="configs/v9_1_pressure_evaluation.yaml")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--acknowledge-one-time-2019", action="store_true")
    args = parser.parse_args()

    config = v9_common.read_yaml(resolve(args.config))
    if list(config["validation_years"]) != [2018] or list(config["test_years"]) != [2019]:
        raise ValueError("V9.1 evaluation chronology must remain 2018 -> 2019")
    if args.mode == "test" and not args.acknowledge_one_time_2019:
        raise ValueError("Pass --acknowledge-one-time-2019 to read the locked test year")
    years = list(config["validation_years"] if args.mode == "validation" else config["test_years"])
    output_dir = resolve(
        config["evaluation"][
            "validation_output_dir" if args.mode == "validation" else "test_output_dir"
        ]
    )
    marker = output_dir / "FINAL_TEST_STARTED.json"
    if args.mode == "test" and args.max_batches is None:
        if marker.exists() or (output_dir / "benchmark.json").exists():
            raise FileExistsError("Refusing to repeat the one-time 2019 pressure test")
    control_config = load_config(resolve(config["control_config"]))
    pressure_config = load_config(resolve(config["pressure_config"]))
    require_development_protocol(control_config)
    require_development_protocol(pressure_config)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    models: dict[str, StormEngineV9ForecastModel] = {}
    metadata: dict[str, Any] = {}
    for name in MODEL_NAMES:
        model_config = pressure_config if name.startswith("pressure") else control_config
        models[name], metadata[name] = checkpoint(resolve(config["checkpoints"][name]), model_config, device)
    unlock = None
    if args.mode == "test":
        unlock = validate_test_unlock(config, metadata)
        if args.max_batches is None:
            output_dir.mkdir(parents=True, exist_ok=True)
            v9_common.write_json(
                marker,
                {"test_years_read": [2019], "checkpoint_metadata": metadata},
            )

    control = make_dataset(control_config, years, augment=False)
    pressure = make_dataset(pressure_config, years, augment=False)
    if control.station_ids != pressure.station_ids or len(control.station_ids) != 390:
        raise ValueError("V9.1 paired datasets must use the same 390 coordinates")
    if not torch.equal(torch.from_numpy(control.window_starts), torch.from_numpy(pressure.window_starts)):
        raise ValueError("V9.1 paired datasets do not contain identical windows")
    if pressure.variable_capability_counts != {"msl": 164}:
        raise ValueError(
            "V9.1 pressure evaluation requires exactly 13 physical plus 151 marine "
            f"MSL-capable positions, got {pressure.variable_capability_counts}"
        )
    variables = list(control_config["data"]["target_variables"])
    normalization = NormalizationStats.load(resolve(config["normalization_stats"]))
    static_data = StaticFields.load(resolve(control_config["data"]["static_fields"]))
    static = static_data.as_tensor().unsqueeze(0).to(device)
    land_mask = torch.from_numpy(static_data.land_sea_mask)
    sea_mask = land_mask < 0.5
    forecast_hours = int(control_config["data"]["forecast_hours"])
    thresholds = derive_event_thresholds(
        control,
        variables,
        normalization,
        static_data.land_sea_mask,
        list(config["threshold_years"]),
        config["events"],
    )
    metrics = {name: ForecastMetricAccumulator(tuple(variables), forecast_hours) for name in MODEL_NAMES}
    events = {name: make_event_accumulators(forecast_hours, thresholds) for name in MODEL_NAMES}
    options = {
        "batch_size": int(config["evaluation"]["batch_size"]),
        "shuffle": False,
        "num_workers": 0,
        "pin_memory": device.type == "cuda",
    }
    control_loader = DataLoader(control, **options)
    pressure_loader = DataLoader(pressure, **options)
    if len(control_loader) != len(pressure_loader):
        raise ValueError("V9.1 paired loaders have different lengths")
    expected_batches = min(len(control_loader), args.max_batches) if args.max_batches else len(control_loader)
    processed = 0
    started = time.perf_counter()
    with torch.no_grad():
        for batch_index, (control_raw, pressure_raw) in enumerate(zip(control_loader, pressure_loader)):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            if not torch.equal(control_raw["start_index"], pressure_raw["start_index"]):
                raise ValueError("Paired V9.1 batches are not time aligned")
            control_batch = move(control_raw, device)
            pressure_batch = move(pressure_raw, device)
            target = denormalize_channels(control_batch["target"], variables, normalization)
            if not torch.equal(control_batch["target"], pressure_batch["target"]):
                raise ValueError("Paired V9.1 target tensors differ")
            for name in MODEL_NAMES:
                batch = pressure_batch if name.startswith("pressure") else control_batch
                prediction, _ = forward_components(models[name], batch, static)
                physical = denormalize_channels(prediction, variables, normalization)
                metrics[name].update(physical, target, land_mask)
                update_events(events[name], physical, target, sea_mask, variables)
            processed += int(target.shape[0])
            completed = batch_index + 1
            every = int(config["evaluation"]["progress_every_batches"])
            if every and (completed % every == 0 or completed == expected_batches):
                print(f"  V9.1 {args.mode} {completed}/{expected_batches}", flush=True)

    metric_values = {name: value.compute() for name, value in metrics.items()}
    event_values = {
        name: {event: accumulator.compute() for event, accumulator in value.items()}
        for name, value in events.items()
    }
    skills = {
        f"pressure_seed{seed}_vs_control_seed{seed}": v9_common.skill(
            metric_values[f"pressure_seed{seed}"], metric_values[f"control_seed{seed}"]
        )
        for seed in (42, 43)
    }
    decision = acceptance(skills, event_values, config)
    bounded = args.max_batches is not None and args.max_batches < len(control_loader)
    status = (
        "bounded_pressure_pipeline_check"
        if bounded
        else "frozen_2018_pressure_validation"
        if args.mode == "validation"
        else "one_time_2019_pressure_final_test"
    )
    result = {
        "schema_version": 1,
        "scientific_status": status,
        "contract": {
            "training_years": [2015, 2016, 2017],
            "threshold_years": list(config["threshold_years"]),
            "evaluation_years": years,
            "test_years_read": [2019] if args.mode == "test" else [],
            "window_stride_hours": int(config["window_stride_hours"]),
            "samples": processed,
            "batches": expected_batches,
            "control_inputs": list(control_config["data"]["input_variables"]),
            "pressure_inputs": list(pressure_config["data"]["input_variables"]),
            "pressure_capability_counts": pressure.variable_capability_counts,
            "no_post_test_tuning": args.mode == "test",
        },
        "unlock_evidence": unlock,
        "checkpoint_metadata": metadata,
        "event_thresholds": thresholds,
        "metrics": metric_values,
        "rmse_skills": skills,
        "event_metrics": event_values,
        "acceptance": decision,
        "elapsed_seconds": time.perf_counter() - started,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "benchmark.json"
    v9_common.write_json(output, result)
    if not bounded:
        publish_dir = config["evaluation"][
            "validation_publish_dir" if args.mode == "validation" else "test_publish_dir"
        ]
        publish(output, resolve(publish_dir), result)
    print(f"V9.1 {args.mode} complete: passed={decision['passed']} output={output}", flush=True)
    control.close(); pressure.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
