#!/usr/bin/env python3
"""Frozen 2026 evaluation against input-compatible reconstruction persistence.

This script is deliberately evaluation-only. It reuses the exact operational
inputs, target indexing, normalization, masks, checkpoints, and dense ERA5T
persistence definition from ``evaluate_v9_2_final16_operational_era5t.py``.
The new baseline decodes the final encoder's last historical latent state with
the checkpoint's validated reconstruction decoder and repeats that field over
leads +1...+6. The forecast processor is not called by this baseline path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_v9_2_final16_operational_era5t as operational  # noqa: E402
from check_v7 import load_config, resolve  # noqa: E402
from train_v9_output_form import forward, forward_components  # noqa: E402
from stormengine_dl.baselines import dense_grid_persistence  # noqa: E402
from stormengine_dl.data import (  # noqa: E402
    NormalizationStats,
    StaticFields,
    load_era5_target_grid,
    load_v7_b_input,
)
from stormengine_dl.data.operational_adapter import load_fixed_registry  # noqa: E402
from stormengine_dl.runtime import denormalize_channels  # noqa: E402
from stormengine_dl.training import ForecastMetricAccumulator  # noqa: E402


TARGETS = operational.TARGETS
INPUTS = operational.INPUTS
METHODS = (
    "v9_2_final16",
    "input_compatible_reconstruction_persistence",
    "dense_era5t_persistence",
    "v9_1_frozen",
)
EXPECTED_FINAL_SHA256 = "ea0afde51397bd88bc6aeca45e452cabadfd956821112c6828eb826fdc03e86f"
EXPECTED_SOURCE_SHA256 = "eca9d3d82ed7bae0b7760bc026a31b6ff684c2cffb4bb97474be7011ef0b2bd0"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def portable(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def require_new_output_directory(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing result directory: {path}")


def validate_checkpoint(
    path: Path,
    *,
    expected_sha256: str,
    expected_station_count: int,
) -> dict[str, Any]:
    actual_sha256 = operational.sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Frozen checkpoint SHA-256 mismatch for {path}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    contract = checkpoint.get("model_contract")
    if not isinstance(contract, dict):
        raise ValueError(f"Checkpoint is missing model_contract: {path}")
    expected = {
        "version": "stormengine-v9-output-form-v1",
        "temporal_mode": "autoregressive",
        "output_mode": "field",
        "history_hours": 12,
        "forecast_hours": 6,
        "input_variables": list(INPUTS),
        "target_variables": list(TARGETS),
        "station_count": expected_station_count,
    }
    mismatches = {
        key: {"expected": value, "actual": contract.get(key)}
        for key, value in expected.items()
        if contract.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Frozen checkpoint contract mismatch: {mismatches}")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("Checkpoint is missing model_state_dict")
    reconstruction_keys = sorted(
        key for key in state if key.startswith("reconstruction_decoder.")
    )
    if len(reconstruction_keys) != 6:
        raise ValueError(
            "Expected six reconstruction_decoder tensors in the frozen checkpoint, "
            f"found {len(reconstruction_keys)}"
        )
    if not all(
        bool(torch.isfinite(state[key]).all()) for key in reconstruction_keys
    ):
        raise ValueError("Frozen reconstruction_decoder contains NaN or Inf")
    return {
        "path": portable(path),
        "sha256": actual_sha256,
        "strict_load": True,
        "contract": contract,
        "reconstruction_decoder_tensor_count": len(reconstruction_keys),
        "reconstruction_decoder_tensor_names": reconstruction_keys,
        "reconstruction_decoder_finite": True,
    }


def reconstruct_current_without_processor(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    static: torch.Tensor,
) -> torch.Tensor:
    """Use the existing V9 reconstruction branch without calling Processor."""
    batch_size = int(batch["point_values"].shape[0])
    encoded = model.encoder(
        batch["point_values"],
        batch["point_coords"],
        batch["value_mask"],
        batch["observation_age"],
        batch["point_static"],
    )
    current = model.reconstruction_decoder(
        encoded[:, -1:], static.expand(batch_size, -1, -1, -1)
    )
    if current.shape[1] != 1:
        raise ValueError(f"Expected one current reconstruction, got {tuple(current.shape)}")
    return current[:, 0]


def rmse_skill(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    def compare(
        candidate_regions: dict[str, Any], baseline_regions: dict[str, Any]
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for region in ("full", "land", "sea"):
            result[region] = {}
            for variable in TARGETS:
                candidate_rmse = float(candidate_regions[region][variable]["rmse"])
                baseline_rmse = float(baseline_regions[region][variable]["rmse"])
                if baseline_rmse <= 0:
                    raise ValueError(
                        f"Cannot compute RMSE skill against non-positive {region}/{variable} baseline"
                    )
                result[region][variable] = {
                    "candidate_rmse": candidate_rmse,
                    "baseline_rmse": baseline_rmse,
                    "skill": 1.0 - candidate_rmse / baseline_rmse,
                }
        return result

    return {
        "definition": "1 - RMSE_v9_2_final16 / RMSE_baseline; positive means V9.2 is better",
        "aggregate": compare(candidate["aggregate"], baseline["aggregate"]),
        "by_lead_hour": {
            lead: compare(regions, baseline["by_lead_hour"][lead])
            for lead, regions in candidate["by_lead_hour"].items()
        },
    }


def validate_timeline(
    times: np.ndarray,
    starts: np.ndarray,
    history: int,
    forecast_hours: int,
) -> dict[str, Any]:
    times = times.astype("datetime64[ns]")
    if len(times) < history + forecast_hours:
        raise ValueError("Input timeline is too short")
    deltas = np.diff(times).astype("timedelta64[s]").astype(np.int64)
    if not np.all(deltas == 3600):
        raise ValueError("Operational input timeline is not strictly hourly")
    origins = times[starts + history - 1]
    future = np.stack(
        [times[start + history : start + history + forecast_hours] for start in starts]
    )
    expected = origins[:, None] + np.arange(1, forecast_hours + 1)[None] * np.timedelta64(
        1, "h"
    )
    if not np.array_equal(future, expected):
        raise ValueError("Forecast leads +1...+6 are temporally misaligned")
    if not np.all(times[starts + history - 1] < future[:, 0]):
        raise ValueError("Current reconstruction would not precede all targets")
    payload = np.concatenate((origins[:, None], future), axis=1).astype("datetime64[ns]")
    return {
        "strictly_hourly_input": True,
        "first_forecast_origin": str(origins[0]),
        "last_forecast_origin": str(origins[-1]),
        "first_target_time": str(future[0, 0]),
        "last_target_time": str(future[-1, -1]),
        "forecast_origin_and_target_timestamp_sha256": sha256_bytes(payload.tobytes()),
        "current_reconstruction_uses_only_origin_or_earlier": True,
        "lead_alignment": "+1 through +6 hours from the last historical input",
    }


def exact_comparison(
    new_metrics: dict[str, Any], old_metrics_path: Path
) -> dict[str, Any]:
    old = json.loads(old_metrics_path.read_text(encoding="utf-8"))
    mismatches: list[str] = []

    def walk(new: Any, previous: Any, path: str) -> None:
        if isinstance(new, dict) and isinstance(previous, dict):
            if set(new) != set(previous):
                mismatches.append(f"{path}: keys differ")
                return
            for key in sorted(new):
                walk(new[key], previous[key], f"{path}.{key}")
        elif isinstance(new, list) and isinstance(previous, list):
            if len(new) != len(previous):
                mismatches.append(f"{path}: list lengths differ")
                return
            for index, (left, right) in enumerate(zip(new, previous)):
                walk(left, right, f"{path}[{index}]")
        elif new != previous:
            mismatches.append(f"{path}: {new!r} != {previous!r}")

    for name in ("v9_2_final16", "v9_1_frozen", "dense_era5t_persistence"):
        if name not in old.get("field_metrics", {}):
            mismatches.append(f"old field_metrics is missing {name}")
            continue
        walk(new_metrics[name], old["field_metrics"][name], f"field_metrics.{name}")
    old_windows = int(old.get("contract", {}).get("forecast_windows", -1))
    if old_windows != 152:
        mismatches.append(f"old contract forecast_windows is {old_windows}, expected 152")
    result = {
        "old_metrics_path": portable(old_metrics_path),
        "old_metrics_sha256": operational.sha256(old_metrics_path),
        "compared_methods": [
            "v9_2_final16",
            "v9_1_frozen",
            "dense_era5t_persistence",
        ],
        "comparison": "recursive exact equality of every aggregate and per-lead MAE/RMSE value",
        "exactly_identical": not mismatches,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:20],
    }
    if mismatches:
        raise ValueError(f"New evaluation does not reproduce old metrics exactly: {result}")
    return result


def write_metric_csvs(output: Path, metrics: dict[str, Any]) -> None:
    with (output / "aggregate_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("method", "region", "variable", "mae", "rmse", "sample_windows"))
        for method in METHODS:
            for region in ("full", "land", "sea"):
                for variable in TARGETS:
                    item = metrics[method]["aggregate"][region][variable]
                    writer.writerow((method, region, variable, item["mae"], item["rmse"], 152))
    with (output / "per_lead_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ("method", "lead_hour", "region", "variable", "mae", "rmse", "sample_windows")
        )
        for method in METHODS:
            for lead in range(1, 7):
                for region in ("full", "land", "sea"):
                    for variable in TARGETS:
                        item = metrics[method]["by_lead_hour"][str(lead)][region][variable]
                        writer.writerow(
                            (method, lead, region, variable, item["mae"], item["rmse"], 152)
                        )


def write_skill_csv(path: Path, skill: dict[str, Any], baseline: str) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "baseline",
                "scope",
                "lead_hour",
                "region",
                "variable",
                "v9_2_rmse",
                "baseline_rmse",
                "rmse_skill",
                "rmse_skill_percent",
            )
        )
        for region in ("full", "land", "sea"):
            for variable in TARGETS:
                item = skill["aggregate"][region][variable]
                writer.writerow(
                    (
                        baseline,
                        "aggregate",
                        "",
                        region,
                        variable,
                        item["candidate_rmse"],
                        item["baseline_rmse"],
                        item["skill"],
                        100.0 * item["skill"],
                    )
                )
        for lead in range(1, 7):
            for region in ("full", "land", "sea"):
                for variable in TARGETS:
                    item = skill["by_lead_hour"][str(lead)][region][variable]
                    writer.writerow(
                        (
                            baseline,
                            "per_lead",
                            lead,
                            region,
                            variable,
                            item["candidate_rmse"],
                            item["baseline_rmse"],
                            item["skill"],
                            100.0 * item["skill"],
                        )
                    )


def plot_skill_figure(output: Path, skill: dict[str, Any]) -> None:
    labels = ("MSL", "u10", "v10", "T2m", "TP")
    colors = ("#355C7D", "#2A9D8F", "#4C78A8", "#E09F3E", "#8E6C8A")
    full = [100.0 * skill["aggregate"]["full"][name]["skill"] for name in TARGETS]
    sea = [100.0 * skill["aggregate"]["sea"][name]["skill"] for name in TARGETS]
    leads = np.arange(1, 7)
    u10 = [
        100.0 * skill["by_lead_hour"][str(lead)]["sea"]["u10"]["skill"]
        for lead in leads
    ]
    v10 = [
        100.0 * skill["by_lead_hour"][str(lead)]["sea"]["v10"]["skill"]
        for lead in leads
    ]
    plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65), constrained_layout=True)
    x = np.arange(len(TARGETS))
    for axis, values, title in (
        (axes[0], full, "A  Full-domain skill"),
        (axes[1], sea, "B  Sea-domain skill"),
    ):
        axis.bar(x, values, color=colors, width=0.72)
        axis.set_xticks(x, labels)
        axis.axhline(0, color="#333333", linewidth=0.8)
        axis.grid(axis="y", color="#D7DEE5", linewidth=0.6, alpha=0.8)
        axis.set_axisbelow(True)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_ylabel("RMSE skill (%)")
    axes[2].plot(leads, u10, marker="o", color="#2A9D8F", label="u10", linewidth=1.6)
    axes[2].plot(leads, v10, marker="s", color="#4C78A8", label="v10", linewidth=1.6)
    axes[2].axhline(0, color="#333333", linewidth=0.8)
    axes[2].grid(color="#D7DEE5", linewidth=0.6, alpha=0.8)
    axes[2].set_axisbelow(True)
    axes[2].set_xticks(leads)
    axes[2].set_xlabel("Lead time (h)")
    axes[2].set_ylabel("RMSE skill (%)")
    axes[2].set_title("C  Sea-wind skill by lead", loc="left", fontweight="bold")
    axes[2].legend(frameon=False, ncol=2, loc="best")
    fig.suptitle(
        "V9.2 Final16 vs input-compatible reconstruction persistence",
        fontsize=11,
        fontweight="bold",
    )
    fig.savefig(output / "skill_vs_input_compatible_persistence.png", dpi=300, facecolor="white")
    fig.savefig(output / "skill_vs_input_compatible_persistence.pdf", facecolor="white")
    plt.close(fig)


def write_readme(output: Path, result: dict[str, Any]) -> None:
    skills = result["rmse_skills"]["v9_2_vs_input_compatible_reconstruction_persistence"]
    lines = [
        "# V9.2 Final16 input-compatible persistence evaluation",
        "",
        "This is a supplemental evaluation of frozen checkpoints. No model was retrained, "
        "no parameter or threshold was changed, and no 2026 result was used for tuning.",
        "",
        "## Fair persistence baseline",
        "",
        "For each of the same 152 operational windows, the baseline receives the same 12 hours "
        "of DPC/MeteoHub physical observations and Open-Meteo marine support, including the same "
        "coordinate registry, variable order, normalization, per-variable masks, observation ages, "
        "station static features, and source type. The frozen Final16 encoder produces the historical "
        "latent sequence. Its last latent state is decoded by the checkpoint's existing, validated "
        "`reconstruction_decoder`; the Processor is not called. The reconstructed current 31 x 33 "
        "field is copied unchanged to leads +1 through +6.",
        "",
        "Dense ERA5T persistence remains a useful stronger-information diagnostic because it receives "
        "the complete ERA5T grid at the forecast origin. It is not input-compatible with the operational "
        "model. The reconstruction-persistence baseline is therefore the fair test of whether the "
        "Processor adds value beyond holding the model-reconstructed current state fixed.",
        "",
        "## Scope",
        "",
        "The period is 2026-08-01 through 2026-08-08, with 152 windows, 12 historical hours, "
        "and +1...+6 h forecasts verified against the same ERA5T 31 x 33 targets as the published "
        "operational evaluation. No observed case reached the frozen 30 mm/6 h heavy-rain or "
        "15 m/s strong-wind threshold, so this experiment evaluates ordinary continuous fields only; "
        "it cannot support an extreme-event skill claim.",
        "",
        "## Aggregate RMSE skill against the fair baseline",
        "",
        "Positive skill means V9.2 Final16 has lower RMSE than input-compatible reconstruction persistence.",
        "",
        "| region | MSL | u10 | v10 | T2m | TP |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for region in ("full", "land", "sea"):
        values = [100.0 * skills["aggregate"][region][name]["skill"] for name in TARGETS]
        lines.append(
            f"| {region} | " + " | ".join(f"{value:.2f}%" for value in values) + " |"
        )
    lines.extend(
        [
            "",
            "## Integrity",
            "",
            f"- Smoke test passed: `{result['integrity_checks']['smoke_contract_verified']}`",
            f"- Full window count: `{result['contract']['forecast_windows']}`",
            f"- All outputs finite: `{result['integrity_checks']['all_predictions_and_targets_finite']}`",
            f"- +1...+6 h alignment passed: `{result['integrity_checks']['timeline']['lead_alignment']}`",
            f"- Original V9.2, V9.1, and dense-persistence metrics reproduced exactly: "
            f"`{result['integrity_checks']['old_evaluation_comparison']['exactly_identical']}`",
            "- TP semantics are unchanged: hourly ERA5T TP is verified at each lead; six-hour event "
            "  diagnostics, when referenced, use `sum(max(tp_hourly_mm, 0))` over +1...+6.",
            "",
            "The JSON and CSV files contain aggregate and lead-specific MAE, RMSE, and RMSE skill "
            "for full, land, and sea domains.",
        ]
    )
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(output: Path, result: dict[str, Any], args: argparse.Namespace) -> None:
    files = []
    for path in sorted(output.iterdir()):
        if path.name == "run_manifest.json" or not path.is_file():
            continue
        files.append(
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": operational.sha256(path),
            }
        )
    manifest = {
        "schema_version": 1,
        "evaluation_only": True,
        "training_performed": False,
        "parameter_tuning_performed": False,
        "frozen_result_modified": False,
        "command_arguments": vars(args),
        "contract": result["contract"],
        "checkpoints": result["checkpoints"],
        "input_files": result["input_files"],
        "integrity_checks": result["integrity_checks"],
        "files": files,
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v9_2_event_aware_final16.yaml")
    parser.add_argument(
        "--dpc-input",
        default="data_external/meteohub/processed/20260801_20260808/official_hourly_physical.npz",
    )
    parser.add_argument(
        "--dpc-msl",
        default="data_external/meteohub/processed/20260801_20260808/official_hourly_physical_msl.npz",
    )
    parser.add_argument(
        "--marine-input",
        default="data_external/open_meteo/processed/20260801_20260808/icon2i_hourly_marine.npz",
    )
    parser.add_argument(
        "--marine-msl",
        default="data_external/open_meteo/processed/20260801_20260808/icon2i_hourly_marine_pressure.npz",
    )
    parser.add_argument("--era5t-instant", required=True)
    parser.add_argument("--era5t-accum", required=True)
    parser.add_argument(
        "--checkpoint", default="artifacts/v9_2_event_aware_final16/seed_42/final.pt"
    )
    parser.add_argument(
        "--source-checkpoint",
        default="artifacts/v9_1_pressure_ablation/pressure_6var/pressure_6var/seed_42/best.pt",
    )
    parser.add_argument(
        "--old-metrics",
        default="artifacts/v9_2_final16_operational_era5t_20260801_20260808/metrics.json",
    )
    parser.add_argument(
        "--output-dir",
        default="results/v9_2_final16_input_compatible_persistence_20260801_20260808",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-windows", type=int)
    args = parser.parse_args()

    output = resolve(args.output_dir)
    require_new_output_directory(output)
    config = load_config(resolve(args.config))
    history = int(config["data"]["history_hours"])
    forecast_hours = int(config["data"]["forecast_hours"])
    if history != 12 or forecast_hours != 6:
        raise ValueError("Frozen Final16 evaluation requires history=12 and forecast_hours=6")
    if list(config["data"]["input_variables"]) != list(INPUTS):
        raise ValueError(f"Input variable order must remain {INPUTS}")
    if list(config["data"]["target_variables"]) != list(TARGETS):
        raise ValueError(f"Target variable order must remain {TARGETS}")
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    normalization_path = resolve(config["data"]["normalization_stats"])
    registry_path = resolve(config["data"]["station_registry"])
    static_path = resolve(config["data"]["static_fields"])
    normalization = NormalizationStats.load(normalization_path)
    registry = load_fixed_registry(registry_path, include_virtual=True)
    if len(registry.station_ids) != 390:
        raise ValueError("Final16 operational evaluation requires exactly 390 input coordinates")
    dpc_input = resolve(args.dpc_input)
    marine_input = resolve(args.marine_input)
    dpc_msl = resolve(args.dpc_msl)
    marine_msl = resolve(args.marine_msl)
    era5t_instant = resolve(args.era5t_instant)
    era5t_accum = resolve(args.era5t_accum)
    old_metrics_path = resolve(args.old_metrics)
    required_paths = (
        dpc_input,
        marine_input,
        dpc_msl,
        marine_msl,
        era5t_instant,
        era5t_accum,
        old_metrics_path,
        normalization_path,
        registry_path,
        static_path,
    )
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required frozen evaluation files are missing: {missing}")

    base = load_v7_b_input(
        dpc_input,
        marine_input,
        normalization_path,
        expected_station_ids=registry.station_ids,
    )
    pressure_values, pressure_mask, pressure_age, pressure_metadata = operational.load_pressure(
        dpc_msl,
        marine_msl,
        times=base.times.astype("datetime64[ns]"),
        station_ids=base.station_ids,
        physical_count=base.physical_station_count,
        stats=normalization,
    )
    values = np.concatenate((pressure_values, base.values), axis=-1)
    mask = np.concatenate((pressure_mask, base.value_mask), axis=-1)
    age = np.concatenate((pressure_age, base.observation_age), axis=-1)
    if values.shape != mask.shape or values.shape != age.shape:
        raise ValueError("Operational values/mask/age tensor shapes differ")
    if values.shape[-1] != len(INPUTS) or not np.isfinite(values).all():
        raise ValueError("Assembled six-variable operational input is invalid")
    if np.any(age[mask] < 0):
        raise ValueError("Observation age contains negative (future) observations")

    times = base.times.astype("datetime64[ns]")
    total_count = len(times) - history - forecast_hours + 1
    if total_count != 152:
        raise ValueError(f"Frozen evaluation requires 152 windows, found {total_count}")
    count = total_count if args.max_windows is None else min(total_count, args.max_windows)
    if count < 1:
        raise ValueError("At least one evaluation window is required")
    starts = np.arange(count, dtype=np.int64)
    timeline = validate_timeline(times, starts, history, forecast_hours)
    target_grid = load_era5_target_grid(era5t_instant, era5t_accum, list(TARGETS))
    future_times = np.stack(
        [times[start + history : start + history + forecast_hours] for start in starts]
    )
    target_indices = target_grid.indices_for(future_times.reshape(-1)).reshape(
        count, forecast_hours
    )
    analysis_indices = target_grid.indices_for(times[starts + history - 1])
    if not np.isfinite(target_grid.values[target_indices]).all():
        raise ValueError("ERA5T target contains NaN or Inf")

    static_data = StaticFields.load(static_path)
    if not np.allclose(target_grid.latitudes, static_data.latitudes) or not np.allclose(
        target_grid.longitudes, static_data.longitudes
    ):
        raise ValueError("ERA5T and frozen Final16 grids differ")
    static = static_data.as_tensor().unsqueeze(0).to(device)
    land_mask = torch.from_numpy(static_data.land_sea_mask)
    coordinates = torch.from_numpy(base.coordinates).to(device)
    point_static = torch.from_numpy(base.station_static).to(device)

    final_path = resolve(args.checkpoint)
    source_path = resolve(args.source_checkpoint)
    final_checkpoint = validate_checkpoint(
        final_path, expected_sha256=EXPECTED_FINAL_SHA256, expected_station_count=390
    )
    source_checkpoint = validate_checkpoint(
        source_path, expected_sha256=EXPECTED_SOURCE_SHA256, expected_station_count=390
    )
    final_model, _ = operational.load_model(config, final_path, device)
    source_model, _ = operational.load_model(config, source_path, device)
    metrics = {
        name: ForecastMetricAccumulator(TARGETS, forecast_hours) for name in METHODS
    }
    processed = 0
    all_finite = True
    smoke_contract_verified = False
    processor_calls_during_reconstruction = 0
    reconstruction_interface_max_abs_difference = None
    started = time.perf_counter()
    with torch.no_grad():
        for offset in range(0, count, args.batch_size):
            batch_starts = starts[offset : offset + args.batch_size]
            size = len(batch_starts)
            batch = {
                "point_values": torch.from_numpy(
                    np.stack([values[start : start + history] for start in batch_starts])
                ).to(device),
                "value_mask": torch.from_numpy(
                    np.stack([mask[start : start + history] for start in batch_starts])
                ).to(device),
                "observation_age": torch.from_numpy(
                    np.stack([age[start : start + history] for start in batch_starts])
                ).to(device),
                "point_coords": coordinates[None].expand(size, -1, -1),
                "point_static": point_static[None].expand(size, -1, -1),
                "target": torch.empty(
                    size, forecast_hours, len(TARGETS), 31, 33, device=device
                ),
            }
            final_normalized = forward(final_model, batch, static)
            source_normalized = forward(source_model, batch, static)

            calls = [0]

            def count_processor_calls(_module: Any, _inputs: Any, _output: Any) -> None:
                calls[0] += 1

            hook = final_model.processor.register_forward_hook(count_processor_calls)
            current_normalized = reconstruct_current_without_processor(final_model, batch, static)
            hook.remove()
            processor_calls_during_reconstruction += calls[0]
            if calls[0] != 0:
                raise RuntimeError("Processor was called by reconstruction-persistence path")

            if offset == 0:
                trusted_forecast, trusted_current = forward_components(final_model, batch, static)
                if tuple(final_normalized.shape) != (size, 6, 5, 31, 33):
                    raise ValueError(f"Final forecast shape mismatch: {tuple(final_normalized.shape)}")
                if tuple(current_normalized.shape) != (size, 5, 31, 33):
                    raise ValueError(
                        f"Current reconstruction shape mismatch: {tuple(current_normalized.shape)}"
                    )
                if not torch.equal(final_normalized, trusted_forecast):
                    raise ValueError("Final forecast differs from existing forward_components interface")
                difference = float((current_normalized - trusted_current).abs().max().item())
                reconstruction_interface_max_abs_difference = difference
                if difference != 0.0:
                    raise ValueError(
                        "Processor-free reconstruction differs from the existing validated interface: "
                        f"max_abs_difference={difference}"
                    )
                smoke_contract_verified = True

            normalized_outputs = (final_normalized, source_normalized, current_normalized)
            if not all(bool(torch.isfinite(item).all()) for item in normalized_outputs):
                raise RuntimeError("Frozen model produced NaN or Inf")
            truth = torch.from_numpy(target_grid.values[target_indices[offset : offset + size]])
            dense_current = torch.from_numpy(
                target_grid.values[analysis_indices[offset : offset + size]]
            )
            current = denormalize_channels(
                current_normalized[:, None], list(TARGETS), normalization
            )[:, 0]
            predictions = {
                "v9_2_final16": denormalize_channels(
                    final_normalized, list(TARGETS), normalization
                ),
                "input_compatible_reconstruction_persistence": dense_grid_persistence(
                    current, forecast_hours
                ),
                "dense_era5t_persistence": dense_grid_persistence(
                    dense_current, forecast_hours
                ),
                "v9_1_frozen": denormalize_channels(
                    source_normalized, list(TARGETS), normalization
                ),
            }
            expected_shape = (size, forecast_hours, len(TARGETS), 31, 33)
            for name, prediction in predictions.items():
                if tuple(prediction.shape) != expected_shape:
                    raise ValueError(f"{name} shape {tuple(prediction.shape)} != {expected_shape}")
                if not bool(torch.isfinite(prediction).all()):
                    all_finite = False
                    raise RuntimeError(f"{name} contains NaN or Inf")
                metrics[name].update(prediction, truth, land_mask)
            processed += size
            print(f"input-compatible persistence evaluation {processed}/{count}", flush=True)

    computed = {name: accumulator.compute() for name, accumulator in metrics.items()}
    skills = {
        "v9_2_vs_input_compatible_reconstruction_persistence": rmse_skill(
            computed["v9_2_final16"],
            computed["input_compatible_reconstruction_persistence"],
        ),
        "v9_2_vs_dense_era5t_persistence": rmse_skill(
            computed["v9_2_final16"], computed["dense_era5t_persistence"]
        ),
        "v9_2_vs_v9_1_frozen": rmse_skill(
            computed["v9_2_final16"], computed["v9_1_frozen"]
        ),
    }
    bounded = count < total_count
    old_comparison: dict[str, Any]
    if bounded:
        old_comparison = {
            "exactly_identical": None,
            "reason": "bounded smoke run; full 152-window comparison is required for acceptance",
        }
    else:
        if args.batch_size != 16:
            raise ValueError("Full exact-reproduction run must use the frozen batch size 16")
        old_comparison = exact_comparison(computed, old_metrics_path)

    target_bytes = target_grid.values[target_indices].tobytes()
    input_files = {
        name: {
            "path": portable(path),
            "bytes": path.stat().st_size,
            "sha256": operational.sha256(path),
        }
        for name, path in {
            "dpc_input": dpc_input,
            "dpc_msl": dpc_msl,
            "marine_input": marine_input,
            "marine_msl": marine_msl,
            "era5t_instant": era5t_instant,
            "era5t_accum": era5t_accum,
            "normalization": normalization_path,
            "station_registry": registry_path,
            "static_fields": static_path,
        }.items()
    }
    result = {
        "schema_version": 1,
        "scientific_status": (
            "bounded_smoke_test" if bounded else "frozen_supplemental_2026_evaluation"
        ),
        "purpose": (
            "Fairly test whether the frozen Final16 Processor adds future-evolution skill "
            "beyond persistence of the model's own current-field reconstruction"
        ),
        "contract": {
            "input_period": [str(times[0]), str(times[-1])],
            "forecast_windows": count,
            "total_available_windows": total_count,
            "history_hours": history,
            "forecast_hours": forecast_hours,
            "physical_input": "DPC/MeteoHub",
            "marine_input": "Open-Meteo",
            "station_count": len(base.station_ids),
            "physical_station_count": base.physical_station_count,
            "marine_station_count": len(base.station_ids) - base.physical_station_count,
            "input_variables": list(INPUTS),
            "target_variables": list(TARGETS),
            "target_grid_shape": [31, 33],
            "target_reference": "ERA5T reanalysis, not direct station truth",
            "regions": ["full", "land", "sea"],
            "years_in_final_training": list(range(2010, 2026)),
            "2026_excluded_from_training": True,
            "training_performed": False,
            "parameter_tuning_performed": False,
        },
        "baseline_definition": {
            "name": "input_compatible_reconstruction_persistence",
            "inputs": "identical last 12 h real operational input tensor used by Final16",
            "path": (
                "Final16 encoder -> last historical latent -> Final16 reconstruction_decoder "
                "-> current 31x33 field -> repeat unchanged over +1...+6 h"
            ),
            "processor_used": False,
            "future_era5t_used": False,
            "dense_era5t_persistence_note": (
                "Uses the complete ERA5T analysis grid at forecast origin and therefore has "
                "strictly stronger input information than the operational model"
            ),
        },
        "checkpoints": {
            "v9_2_final16": final_checkpoint,
            "v9_1_frozen": source_checkpoint,
        },
        "input_files": input_files,
        "input_coverage": {
            "overall_valid_fraction": float(mask.mean()),
            "physical_valid_fraction": float(mask[:, : base.physical_station_count].mean()),
            "marine_valid_fraction": float(mask[:, base.physical_station_count :].mean()),
            "by_variable": {
                variable: {
                    "valid_fraction": float(mask[:, :, channel].mean()),
                    "valid_cells": int(mask[:, :, channel].sum()),
                }
                for channel, variable in enumerate(INPUTS)
            },
            "pressure_sources": pressure_metadata,
        },
        "field_metrics": computed,
        "rmse_skills": skills,
        "integrity_checks": {
            "smoke_contract_verified": smoke_contract_verified,
            "forecast_shape": [count, 6, 5, 31, 33],
            "current_reconstruction_shape": [count, 5, 31, 33],
            "reconstruction_matches_existing_validated_interface_exactly": (
                reconstruction_interface_max_abs_difference == 0.0
            ),
            "reconstruction_interface_max_abs_difference": (
                reconstruction_interface_max_abs_difference
            ),
            "processor_calls_during_reconstruction": processor_calls_during_reconstruction,
            "all_predictions_and_targets_finite": all_finite,
            "method_sample_counts": {name: count for name in METHODS},
            "sample_counts_identical": True,
            "same_target_tensor_used_for_all_methods": True,
            "target_tensor_sha256": sha256_bytes(target_bytes),
            "target_indices_sha256": sha256_bytes(target_indices.tobytes()),
            "timeline": timeline,
            "tp_semantics": (
                "Unchanged from the existing operational evaluation: hourly ERA5T TP at each "
                "lead; six-hour event TP is sum(max(tp_hourly_mm, 0)) over +1...+6"
            ),
            "no_future_era5t_in_input_compatible_baseline": True,
            "old_evaluation_comparison": old_comparison,
        },
        "event_scope": {
            "frozen_heavy_rain_threshold_mm_6h": 30.0,
            "frozen_strong_wind_threshold_ms": 15.0,
            "observed_event_cases_in_existing_152_window_evaluation": 0,
            "supports_only_ordinary_continuous_field_evaluation": True,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }

    output.mkdir(parents=True)
    (output / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    if not bounded:
        write_metric_csvs(output, computed)
        write_skill_csv(
            output / "skill_vs_reconstruction_persistence.csv",
            skills["v9_2_vs_input_compatible_reconstruction_persistence"],
            "input_compatible_reconstruction_persistence",
        )
        write_skill_csv(
            output / "skill_vs_dense_persistence.csv",
            skills["v9_2_vs_dense_era5t_persistence"],
            "dense_era5t_persistence",
        )
        plot_skill_figure(
            output,
            skills["v9_2_vs_input_compatible_reconstruction_persistence"],
        )
        write_readme(output, result)
    write_manifest(output, result, args)
    print(f"Evaluation complete: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
