#!/usr/bin/env python3
"""Sequentially run the four preregistered V8 Stage-2 experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "scripts" / "train_v8_stage2.py"
RESULT = ROOT / "results" / "v8_stage2_dev3y_processor_only"


@dataclass(frozen=True)
class Run:
    point_hidden: int
    latent_channels: int
    seed: int
    config: str
    output: str

    @property
    def spatial_name(self) -> str:
        return f"ph{self.point_hidden:03d}_lat{self.latent_channels:03d}"

    @property
    def name(self) -> str:
        return f"{self.spatial_name}_seed{self.seed}"


RUNS = (
    Run(64, 96, 42, "configs/v8_stage2_dev3y_ph064_lat096_seed42.yaml", "artifacts/v8_stage2_dev3y_ph064_lat096_seed42"),
    Run(64, 96, 43, "configs/v8_stage2_dev3y_ph064_lat096_seed43.yaml", "artifacts/v8_stage2_dev3y_ph064_lat096_seed43"),
    Run(96, 96, 42, "configs/v8_stage2_dev3y_ph096_lat096_seed42.yaml", "artifacts/v8_stage2_dev3y_ph096_lat096_seed42"),
    Run(96, 96, 43, "configs/v8_stage2_dev3y_ph096_lat096_seed43.yaml", "artifacts/v8_stage2_dev3y_ph096_lat096_seed43"),
)


def execute(command: list[str], dry_run: bool) -> None:
    print("\nRunning:", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def command(run: Run, mode: str, device: str, output: Path | None = None) -> list[str]:
    value = [
        sys.executable, "-u", str(TRAIN), mode,
        "--config", str(ROOT / run.config), "--device", device,
    ]
    if output is not None:
        value.extend(["--output-dir", str(output)])
    return value


def read_summary(run: Run) -> dict[str, Any]:
    path = ROOT / run.output / "train_summary.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    spatial = value.get("contract", {}).get("spatial_pretraining", {}).get("contract", {}).get("spatial_model", {})
    processor = value.get("contract", {}).get("processor", {})
    checks = {
        "status": value.get("scientific_status") == "stage2_validation_candidate",
        "seed": int(value.get("seed", -1)) == run.seed,
        "train_years": value.get("train_years") == [2013, 2014, 2015],
        "validation_years": value.get("validation_years") == [2016],
        "test_locked": value.get("test_years_read") == [],
        "converged": value.get("stopped_early") is True,
        "point_hidden": int(spatial.get("point_hidden", -1)) == run.point_hidden,
        "latent_channels": int(spatial.get("latent_channels", -1)) == run.latent_channels,
        "processor_layers": int(processor.get("layers", -1)) == 3,
        "processor_kernel": int(processor.get("kernel_size", -1)) == 3,
        "processor_random": processor.get("initialization") == "random_from_stage2_seed",
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Invalid completed Stage-2 run {path}: {failed}")
    return value


def ensure_train(run: Run, device: str, dry_run: bool) -> dict[str, Any] | None:
    output = ROOT / run.output
    summary = output / "train_summary.json"
    if summary.is_file():
        value = read_summary(run)
        print(
            f"SKIP {run.name}: converged validation={value['best_validation_loss']:.6f}",
            flush=True,
        )
        return value
    value = command(run, "train", device)
    last = output / "last.pt"
    if last.is_file():
        value.extend(["--resume", str(last)])
        print(f"RESUME {run.name}: {last}", flush=True)
    else:
        incomplete = [path for path in (output / "best.pt", output / "history.json") if path.exists()]
        if incomplete:
            raise RuntimeError(
                "Incomplete run has no last.pt; refusing to overwrite: "
                + ", ".join(map(str, incomplete))
            )
    execute(value, dry_run)
    return None if dry_run else read_summary(run)


def run_checks(device: str, dry_run: bool, include_pilot: bool) -> None:
    print("Stage 2 preflight: validate all four checkpoint/config contracts", flush=True)
    for run in RUNS:
        execute(command(run, "preflight", device), dry_run)
    # Shape and backward checks need one seed per spatial configuration; seed
    # replication does not change the tensor contract.
    representatives = (RUNS[0], RUNS[2])
    print("\nStage 2 smoke: one run per spatial configuration", flush=True)
    for run in representatives:
        output = ROOT / "artifacts" / "v8_stage2_smoke" / run.spatial_name
        if (output / "smoke_summary.json").is_file():
            print(f"SKIP smoke {run.spatial_name}: already complete", flush=True)
        else:
            execute(command(run, "smoke", device, output), dry_run)
    if include_pilot:
        print("\nStage 2 capped pilot: one seed per spatial configuration", flush=True)
        for run in representatives:
            output = ROOT / "artifacts" / "v8_stage2_pilot" / run.spatial_name
            if (output / "pilot_summary.json").is_file():
                print(f"SKIP pilot {run.spatial_name}: already complete", flush=True)
            else:
                execute(command(run, "pilot", device, output), dry_run)


def compare(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for point_hidden in (64, 96):
        selected = [
            summaries[run.name] for run in RUNS if run.point_hidden == point_hidden
        ]
        losses = [float(value["best_validation_loss"]) for value in selected]
        rows.append({
            "candidate": f"PH{point_hidden}-LAT96",
            "point_hidden": point_hidden,
            "latent_channels": 96,
            "validation_loss_by_seed": {
                str(run.seed): float(summaries[run.name]["best_validation_loss"])
                for run in RUNS if run.point_hidden == point_hidden
            },
            "mean_validation_loss": statistics.mean(losses),
            "sample_std_validation_loss": statistics.stdev(losses),
            "best_epoch_by_seed": {
                str(run.seed): int(summaries[run.name]["best_epoch"])
                for run in RUNS if run.point_hidden == point_hidden
            },
        })
    rows.sort(key=lambda row: row["mean_validation_loss"])
    gap = 100.0 * (
        rows[1]["mean_validation_loss"] - rows[0]["mean_validation_loss"]
    ) / rows[1]["mean_validation_loss"]
    if gap < 1.0:
        recommended = min(rows, key=lambda row: row["point_hidden"])
        rule = "two-seed mean losses differ by <1%; prefer the smaller spatial MLP"
    else:
        recommended = rows[0]
        rule = "prefer the lower two-seed mean 2016 validation loss"
    return {
        "schema_version": 1,
        "scientific_status": "v8_stage2_processor_only_two_seed_validation",
        "task": "frozen Stage-1 Encoder/Decoder plus trainable random-init L3K3 Processor",
        "train_years": [2013, 2014, 2015],
        "validation_years": [2016],
        "test_years_read": [],
        "primary_metric": "normalized sea-weighted validation loss",
        "ranking_by_two_seed_mean": rows,
        "relative_mean_gap_percent": gap,
        "provisional_spatial_recommendation": recommended["candidate"],
        "recommendation_rule": rule,
        "next_gate": "review per-variable/lead-hour metrics, then Stage 3 joint fine-tuning",
    }


def publish(summaries: dict[str, dict[str, Any]], result: dict[str, Any]) -> None:
    RESULT.mkdir(parents=True, exist_ok=True)
    for run in RUNS:
        source = ROOT / run.output
        destination = RESULT / "runs" / run.name
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("train_summary.json", "history.json"):
            shutil.copy2(source / name, destination / name)
    (RESULT / "comparison.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    recommendation = result["provisional_spatial_recommendation"]
    (RESULT / "README.md").write_text(
        "# V8 Stage 2 — Processor-only sparse forecasting\n\n"
        "Two converged Stage-1 spatial candidates were each combined with a "
        "randomly initialized ConvGRU L3K3. Encoder and Decoder were frozen; "
        "only Processor parameters were optimized. Each candidate was repeated "
        "with seeds 42 and 43 on 2013--2015 training and 2016 validation. "
        "The 2017 test set was not read.\n\n"
        f"Provisional spatial recommendation: **{recommendation}**. This is a "
        "validation-stage decision; Stage 3 joint fine-tuning remains required.\n",
        encoding="utf-8",
    )
    files = sorted(path for path in RESULT.rglob("*") if path.is_file() and path.name != "manifest.json")
    manifest = {
        "schema_version": 1,
        "test_years_read": [],
        "files": [
            {
                "path": str(path.relative_to(RESULT)),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in files
        ],
    }
    (RESULT / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=("preflight", "checks", "train", "all"), default="all"
    )
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--skip-pilot", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.phase == "preflight":
        for run in RUNS:
            execute(command(run, "preflight", args.device), args.dry_run)
        return 0
    if args.phase in {"checks", "all"}:
        run_checks(args.device, args.dry_run, not args.skip_pilot)
        if args.phase == "checks":
            return 0

    summaries: dict[str, dict[str, Any]] = {}
    print("\nFormal Stage 2: four runs execute sequentially on one GPU", flush=True)
    for run in RUNS:
        value = ensure_train(run, args.device, args.dry_run)
        if value is not None:
            summaries[run.name] = value
    if args.dry_run and len(summaries) != len(RUNS):
        print("Dry run complete; comparison waits for unfinished runs.")
        return 0
    result = compare(summaries)
    publish(summaries, result)
    print("\nStage-2 comparison:", json.dumps(result, indent=2), flush=True)
    print("2017 test data read: no", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
