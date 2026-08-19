#!/usr/bin/env python3
"""Run the two-seed V8 Stage-3 gradual-unfreezing workflow sequentially."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "scripts" / "train_v8_stage3.py"
RESULT = ROOT / "results" / "v8_stage3_6y_gradual_unfreezing"


@dataclass(frozen=True)
class Run:
    seed: int
    config: str
    stage3a_output: str
    stage3b_output: str

    @property
    def name(self) -> str:
        return f"seed{self.seed}"


RUNS = (
    Run(
        42, "configs/v8_stage3_6y.yaml",
        "artifacts/v8_stage3a_6y_ph064_lat096_seed42",
        "artifacts/v8_stage3b_6y_ph064_lat096_seed42",
    ),
    Run(
        43, "configs/v8_stage3_6y_seed43.yaml",
        "artifacts/v8_stage3a_6y_ph064_lat096_seed43",
        "artifacts/v8_stage3b_6y_ph064_lat096_seed43",
    ),
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, indent=2) + "\n").encode("utf-8"))


def execute(command: list[str], dry_run: bool) -> None:
    print("\nRunning:", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def command(
    run: Run, phase: str, mode: str, device: str, output: Path | None = None
) -> list[str]:
    value = [
        sys.executable, "-u", str(TRAIN), phase, mode,
        "--config", str(ROOT / run.config), "--device", device,
    ]
    if output is not None:
        value.extend(["--output-dir", str(output)])
    return value


def output_for(run: Run, phase: str) -> Path:
    return ROOT / (run.stage3a_output if phase == "stage3a" else run.stage3b_output)


def read_summary(run: Run, phase: str) -> dict[str, Any]:
    path = output_for(run, phase) / "train_summary.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    contract = value.get("contract", {})
    expected_modules = ["processor", "decoder"] if phase == "stage3a" else [
        "encoder", "processor", "decoder"
    ]
    checks = {
        "status": value.get("scientific_status") == f"{phase}_six_year_validation_candidate",
        "phase": value.get("phase") == phase,
        "seed": int(value.get("seed", -1)) == run.seed,
        "train_years": value.get("train_years") == list(range(2010, 2016)),
        "validation_years": value.get("validation_years") == [2016],
        "test locked": value.get("test_years_read") == [],
        "contract": contract.get("version") == "stormengine-v8-stage3-gradual-unfreeze-v1",
        "contract phase": contract.get("phase") == phase,
        "trainable modules": contract.get("optimization", {}).get("trainable_modules") == expected_modules,
        "finite loss": isinstance(value.get("best_validation_loss"), (int, float)),
        "frozen encoder verified": (
            value.get("frozen_encoder_verified") is True
            if phase == "stage3a" else True
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Invalid completed {phase} run {path}: {failed}")
    return value


def ensure_train(
    run: Run, phase: str, device: str, dry_run: bool
) -> dict[str, Any] | None:
    output = output_for(run, phase)
    summary = output / "train_summary.json"
    if summary.is_file():
        value = read_summary(run, phase)
        print(
            f"SKIP {phase} {run.name}: validation={value['best_validation_loss']:.6f}",
            flush=True,
        )
        return value
    value = command(run, phase, "train", device)
    last = output / "last.pt"
    if last.is_file():
        value.extend(["--resume", str(last)])
        print(f"RESUME {phase} {run.name}: {last}", flush=True)
    else:
        incomplete = [path for path in (output / "best.pt", output / "history.json") if path.exists()]
        if incomplete:
            raise RuntimeError(
                "Incomplete run has no last.pt; refusing to overwrite: "
                + ", ".join(map(str, incomplete))
            )
    execute(value, dry_run)
    return None if dry_run else read_summary(run, phase)


def run_checks(
    phase: str, device: str, dry_run: bool, include_pilot: bool
) -> None:
    print(f"{phase} preflight: validate both seed/checkpoint contracts", flush=True)
    for run in RUNS:
        execute(command(run, phase, "preflight", device), dry_run)
    representative = RUNS[0]
    smoke = ROOT / "artifacts" / "v8_stage3_smoke" / phase
    if (smoke / "smoke_summary.json").is_file():
        print(f"SKIP {phase} smoke: already complete", flush=True)
    else:
        execute(command(representative, phase, "smoke", device, smoke), dry_run)
    if include_pilot:
        pilot = ROOT / "artifacts" / "v8_stage3_pilot" / phase
        if (pilot / "pilot_summary.json").is_file():
            print(f"SKIP {phase} pilot: already complete", flush=True)
        else:
            execute(command(representative, phase, "pilot", device, pilot), dry_run)


def compare(
    stage3a: dict[str, dict[str, Any]], stage3b: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    rows = []
    for run in RUNS:
        a, b = stage3a[run.name], stage3b[run.name]
        stage2_loss = float(a["source_best_validation_loss"])
        stage3a_loss = float(a["best_validation_loss"])
        stage3b_loss = float(b["best_validation_loss"])
        rows.append({
            "seed": run.seed,
            "stage2_processor_only_validation_loss": stage2_loss,
            "stage3a_processor_decoder_validation_loss": stage3a_loss,
            "stage3b_joint_validation_loss": stage3b_loss,
            "stage3a_skill_vs_stage2_percent": 100.0 * (1.0 - stage3a_loss / stage2_loss),
            "stage3b_skill_vs_stage2_percent": 100.0 * (1.0 - stage3b_loss / stage2_loss),
            "stage3b_skill_vs_stage3a_percent": 100.0 * (1.0 - stage3b_loss / stage3a_loss),
            "stage3a_reconstruction_degradation_percent": float(
                a["reconstruction"]["degradation_percent_vs_stage2"]
            ),
            "stage3b_reconstruction_degradation_percent": float(
                b["reconstruction"]["degradation_percent_vs_stage2"]
            ),
            "stage3a_reconstruction_gate_passed": bool(
                a["reconstruction"]["preservation_gate_passed"]
            ),
            "stage3b_reconstruction_gate_passed": bool(
                b["reconstruction"]["preservation_gate_passed"]
            ),
            "stage3a_best_epoch": int(a["best_epoch"]),
            "stage3b_best_epoch": int(b["best_epoch"]),
        })
    final_losses = [row["stage3b_joint_validation_loss"] for row in rows]
    gates_pass = all(row["stage3b_reconstruction_gate_passed"] for row in rows)
    best = min(rows, key=lambda row: row["stage3b_joint_validation_loss"])
    return {
        "schema_version": 1,
        "scientific_status": "v8_stage3_two_seed_six_year_validation",
        "model": "PH64-LAT96 plus ConvGRU-L3K3",
        "train_years": list(range(2010, 2016)),
        "validation_years": [2016],
        "test_years_read": [],
        "reconstruction_preservation_threshold_percent": 3.0,
        "runs": rows,
        "stage3b_mean_validation_loss": statistics.mean(final_losses),
        "stage3b_sample_std_validation_loss": statistics.stdev(final_losses),
        "all_stage3b_reconstruction_gates_passed": gates_pass,
        "provisional_checkpoint_seed": best["seed"] if gates_pass else None,
        "next_gate": (
            "compare Stage 3 against frozen V7-B on 2016, then freeze architecture"
            if gates_pass else
            "do not unlock 2017; test reconstruction auxiliary loss first"
        ),
    }


def publish(
    stage3a: dict[str, dict[str, Any]],
    stage3b: dict[str, dict[str, Any]],
    comparison: dict[str, Any],
) -> None:
    RESULT.mkdir(parents=True, exist_ok=True)
    checkpoint_manifest: dict[str, Any] = {
        "schema_version": 1,
        "checkpoint_policy": "local_only_sha256_recorded",
        "checkpoints": {},
    }
    for run in RUNS:
        for phase, summary in (("stage3a", stage3a[run.name]), ("stage3b", stage3b[run.name])):
            source = output_for(run, phase)
            destination = RESULT / "runs" / run.name / phase
            write_json(destination / "train_summary.json", summary)
            history = json.loads((source / "history.json").read_text(encoding="utf-8"))
            write_json(destination / "history.json", history)
            checkpoint_manifest["checkpoints"][f"{run.name}_{phase}"] = {
                name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                for name in ("best.pt", "last.pt")
            }
    write_json(RESULT / "comparison.json", comparison)
    write_json(RESULT / "checkpoint_manifest.json", checkpoint_manifest)
    readme = (
        "# V8 Stage 3 — six-year gradual unfreezing\n\n"
        "Stage 3A trains Processor and Decoder while keeping Encoder frozen. "
        "Stage 3B then jointly fine-tunes Encoder, Processor, and Decoder with "
        "discriminative learning rates. Both seeds use 2010--2015 training and "
        "2016 validation; 2017 is not read. The fixed simultaneous reconstruction "
        "diagnostic must remain within the preregistered 3% degradation gate.\n"
    )
    (RESULT / "README.md").write_bytes(readme.encode("utf-8"))
    files = sorted(
        path for path in RESULT.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )
    manifest = {
        "schema_version": 1,
        "test_years_read": [],
        "hash_scope": "Git-ready LF bytes produced by this runner",
        "files": [
            {
                "path": path.relative_to(RESULT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in files
        ],
    }
    write_json(RESULT / "manifest.json", manifest)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("preflight", "checks", "stage3a", "stage3b", "all"),
        default="all",
    )
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--skip-pilot", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.phase == "preflight":
        for run in RUNS:
            execute(command(run, "stage3a", "preflight", args.device), args.dry_run)
        return 0
    if args.phase == "checks":
        run_checks("stage3a", args.device, args.dry_run, not args.skip_pilot)
        return 0

    stage3a: dict[str, dict[str, Any]] = {}
    if args.phase in {"stage3a", "all"}:
        run_checks("stage3a", args.device, args.dry_run, not args.skip_pilot)
        print("\nFormal Stage 3A: two seeds execute sequentially", flush=True)
        for run in RUNS:
            value = ensure_train(run, "stage3a", args.device, args.dry_run)
            if value is not None:
                stage3a[run.name] = value
        if args.phase == "stage3a":
            return 0

    if args.phase in {"stage3b", "all"}:
        for run in RUNS:
            stage3a[run.name] = read_summary(run, "stage3a")
        run_checks("stage3b", args.device, args.dry_run, not args.skip_pilot)
        print("\nFormal Stage 3B: two seeds execute sequentially", flush=True)
        stage3b: dict[str, dict[str, Any]] = {}
        for run in RUNS:
            value = ensure_train(run, "stage3b", args.device, args.dry_run)
            if value is not None:
                stage3b[run.name] = value
        if args.dry_run and len(stage3b) != len(RUNS):
            print("Dry run complete; comparison waits for unfinished runs.")
            return 0
        comparison = compare(stage3a, stage3b)
        publish(stage3a, stage3b, comparison)
        print("\nStage-3 comparison:", json.dumps(comparison, indent=2), flush=True)
        print("2017 test data read: no", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
