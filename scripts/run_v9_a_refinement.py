#!/usr/bin/env python3
"""Diagnose and refine V9-A on 2023 without unlocking 2024 or 2025."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_v7 import load_config, resolve  # noqa: E402
from train_v9_output_form import require_development_protocol  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def trainer_command(
    args: argparse.Namespace,
    mode: str,
    variant: str,
    seed: int,
    *,
    learning_rate: float | None = None,
    reconstruction_weight: float | None = None,
    resume: Path | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(ROOT / "scripts" / "train_v9_output_form.py"),
        mode,
        "--config",
        str(resolve(args.config)),
        "--temporal-mode",
        "autoregressive",
        "--output-mode",
        "field",
        "--variant-name",
        variant,
        "--seed",
        str(seed),
        "--device",
        args.device,
        "--warm-start",
        str(resolve(args.warm_start)),
    ]
    if learning_rate is not None:
        command.extend(("--learning-rate", str(learning_rate)))
    if reconstruction_weight is not None:
        command.extend(("--reconstruction-loss-weight", str(reconstruction_weight)))
    if resume is not None:
        command.extend(("--resume", str(resume)))
    return command


def run(command: list[str]) -> None:
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v9_a_refinement.yaml")
    parser.add_argument("--warm-start", required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    args = parser.parse_args()

    config = load_config(resolve(args.config))
    require_development_protocol(config)
    refinement = config["refinement"]
    output_root = resolve(config["training"]["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    primary_seed = int(refinement["primary_seed"])
    replication_seed = int(refinement["replication_seed"])

    baseline_variant = str(refinement["epoch_zero_variant"])
    baseline_path = output_root / baseline_variant / f"seed_{primary_seed}" / "baseline_summary.json"
    if not baseline_path.exists():
        run(trainer_command(args, "baseline", baseline_variant, primary_seed))
    baseline = read_json(baseline_path)

    incumbent_config = refinement["incumbent"]
    incumbent_primary = read_json(resolve(incumbent_config["primary_summary"]))
    incumbent_replication = read_json(resolve(incumbent_config["replication_summary"]))
    incumbent_primary_value = float(incumbent_primary["best_validation_loss"])
    incumbent_mean = (
        incumbent_primary_value
        + float(incumbent_replication["best_validation_loss"])
    ) / 2.0

    primary_results: list[dict[str, Any]] = []
    for candidate in refinement["candidates"]:
        name = str(candidate["name"])
        kwargs = {
            "learning_rate": float(candidate["learning_rate"]),
            "reconstruction_weight": float(candidate["reconstruction_loss_weight"]),
        }
        summary = output_root / name / f"seed_{primary_seed}" / "train_summary.json"
        if not summary.exists():
            run(trainer_command(args, "preflight", name, primary_seed, **kwargs))
            run(trainer_command(args, "smoke", name, primary_seed, **kwargs))
            last = output_root / name / f"seed_{primary_seed}" / "last.pt"
            run(
                trainer_command(
                    args,
                    "train",
                    name,
                    primary_seed,
                    resume=last if last.exists() else None,
                    **kwargs,
                )
            )
        value = read_json(summary)
        primary_results.append(
            {
                **candidate,
                "primary_validation_loss": float(value["best_validation_loss"]),
                "primary_best_epoch": int(value["best_epoch"]),
            }
        )

    primary_results.sort(key=lambda row: float(row["primary_validation_loss"]))
    best_candidate = primary_results[0]
    primary_passed = (
        float(best_candidate["primary_validation_loss"]) < incumbent_primary_value
    )
    replication_result: dict[str, Any] | None = None
    selected = "incumbent"
    selected_checkpoint = resolve(incumbent_config["primary_checkpoint"])

    if primary_passed or not bool(
        refinement["replicate_only_if_primary_beats_incumbent"]
    ):
        name = str(best_candidate["name"])
        kwargs = {
            "learning_rate": float(best_candidate["learning_rate"]),
            "reconstruction_weight": float(best_candidate["reconstruction_loss_weight"]),
        }
        summary = output_root / name / f"seed_{replication_seed}" / "train_summary.json"
        if not summary.exists():
            run(trainer_command(args, "preflight", name, replication_seed, **kwargs))
            run(trainer_command(args, "smoke", name, replication_seed, **kwargs))
            last = output_root / name / f"seed_{replication_seed}" / "last.pt"
            run(
                trainer_command(
                    args,
                    "train",
                    name,
                    replication_seed,
                    resume=last if last.exists() else None,
                    **kwargs,
                )
            )
        replicated = read_json(summary)
        replication_value = float(replicated["best_validation_loss"])
        candidate_mean = (
            float(best_candidate["primary_validation_loss"]) + replication_value
        ) / 2.0
        replication_result = {
            "variant": name,
            "replication_validation_loss": replication_value,
            "replication_best_epoch": int(replicated["best_epoch"]),
            "two_seed_mean": candidate_mean,
            "incumbent_two_seed_mean": incumbent_mean,
        }
        if candidate_mean < incumbent_mean:
            selected = name
            selected_checkpoint = output_root / name / f"seed_{primary_seed}" / "best.pt"

    report = {
        "protocol": "v9-a-warm-start-refinement-v1",
        "development_years_only": {
            "train": config["data"]["train_years"],
            "validation": config["data"]["validation_years"],
            "confirmation_not_read": config["data"]["confirmation_years"],
            "test_not_read": config["data"]["test_years"],
        },
        "epoch_zero": baseline,
        "incumbent": {
            "variant": incumbent_config["variant"],
            "primary_validation_loss": incumbent_primary_value,
            "two_seed_mean": incumbent_mean,
        },
        "primary_results": primary_results,
        "primary_gate_passed": primary_passed,
        "replication": replication_result,
        "selected_for_frozen_2024_confirmation": selected,
        "selected_primary_checkpoint": str(selected_checkpoint),
        "selected_primary_checkpoint_sha256": sha256(selected_checkpoint),
    }
    (output_root / "refinement_protocol.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

