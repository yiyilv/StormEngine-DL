#!/usr/bin/env python3
"""Leakage-safe entry point for the original-document event-aware objective."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "base": ROOT / "configs" / "v9_2_event_aware.yaml",
    "strong": ROOT / "configs" / "v9_2_event_aware_strong.yaml",
}


def load_config(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if "extends" not in payload:
        return payload
    base = load_config(path.parent / payload["extends"])
    for key, value in payload.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key].update(value)
        else:
            base[key] = value
    return base


def trainer(
    mode: str,
    variant: str,
    seed: int,
    device: str,
    extra: list[str],
    config_path: Path,
) -> None:
    config = load_config(config_path)
    development = config["development"]
    source = ROOT / development["source_checkpoint"]
    run_extra = list(extra)
    if mode == "train":
        run_dir = ROOT / config["training"]["output_dir"] / variant / f"seed_{seed}"
        summary = run_dir / "train_summary.json"
        if summary.exists():
            raise FileExistsError(f"completed experiment already exists: {summary}")
        last = run_dir / "last.pt"
        if last.exists():
            run_extra.extend(("--resume", str(last)))
    command = [
        sys.executable,
        "-u",
        str(ROOT / "scripts" / "train_v9_output_form.py"),
        mode,
        "--config",
        str(config_path),
        "--temporal-mode",
        "autoregressive",
        "--output-mode",
        "field",
        "--variant-name",
        variant,
        "--seed",
        str(seed),
        "--device",
        device,
        "--initial-checkpoint",
        str(source),
        "--initial-checkpoint-sha256",
        development["source_checkpoint_sha256"],
        *run_extra,
    ]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=("preflight", "smoke", "pilot", "evaluate-pilot", "train")
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile", choices=tuple(CONFIGS), default="base")
    args = parser.parse_args()
    config_path = CONFIGS[args.profile]
    if args.mode == "preflight":
        trainer("preflight", "preflight", args.seed, args.device, [], config_path)
    elif args.mode == "smoke":
        trainer("smoke", "smoke", args.seed, args.device, [], config_path)
    elif args.mode == "pilot":
        trainer(
            "train",
            "pilot",
            args.seed,
            args.device,
            ["--epochs", "5", "--max-train-batches", "300", "--max-validation-batches", "75"],
            config_path,
        )
    elif args.mode == "evaluate-pilot":
        config = load_config(config_path)
        candidate = ROOT / config["training"]["output_dir"] / "pilot" / f"seed_{args.seed}" / "best.pt"
        command = [
            sys.executable,
            "-u",
            str(ROOT / "scripts" / "evaluate_v9_2_event_aware.py"),
            "--candidate",
            str(candidate),
            "--config",
            str(config_path),
            "--device",
            args.device,
        ]
        print("Running:", " ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    else:
        trainer("train", "formal", args.seed, args.device, [], config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
