#!/usr/bin/env python3
"""Single Windows entry point for V9 events and the V9.1 pressure ablation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def trainer(
    mode: str,
    config: str,
    variant: str,
    seed: int,
    device: str,
    warm_start: str,
    *,
    resume: Path | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(ROOT / "scripts" / "train_v9_output_form.py"),
        mode,
        "--config",
        str(ROOT / config),
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
        "--warm-start",
        warm_start,
    ]
    if resume is not None:
        command.extend(("--resume", str(resume)))
    return command


def specs() -> tuple[tuple[str, str, str], ...]:
    return (
        (
            "configs/v9_1_control_5var.yaml",
            "control_5var",
            "artifacts/v9_1_pressure_ablation/control_5var/control_5var",
        ),
        (
            "configs/v9_1_pressure_6var.yaml",
            "pressure_6var",
            "artifacts/v9_1_pressure_ablation/pressure_6var/pressure_6var",
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=("events", "preflight", "smoke", "train", "validate", "test")
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--warm-start", default="artifacts/v7_b_2010_2017/best.pt"
    )
    parser.add_argument("--acknowledge-one-time-2019", action="store_true")
    args = parser.parse_args()
    warm_start = str((ROOT / args.warm_start).resolve())

    if args.mode == "events":
        run(
            [
                sys.executable,
                "-u",
                str(ROOT / "scripts" / "evaluate_v9_2025_events.py"),
                "--device",
                args.device,
            ]
        )
        return 0
    if args.mode in {"preflight", "smoke"}:
        for config, variant, _ in specs():
            run(trainer(args.mode, config, variant, 42, args.device, warm_start))
        return 0
    if args.mode == "train":
        for seed in (42, 43):
            for config, variant, output in specs():
                root = ROOT / output / f"seed_{seed}"
                summary = root / "train_summary.json"
                if summary.exists():
                    print(f"Skipping completed run: {summary}", flush=True)
                    continue
                last = root / "last.pt"
                run(
                    trainer(
                        "train",
                        config,
                        variant,
                        seed,
                        args.device,
                        warm_start,
                        resume=last if last.exists() else None,
                    )
                )
        return 0
    if args.mode == "validate":
        run(
            [
                sys.executable,
                "-u",
                str(ROOT / "scripts" / "evaluate_v9_1_pressure_ablation.py"),
                "validation",
                "--device",
                args.device,
            ]
        )
        return 0
    if not args.acknowledge_one_time_2019:
        raise ValueError("Inspect the frozen 2018 validation, then acknowledge the one-time 2019 test")
    run(
        [
            sys.executable,
            "-u",
            str(ROOT / "scripts" / "evaluate_v9_1_pressure_ablation.py"),
            "test",
            "--device",
            args.device,
            "--acknowledge-one-time-2019",
        ]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
