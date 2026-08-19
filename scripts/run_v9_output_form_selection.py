#!/usr/bin/env python3
"""Run the four V9 candidates, then replicate only the best two on one GPU."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from check_v7 import load_config, resolve  # noqa: E402


def command(
    args: argparse.Namespace,
    variant: dict[str, str],
    seed: int,
    mode: str,
    *,
    resume: Path | None = None,
) -> list[str]:
    result = [
        sys.executable,
        "-u",
        str(ROOT / "scripts" / "train_v9_output_form.py"),
        mode,
        "--config",
        str(resolve(args.config)),
        "--temporal-mode",
        variant["temporal_mode"],
        "--output-mode",
        variant["output_mode"],
        "--variant-name",
        variant["name"],
        "--seed",
        str(seed),
        "--device",
        args.device,
        "--warm-start",
        str(Path(args.warm_start).expanduser().resolve()),
    ]
    if resume is not None:
        result.extend(("--resume", str(resume)))
    return result


def summary_path(config: dict[str, Any], variant: str, seed: int) -> Path:
    output = resolve(config["training"]["output_dir"])
    return output / variant / f"seed_{seed}" / "train_summary.json"


def last_checkpoint_path(config: dict[str, Any], variant: str, seed: int) -> Path:
    output = resolve(config["training"]["output_dir"])
    return output / variant / f"seed_{seed}" / "last.pt"


def run(command_value: list[str], *, dry_run: bool) -> None:
    print("Running:", " ".join(command_value), flush=True)
    if not dry_run:
        subprocess.run(command_value, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v9_dev_output_form.yaml")
    parser.add_argument("--warm-start", required=True, help="Frozen V7-B best.pt")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    config = load_config(resolve(args.config))
    development = config["development"]
    variants = list(development["variants"])
    primary_seed = int(development["primary_seed"])
    replication_seed = int(development["replication_seed"])
    top_k = int(development["top_k_for_replication"])

    # Fail before an overnight run if paths/config/model shapes are invalid.
    for variant in variants:
        run(command(args, variant, primary_seed, "preflight"), dry_run=args.dry_run)
        run(command(args, variant, primary_seed, "smoke"), dry_run=args.dry_run)
        path = summary_path(config, variant["name"], primary_seed)
        if args.rerun or not path.exists():
            last = last_checkpoint_path(config, variant["name"], primary_seed)
            resume = last if last.exists() and not args.rerun else None
            run(
                command(args, variant, primary_seed, "train", resume=resume),
                dry_run=args.dry_run,
            )
        else:
            print(f"Keeping completed primary run: {path}", flush=True)

    if args.dry_run:
        print("Dry run stops before data-dependent top-two selection.")
        return 0

    ranked: list[tuple[float, dict[str, str], Path]] = []
    for variant in variants:
        path = summary_path(config, variant["name"], primary_seed)
        value = json.loads(path.read_text(encoding="utf-8"))
        ranked.append((float(value["best_validation_loss"]), variant, path))
    ranked.sort(key=lambda item: item[0])
    selected = ranked[:top_k]
    print("Primary ranking:", flush=True)
    for rank, (metric, variant, _) in enumerate(ranked, start=1):
        print(f"  {rank}. {variant['name']}: {metric:.8f}", flush=True)

    for _, variant, _ in selected:
        run(command(args, variant, replication_seed, "preflight"), dry_run=False)
        run(command(args, variant, replication_seed, "smoke"), dry_run=False)
        path = summary_path(config, variant["name"], replication_seed)
        if args.rerun or not path.exists():
            last = last_checkpoint_path(config, variant["name"], replication_seed)
            resume = last if last.exists() and not args.rerun else None
            run(
                command(args, variant, replication_seed, "train", resume=resume),
                dry_run=False,
            )
        else:
            print(f"Keeping completed replication: {path}", flush=True)

    replicated_ranking: list[dict[str, object]] = []
    for primary_metric, variant, _ in selected:
        replication = json.loads(
            summary_path(config, variant["name"], replication_seed).read_text(encoding="utf-8")
        )
        replication_metric = float(replication["best_validation_loss"])
        replicated_ranking.append(
            {
                "variant": variant["name"],
                "seed_values": {
                    str(primary_seed): primary_metric,
                    str(replication_seed): replication_metric,
                },
                "mean": (primary_metric + replication_metric) / 2.0,
                "range": abs(primary_metric - replication_metric),
            }
        )
    replicated_ranking.sort(key=lambda item: float(item["mean"]))

    report = {
        "protocol": "v9-output-form-development-v1",
        "selection_metric": development["selection_metric"],
        "train_years": config["data"]["train_years"],
        "validation_years": config["data"]["validation_years"],
        "confirmation_years_not_read": config["data"]["confirmation_years"],
        "locked_test_years_not_read": config["data"]["test_years"],
        "primary_ranking": [
            {"rank": rank, "variant": variant["name"], "value": metric}
            for rank, (metric, variant, _) in enumerate(ranked, start=1)
        ],
        "replicated_variants": [variant["name"] for _, variant, _ in selected],
        "replicated_ranking": replicated_ranking,
        "candidate_for_one_time_2024_confirmation": replicated_ranking[0]["variant"],
        "confirmation_gate_not_read": config["confirmation"],
        "final_test_not_read": config["final_test"],
        "seeds": [primary_seed, replication_seed],
    }
    output = resolve(config["training"]["output_dir"])
    (output / "selection_protocol.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
