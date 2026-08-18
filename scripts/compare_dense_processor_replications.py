#!/usr/bin/env python3
"""Aggregate two-seed Processor family replications against persistence."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


FAMILIES = ("convgru", "factorized_vit")
REQUIRED_SEEDS = {42, 43}


def read(path_value: str) -> tuple[Path, dict[str, Any]]:
    path = Path(path_value).expanduser().resolve()
    if path.is_dir():
        path = path / "develop_summary.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path, json.loads(path.read_text(encoding="utf-8"))


def comparable_contract(summary: dict[str, Any]) -> dict[str, Any]:
    contract = dict(summary["contract"])
    contract.pop("processor")
    contract.pop("trainable_parameters")
    return contract


def compare(summary_values: list[str], persistence_value: str) -> dict[str, Any]:
    if len(summary_values) != 4:
        raise ValueError("four summaries are required: two families x two seeds")
    loaded = [read(value) for value in summary_values]
    persistence_path, persistence = read(persistence_value)
    if persistence.get("scientific_status") != "processor_development_validation_baseline":
        raise ValueError("persistence file does not have the development-baseline contract")
    reference = loaded[0][1]
    if int(persistence["validation_batches"]) != int(reference["validation_batches_per_epoch"]):
        raise ValueError("persistence and learned models use different validation batching")
    contract = comparable_contract(reference)
    split = (
        tuple(reference["train_years"]), tuple(reference["validation_years"]),
        reference["train_batches_per_epoch"], reference["validation_batches_per_epoch"],
    )
    rows: list[dict[str, Any]] = []
    for path, summary in loaded:
        if summary.get("scientific_status") != "processor_family_development_only":
            raise ValueError(f"not a Processor development result: {path}")
        if not summary.get("stopped_early", False):
            raise ValueError(f"candidate has not reached validation early stopping: {path}")
        if comparable_contract(summary) != contract:
            raise ValueError(f"dense Processor forecast contracts differ: {path}")
        candidate_split = (
            tuple(summary["train_years"]), tuple(summary["validation_years"]),
            summary["train_batches_per_epoch"], summary["validation_batches_per_epoch"],
        )
        if candidate_split != split:
            raise ValueError(f"development splits or budgets differ: {path}")
        rows.append({
            "summary": str(path),
            "family": summary["contract"]["processor"]["family"],
            "seed": int(summary["seed"]),
            "trainable_parameters": int(summary["contract"]["trainable_parameters"]),
            "best_epoch": int(summary["best_epoch"]),
            "completed_epochs": int(summary["completed_epochs"]),
            "best_validation_loss": float(summary["best_validation_loss"]),
        })
    for family in FAMILIES:
        seeds = {row["seed"] for row in rows if row["family"] == family}
        if seeds != REQUIRED_SEEDS:
            raise ValueError(f"{family} requires seeds {sorted(REQUIRED_SEEDS)}, got {sorted(seeds)}")
    persistence_loss = float(persistence["normalized_sea_weighted_validation_loss"])
    family_summary: dict[str, Any] = {}
    for family in FAMILIES:
        family_rows = sorted(
            (row for row in rows if row["family"] == family), key=lambda row: row["seed"]
        )
        losses = [row["best_validation_loss"] for row in family_rows]
        skills = [1.0 - loss / persistence_loss for loss in losses]
        family_summary[family] = {
            "runs": family_rows,
            "mean_validation_loss": statistics.mean(losses),
            "validation_loss_std": statistics.stdev(losses),
            "mean_skill_vs_persistence": statistics.mean(skills),
            "skill_vs_persistence_by_seed": {
                str(row["seed"]): skill for row, skill in zip(family_rows, skills)
            },
        }
    seed_winners = {
        str(seed): min(
            (row for row in rows if row["seed"] == seed),
            key=lambda row: row["best_validation_loss"],
        )["family"]
        for seed in sorted(REQUIRED_SEEDS)
    }
    ranking = sorted(FAMILIES, key=lambda family: family_summary[family]["mean_validation_loss"])
    return {
        "schema_version": 1,
        "scientific_status": "two_seed_processor_family_development",
        "primary_metric": "normalized sea-weighted 2016 validation loss",
        "persistence": {
            "path": str(persistence_path),
            "validation_loss": persistence_loss,
        },
        "families": family_summary,
        "mean_loss_ranking": ranking,
        "winner_by_seed": seed_winners,
        "ranking_consistent_across_seeds": len(set(seed_winners.values())) == 1,
        "selection_rule": (
            "Select a family only if the ranking is consistent and variable/lead-hour metrics "
            "show no material deployment-relevant regression. If mean losses differ by less than "
            "one percent, prefer the smaller model or carry both into the end-to-end gate."
        ),
        "test_years_read": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("summaries", nargs="+")
    parser.add_argument("--persistence", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = compare(args.summaries, args.persistence)
    print("family          mean_loss  std       mean_skill_vs_persistence")
    for family in result["mean_loss_ranking"]:
        item = result["families"][family]
        print(
            f"{family:<15} {item['mean_validation_loss']:.6f}  "
            f"{item['validation_loss_std']:.6f}  {item['mean_skill_vs_persistence']:.4f}"
        )
    print("ranking consistent:", result["ranking_consistent_across_seeds"])
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Replication comparison: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
