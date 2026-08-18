#!/usr/bin/env python3
"""Compare converged dense Processor family-development results."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


def load_summary(value: str) -> tuple[Path, dict[str, Any]]:
    path = Path(value).expanduser().resolve()
    if path.is_dir():
        path = path / "develop_summary.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("scientific_status") != "processor_family_development_only":
        raise ValueError(f"Not a Processor development summary: {path}")
    if not summary.get("stopped_early", False):
        raise ValueError(f"Processor candidate has not converged: {path}")
    return path, summary


def fixed_contract(summary: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(summary["contract"])
    value.pop("processor")
    value.pop("trainable_parameters")
    return value


def compare(values: list[str]) -> dict[str, Any]:
    if len(values) != 2:
        raise ValueError("Exactly two Processor family summaries are required")
    loaded = [load_summary(value) for value in values]
    reference = loaded[0][1]
    common = fixed_contract(reference)
    budget = (
        reference["seed"], tuple(reference["train_years"]),
        tuple(reference["validation_years"]), reference["train_batches_per_epoch"],
        reference["validation_batches_per_epoch"],
    )
    rows = []
    for path, summary in loaded:
        if fixed_contract(summary) != common:
            raise ValueError(f"Dense forecast contracts differ: {path}")
        candidate_budget = (
            summary["seed"], tuple(summary["train_years"]),
            tuple(summary["validation_years"]), summary["train_batches_per_epoch"],
            summary["validation_batches_per_epoch"],
        )
        if candidate_budget != budget:
            raise ValueError(f"Processor development budgets differ: {path}")
        rows.append({
            "summary": str(path),
            "family": summary["contract"]["processor"]["family"],
            "trainable_parameters": int(summary["contract"]["trainable_parameters"]),
            "best_epoch": int(summary["best_epoch"]),
            "completed_epochs": int(summary["completed_epochs"]),
            "best_validation_loss": float(summary["best_validation_loss"]),
            "validation_metrics": summary["validation_metrics"],
        })
    families = {row["family"] for row in rows}
    if families != {"convgru", "factorized_vit"}:
        raise ValueError("The comparison requires ConvGRU and factorized_vit")
    ranked = sorted(rows, key=lambda row: row["best_validation_loss"])
    winner, runner_up = ranked
    improvement = 100.0 * (
        runner_up["best_validation_loss"] - winner["best_validation_loss"]
    ) / runner_up["best_validation_loss"]
    return {
        "schema_version": 1,
        "scientific_status": "single_seed_processor_family_screen",
        "primary_metric": "normalized sea-weighted 2016 validation loss",
        "train_years": list(reference["train_years"]),
        "validation_years": list(reference["validation_years"]),
        "test_years_read": [],
        "ranking": ranked,
        "best_family_this_seed": winner["family"],
        "best_over_runner_up_percent": improvement,
        "selection_warning": (
            "This is a single-seed family screen. Repeat both families with a second seed "
            "before choosing a Processor family; if the ranking is unstable or nearly tied, "
            "prefer the smaller model and test both at the end-to-end compatibility gate."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("summaries", nargs="+")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = compare(args.summaries)
    print("family          parameters  val_loss  best_epoch")
    for row in result["ranking"]:
        print(
            f"{row['family']:<15} {row['trainable_parameters']:>10,d}  "
            f"{row['best_validation_loss']:.6f}  {row['best_epoch']:>10d}"
        )
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Comparison: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
