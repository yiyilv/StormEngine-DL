#!/usr/bin/env python3
"""Compare like-for-like V8 Stage-1 candidate screening summaries."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


def load_summary(value: str) -> tuple[Path, dict[str, Any]]:
    path = Path(value).expanduser().resolve()
    if path.is_dir():
        path = path / "screen_summary.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("mode") != "screen":
        raise ValueError(f"Not a candidate screen summary: {path}")
    if summary.get("scientific_status") != "candidate_screening_only":
        raise ValueError(f"Unexpected scientific status in {path}")
    return path, summary


def comparison_contract(summary: dict[str, Any]) -> dict[str, Any]:
    """Return the contract after removing the one intended ablation variable."""

    contract = copy.deepcopy(summary["contract"])
    contract["spatial_model"].pop("gaussian_sigma")
    return contract


def candidate_row(path: Path, summary: dict[str, Any]) -> dict[str, Any]:
    metrics = summary["validation_metrics"]
    return {
        "summary": str(path),
        "gaussian_sigma": float(summary["contract"]["spatial_model"]["gaussian_sigma"]),
        "best_epoch": int(summary["best_epoch"]),
        "best_validation_loss": float(summary["best_validation_loss"]),
        "sea_rmse": {
            variable: float(metrics["sea"][variable]["rmse"])
            for variable in ("msl", "u10", "v10", "t2m", "tp")
        },
        "full_rmse": {
            variable: float(metrics["full"][variable]["rmse"])
            for variable in ("msl", "u10", "v10", "t2m", "tp")
        },
    }


def compare(paths: list[str]) -> dict[str, Any]:
    loaded = [load_summary(path) for path in paths]
    if len(loaded) < 2:
        raise ValueError("At least two candidate summaries are required")
    reference = loaded[0][1]
    expected_contract = comparison_contract(reference)
    expected_train = reference["train_years"]
    expected_validation = reference["validation_years"]
    expected_budget = (
        reference["seed"], reference["initial_epoch"], reference["target_max_epoch"],
        reference["train_batches_per_epoch"], reference["validation_batches_per_epoch"],
    )
    rows: list[dict[str, Any]] = []
    seen_sigma: set[float] = set()
    for path, summary in loaded:
        if comparison_contract(summary) != expected_contract:
            raise ValueError(f"Candidate contract differs beyond gaussian_sigma: {path}")
        if summary["train_years"] != expected_train or summary["validation_years"] != expected_validation:
            raise ValueError(f"Candidate data split differs: {path}")
        candidate_budget = (
            summary["seed"], summary["initial_epoch"], summary["target_max_epoch"],
            summary["train_batches_per_epoch"], summary["validation_batches_per_epoch"],
        )
        if candidate_budget != expected_budget:
            raise ValueError(f"Candidate screen budget differs: {path}")
        row = candidate_row(path, summary)
        sigma = row["gaussian_sigma"]
        if sigma in seen_sigma:
            raise ValueError(f"Duplicate gaussian_sigma candidate: {sigma}")
        seen_sigma.add(sigma)
        rows.append(row)
    rows.sort(key=lambda row: row["best_validation_loss"])
    return {
        "schema_version": 1,
        "scientific_status": "screening_ranking_not_final_model_selection",
        "ranking_metric": "normalized sea-weighted 2016 validation loss",
        "candidate_count": len(rows),
        "screen_budget": {
            "seed": expected_budget[0],
            "initial_epoch": expected_budget[1],
            "target_max_epoch": expected_budget[2],
            "train_batches_per_epoch": expected_budget[3],
            "validation_batches_per_epoch": expected_budget[4],
        },
        "ranking": rows,
        "screening_winner_sigma": rows[0]["gaussian_sigma"],
        "next_gate": (
            "Run the strongest non-baseline candidate to full convergence, then compare "
            "full reconstruction and identical Processor transfer pilots."
        ),
    }


def print_table(result: dict[str, Any]) -> None:
    print("sigma  val_loss   sea_u10  sea_v10  sea_t2m  sea_tp")
    for row in result["ranking"]:
        sea = row["sea_rmse"]
        print(
            f"{row['gaussian_sigma']:>5.2f}  {row['best_validation_loss']:.6f}  "
            f"{sea['u10']:>7.4f}  {sea['v10']:>7.4f}  "
            f"{sea['t2m']:>7.4f}  {sea['tp']:>7.4f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("summaries", nargs="+")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = compare(args.summaries)
    print_table(result)
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Comparison: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
