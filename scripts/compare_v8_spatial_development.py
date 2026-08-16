#!/usr/bin/env python3
"""Compare equal-budget three-year V8 spatial development candidates."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


VARIABLES = ("msl", "u10", "v10", "t2m", "tp")


def load_summary(value: str) -> tuple[Path, dict[str, Any]]:
    path = Path(value).expanduser().resolve()
    if path.is_dir():
        path = path / "develop_summary.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("mode") != "develop":
        raise ValueError(f"Not a development summary: {path}")
    if summary.get("scientific_status") != "development_candidate_only":
        raise ValueError(f"Unexpected scientific status in {path}")
    return path, summary


def comparison_contract(summary: dict[str, Any]) -> dict[str, Any]:
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
        "full_rmse": {name: float(metrics["full"][name]["rmse"]) for name in VARIABLES},
        "land_rmse": {name: float(metrics["land"][name]["rmse"]) for name in VARIABLES},
        "sea_rmse": {name: float(metrics["sea"][name]["rmse"]) for name in VARIABLES},
    }


def relative_improvement(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Positive percentages mean sigma=0.15 has lower RMSE than sigma=0.10."""

    return {
        region: {
            name: 100.0 * (reference[region][name] - candidate[region][name]) / reference[region][name]
            for name in VARIABLES
        }
        for region in ("full_rmse", "land_rmse", "sea_rmse")
    }


def compare(paths: list[str]) -> dict[str, Any]:
    loaded = [load_summary(path) for path in paths]
    if len(loaded) != 2:
        raise ValueError("Exactly two three-year development candidates are required")
    reference = loaded[0][1]
    expected_contract = comparison_contract(reference)
    expected_budget = (
        reference["seed"], tuple(reference["train_years"]),
        tuple(reference["validation_years"]), reference["initial_epoch"],
        reference["target_max_epoch"], reference["train_batches_per_epoch"],
        reference["validation_batches_per_epoch"],
    )
    rows: list[dict[str, Any]] = []
    for path, summary in loaded:
        if not summary.get("stopped_early", False):
            raise ValueError(
                f"Candidate has not demonstrated validation convergence; extend its epoch ceiling: {path}"
            )
        if comparison_contract(summary) != expected_contract:
            raise ValueError(f"Candidate contract differs beyond gaussian_sigma: {path}")
        budget = (
            summary["seed"], tuple(summary["train_years"]), tuple(summary["validation_years"]),
            summary["initial_epoch"], summary["target_max_epoch"],
            summary["train_batches_per_epoch"], summary["validation_batches_per_epoch"],
        )
        if budget != expected_budget:
            raise ValueError(f"Candidate split or development budget differs: {path}")
        rows.append(candidate_row(path, summary))
    by_sigma = {item["gaussian_sigma"]: item for item in rows}
    if set(by_sigma) != {0.10, 0.15}:
        raise ValueError("Development comparison requires sigma=0.10 and sigma=0.15")
    ranked = sorted(rows, key=lambda item: item["best_validation_loss"])
    return {
        "schema_version": 1,
        "scientific_status": "three_year_architecture_selection_only",
        "train_years": list(reference["train_years"]),
        "validation_years": list(reference["validation_years"]),
        "ranking_metric": "normalized sea-weighted 2016 validation loss",
        "ranking": ranked,
        "sigma015_relative_improvement_percent": relative_improvement(
            by_sigma[0.10], by_sigma[0.15]
        ),
        "development_winner_sigma": ranked[0]["gaussian_sigma"],
        "next_gate": (
            "Retrain only the selected sigma once on ERA5 2010-2015, then run the "
            "identical Processor-transfer gate."
        ),
    }


def print_result(result: dict[str, Any]) -> None:
    print("sigma  val_loss   best_epoch  sea_u10  sea_v10  sea_t2m  sea_tp")
    for item in result["ranking"]:
        sea = item["sea_rmse"]
        print(
            f"{item['gaussian_sigma']:>5.2f}  {item['best_validation_loss']:.6f}  "
            f"{item['best_epoch']:>10d}  {sea['u10']:>7.4f}  {sea['v10']:>7.4f}  "
            f"{sea['t2m']:>7.4f}  {sea['tp']:>7.4f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("summaries", nargs="+")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = compare(args.summaries)
    print_result(result)
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Comparison: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
