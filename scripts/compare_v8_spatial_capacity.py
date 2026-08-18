#!/usr/bin/env python3
"""Compare converged V8 point-hidden/latent capacity candidates."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


VARIABLES = ("msl", "u10", "v10", "t2m", "tp")
REQUIRED_CAPACITIES = {(64, 64), (96, 64), (64, 96), (96, 96)}


def load_summary(value: str) -> tuple[Path, dict[str, Any]]:
    path = Path(value).expanduser().resolve()
    if path.is_dir():
        path = path / "develop_summary.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("mode") != "develop" or summary.get("scientific_status") != "development_candidate_only":
        raise ValueError(f"Not a converged development summary: {path}")
    return path, summary


def fixed_contract(summary: dict[str, Any]) -> dict[str, Any]:
    contract = copy.deepcopy(summary["contract"])
    spatial = contract["spatial_model"]
    spatial.pop("point_hidden")
    spatial.pop("latent_channels")
    return contract


def row(path: Path, summary: dict[str, Any]) -> dict[str, Any]:
    spatial = summary["contract"]["spatial_model"]
    metrics = summary["validation_metrics"]
    return {
        "summary": str(path),
        "point_hidden": int(spatial["point_hidden"]),
        "latent_channels": int(spatial["latent_channels"]),
        "best_epoch": int(summary["best_epoch"]),
        "completed_epochs": int(summary["completed_epochs"]),
        "best_validation_loss": float(summary["best_validation_loss"]),
        **{
            f"{region}_rmse": {
                name: float(metrics[region][name]["rmse"]) for name in VARIABLES
            }
            for region in ("full", "land", "sea")
        },
    }


def improvements(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "validation_loss_percent": 100.0 * (
            reference["best_validation_loss"] - candidate["best_validation_loss"]
        ) / reference["best_validation_loss"]
    }
    for region in ("full", "land", "sea"):
        key = f"{region}_rmse"
        result[f"{region}_rmse_percent"] = {
            name: 100.0 * (reference[key][name] - candidate[key][name]) / reference[key][name]
            for name in VARIABLES
        }
    return result


def compare(paths: list[str]) -> dict[str, Any]:
    loaded = [load_summary(path) for path in paths]
    if len(loaded) != 4:
        raise ValueError("Exactly four capacity configurations are required")
    reference = loaded[0][1]
    expected_contract = fixed_contract(reference)
    expected_budget = (
        reference["seed"], tuple(reference["train_years"]), tuple(reference["validation_years"]),
        reference["target_max_epoch"], reference["train_batches_per_epoch"],
        reference["validation_batches_per_epoch"],
    )
    rows = []
    for path, summary in loaded:
        if not summary.get("stopped_early", False):
            raise ValueError(f"Capacity candidate has not converged: {path}")
        if fixed_contract(summary) != expected_contract:
            raise ValueError(f"Candidate differs beyond point_hidden/latent_channels: {path}")
        budget = (
            summary["seed"], tuple(summary["train_years"]), tuple(summary["validation_years"]),
            summary["target_max_epoch"], summary["train_batches_per_epoch"],
            summary["validation_batches_per_epoch"],
        )
        if budget != expected_budget:
            raise ValueError(f"Candidate split or development budget differs: {path}")
        rows.append(row(path, summary))
    by_capacity = {(item["point_hidden"], item["latent_channels"]): item for item in rows}
    if set(by_capacity) != REQUIRED_CAPACITIES:
        raise ValueError(f"Required capacities are {sorted(REQUIRED_CAPACITIES)}")
    if float(reference["contract"]["spatial_model"]["gaussian_sigma"]) != 0.10:
        raise ValueError("Capacity development is fixed to gaussian_sigma=0.10")
    baseline = by_capacity[(64, 64)]
    ranked = sorted(rows, key=lambda item: item["best_validation_loss"])
    return {
        "schema_version": 1,
        "scientific_status": "three_year_capacity_selection_only",
        "train_years": list(reference["train_years"]),
        "validation_years": list(reference["validation_years"]),
        "gaussian_sigma": 0.10,
        "ranking_metric": "normalized sea-weighted 2016 validation loss",
        "ranking": ranked,
        "improvement_over_64x64_percent": {
            f"ph{key[0]}_lat{key[1]}": improvements(baseline, item)
            for key, item in by_capacity.items() if key != (64, 64)
        },
        "top_two_capacity_configs": [
            {"point_hidden": item["point_hidden"], "latent_channels": item["latent_channels"]}
            for item in ranked[:2]
        ],
        "next_gate": "Inspect effect size and variable trade-offs, then repeat only the strongest configurations with a second seed before Processor transfer.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("summaries", nargs="+")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = compare(args.summaries)
    print("point_hidden latent  val_loss  best_epoch")
    for item in result["ranking"]:
        print(
            f"{item['point_hidden']:>12d} {item['latent_channels']:>6d}  "
            f"{item['best_validation_loss']:.6f}  {item['best_epoch']:>10d}"
        )
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Comparison: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
