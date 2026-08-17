#!/usr/bin/env python3
"""Compare the converged local ConvGRU depth/kernel development screen."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


REQUIRED = {(1, 3), (2, 3), (3, 3), (2, 5)}


def read_summary(value: str) -> tuple[Path, dict[str, Any]]:
    path = Path(value).expanduser().resolve()
    if path.is_dir():
        path = path / "develop_summary.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("scientific_status") != "processor_family_development_only":
        raise ValueError(f"not a converged Processor development summary: {path}")
    if not summary.get("stopped_early", False):
        raise ValueError(f"candidate has not reached validation early stopping: {path}")
    processor = summary["contract"]["processor"]
    if processor.get("family") != "convgru":
        raise ValueError(f"not a ConvGRU candidate: {path}")
    return path, summary


def fixed_contract(summary: dict[str, Any]) -> dict[str, Any]:
    contract = copy.deepcopy(summary["contract"])
    processor = contract["processor"]
    processor.pop("layers")
    processor.pop("kernel_size")
    contract.pop("trainable_parameters")
    return contract


def improvement(reference: float, candidate: float) -> float:
    return 100.0 * (reference - candidate) / reference


def compare(values: list[str]) -> dict[str, Any]:
    if len(values) != 4:
        raise ValueError("exactly four ConvGRU configurations are required")
    loaded = [read_summary(value) for value in values]
    reference = loaded[0][1]
    common_contract = fixed_contract(reference)
    common_budget = (
        int(reference["seed"]), tuple(reference["train_years"]),
        tuple(reference["validation_years"]), int(reference["train_batches_per_epoch"]),
        int(reference["validation_batches_per_epoch"]),
    )
    rows: list[dict[str, Any]] = []
    for path, summary in loaded:
        if fixed_contract(summary) != common_contract:
            raise ValueError(f"candidate differs beyond layers/kernel_size: {path}")
        budget = (
            int(summary["seed"]), tuple(summary["train_years"]),
            tuple(summary["validation_years"]), int(summary["train_batches_per_epoch"]),
            int(summary["validation_batches_per_epoch"]),
        )
        if budget != common_budget:
            raise ValueError(f"candidate split, seed, or budget differs: {path}")
        processor = summary["contract"]["processor"]
        rows.append({
            "summary": str(path),
            "layers": int(processor["layers"]),
            "kernel_size": int(processor["kernel_size"]),
            "trainable_parameters": int(summary["contract"]["trainable_parameters"]),
            "best_epoch": int(summary["best_epoch"]),
            "completed_epochs": int(summary["completed_epochs"]),
            "best_validation_loss": float(summary["best_validation_loss"]),
            "validation_metrics": summary["validation_metrics"],
        })
    by_config = {(row["layers"], row["kernel_size"]): row for row in rows}
    if set(by_config) != REQUIRED:
        raise ValueError(f"required layer/kernel configurations are {sorted(REQUIRED)}")
    baseline = by_config[(2, 3)]
    ranked = sorted(rows, key=lambda row: row["best_validation_loss"])
    depth_effects = {
        "1_vs_2_layers_k3_percent": improvement(
            baseline["best_validation_loss"], by_config[(1, 3)]["best_validation_loss"]
        ),
        "3_vs_2_layers_k3_percent": improvement(
            baseline["best_validation_loss"], by_config[(3, 3)]["best_validation_loss"]
        ),
    }
    kernel_effect = improvement(
        baseline["best_validation_loss"], by_config[(2, 5)]["best_validation_loss"]
    )
    best_depth_k3 = min((1, 2, 3), key=lambda layers: by_config[(layers, 3)]["best_validation_loss"])
    interaction_needed = best_depth_k3 != 2 and kernel_effect > 0.0
    return {
        "schema_version": 1,
        "scientific_status": "convgru_local_tuning_seed42",
        "primary_metric": "normalized sea-weighted 2016 validation loss",
        "fixed": {
            "family": "convgru",
            "latent_channels": int(reference["contract"]["processor"]["latent_channels"]),
            "train_years": list(reference["train_years"]),
            "validation_years": list(reference["validation_years"]),
            "seed": int(reference["seed"]),
        },
        "ranking": ranked,
        "depth_effects_percent": depth_effects,
        "kernel_5_vs_3_at_2_layers_percent": kernel_effect,
        "best_depth_at_kernel3": best_depth_k3,
        "best_configuration_this_seed": {
            "layers": ranked[0]["layers"],
            "kernel_size": ranked[0]["kernel_size"],
        },
        "interaction_run_needed": interaction_needed,
        "interaction_candidate": (
            {"layers": best_depth_k3, "kernel_size": 5} if interaction_needed else None
        ),
        "next_gate": (
            "Run the indicated depth-by-kernel interaction candidate before replication."
            if interaction_needed else
            "Repeat the best non-baseline configuration with seed 43 only if its effect is meaningful; "
            "otherwise retain the two-seed 2-layer 3x3 baseline."
        ),
        "test_years_read": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("summaries", nargs="+")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = compare(args.summaries)
    print("layers kernel parameters  val_loss  best_epoch")
    for row in result["ranking"]:
        print(
            f"{row['layers']:>6d} {row['kernel_size']:>6d} "
            f"{row['trainable_parameters']:>10,d}  {row['best_validation_loss']:.6f} "
            f"{row['best_epoch']:>10d}"
        )
    print("interaction needed:", result["interaction_run_needed"])
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Local tuning comparison: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
