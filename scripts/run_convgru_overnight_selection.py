#!/usr/bin/env python3
"""Run and replicate the V8 ConvGRU local search without reading the test set.

The workflow is deliberately sequential for a single GPU:

1. Ensure all five seed-42 depth/kernel candidates have converged.
2. Rank them by normalized sea-weighted 2016 validation loss.
3. Repeat the best two candidates with seed 43.
4. Produce a two-seed stability comparison and a provisional recommendation.

Completed runs are validated and skipped. Interrupted runs resume from last.pt.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "scripts" / "train_dense_processor.py"
RESULT_DIR = ROOT / "results" / "v8_processor_convgru_overnight_selection"


@dataclass(frozen=True)
class Candidate:
    layers: int
    kernel_size: int
    seed42_config: str
    seed42_output: str
    seed43_config: str
    seed43_output: str

    @property
    def name(self) -> str:
        return f"L{self.layers}K{self.kernel_size}"

    def config(self, seed: int) -> Path:
        return ROOT / (self.seed42_config if seed == 42 else self.seed43_config)

    def output(self, seed: int) -> Path:
        return ROOT / (self.seed42_output if seed == 42 else self.seed43_output)


CANDIDATES = (
    Candidate(
        1, 3,
        "configs/v8_processor_dev3y_convgru_l1k3.yaml",
        "artifacts/v8_processor_dev3y_convgru_l1k3_seed42",
        "configs/v8_processor_dev3y_convgru_l1k3_seed43.yaml",
        "artifacts/v8_processor_dev3y_convgru_l1k3_seed43",
    ),
    Candidate(
        2, 3,
        "configs/v8_processor_dev3y_convgru.yaml",
        "artifacts/v8_processor_dev3y_convgru_seed42",
        "configs/v8_processor_dev3y_convgru_seed43.yaml",
        "artifacts/v8_processor_dev3y_convgru_seed43",
    ),
    Candidate(
        3, 3,
        "configs/v8_processor_dev3y_convgru_l3k3.yaml",
        "artifacts/v8_processor_dev3y_convgru_l3k3_seed42",
        "configs/v8_processor_dev3y_convgru_l3k3_seed43.yaml",
        "artifacts/v8_processor_dev3y_convgru_l3k3_seed43",
    ),
    Candidate(
        2, 5,
        "configs/v8_processor_dev3y_convgru_l2k5.yaml",
        "artifacts/v8_processor_dev3y_convgru_l2k5_seed42",
        "configs/v8_processor_dev3y_convgru_l2k5_seed43.yaml",
        "artifacts/v8_processor_dev3y_convgru_l2k5_seed43",
    ),
    Candidate(
        3, 5,
        "configs/v8_processor_dev3y_convgru_l3k5.yaml",
        "artifacts/v8_processor_dev3y_convgru_l3k5_seed42",
        "configs/v8_processor_dev3y_convgru_l3k5_seed43.yaml",
        "artifacts/v8_processor_dev3y_convgru_l3k5_seed43",
    ),
)


def _common_contract(summary: dict[str, Any]) -> dict[str, Any]:
    contract = copy.deepcopy(summary["contract"])
    contract["processor"].pop("layers")
    contract["processor"].pop("kernel_size")
    contract.pop("trainable_parameters")
    return contract


def read_and_validate_summary(
    path: Path,
    *,
    candidate: Candidate,
    seed: int,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    summary = json.loads(path.read_text(encoding="utf-8"))
    processor = summary.get("contract", {}).get("processor", {})
    checks = {
        "scientific_status": summary.get("scientific_status") == "processor_family_development_only",
        "early_stopping": summary.get("stopped_early") is True,
        "seed": int(summary.get("seed", -1)) == seed,
        "train_years": summary.get("train_years") == [2013, 2014, 2015],
        "validation_years": summary.get("validation_years") == [2016],
        "test_not_read": summary.get("test_years_read") == [],
        "family": processor.get("family") == "convgru",
        "layers": int(processor.get("layers", -1)) == candidate.layers,
        "kernel_size": int(processor.get("kernel_size", -1)) == candidate.kernel_size,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"invalid completed run {path}; failed checks: {', '.join(failed)}")
    return summary


def _run(command: list[str], *, dry_run: bool) -> None:
    print("\nRunning:", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def ensure_run(
    candidate: Candidate,
    *,
    seed: int,
    device: str,
    dry_run: bool,
) -> dict[str, Any] | None:
    config = candidate.config(seed)
    output = candidate.output(seed)
    summary_path = output / "develop_summary.json"
    if not config.is_file():
        raise FileNotFoundError(config)
    if summary_path.is_file():
        summary = read_and_validate_summary(
            summary_path, candidate=candidate, seed=seed
        )
        print(
            f"SKIP {candidate.name} seed={seed}: already converged, "
            f"validation={summary['best_validation_loss']:.6f}",
            flush=True,
        )
        return summary

    base = [
        sys.executable, "-u", str(TRAIN_SCRIPT),
        "preflight", "--config", str(config), "--device", device,
    ]
    _run(base, dry_run=dry_run)

    command = [
        sys.executable, "-u", str(TRAIN_SCRIPT),
        "develop", "--config", str(config), "--device", device,
    ]
    checkpoint = output / "last.pt"
    if checkpoint.is_file():
        command.extend(["--resume", str(checkpoint)])
        print(f"RESUME {candidate.name} seed={seed} from {checkpoint}", flush=True)
    else:
        incomplete = [
            path for path in (output / "best.pt", output / "history.json") if path.exists()
        ]
        if incomplete:
            raise RuntimeError(
                "incomplete artifacts exist but last.pt is missing; refusing to overwrite: "
                + ", ".join(map(str, incomplete))
            )
    _run(command, dry_run=dry_run)
    if dry_run:
        return None
    return read_and_validate_summary(summary_path, candidate=candidate, seed=seed)


def rank_seed42(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if set(summaries) != {candidate.name for candidate in CANDIDATES}:
        raise ValueError("all five seed-42 candidates are required")
    reference = next(iter(summaries.values()))
    common = _common_contract(reference)
    rows: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        summary = summaries[candidate.name]
        if _common_contract(summary) != common:
            raise ValueError(f"{candidate.name} differs beyond layers and kernel size")
        rows.append({
            "candidate": candidate.name,
            "layers": candidate.layers,
            "kernel_size": candidate.kernel_size,
            "seed": 42,
            "trainable_parameters": int(summary["contract"]["trainable_parameters"]),
            "best_epoch": int(summary["best_epoch"]),
            "completed_epochs": int(summary["completed_epochs"]),
            "best_validation_loss": float(summary["best_validation_loss"]),
            "validation_metrics": summary["validation_metrics"],
            "summary": str(Path(candidate.seed42_output) / "develop_summary.json"),
        })
    rows.sort(key=lambda row: row["best_validation_loss"])
    return {
        "schema_version": 1,
        "scientific_status": "convgru_five_candidate_seed42_validation_screen",
        "primary_metric": "normalized sea-weighted 2016 validation loss",
        "selection_rule": "lowest validation loss; 2017 test remains unread",
        "ranking": rows,
        "selected_for_seed43": [row["candidate"] for row in rows[:2]],
        "test_years_read": [],
    }


def compare_two_seeds(
    selected: list[Candidate],
    seed42: dict[str, dict[str, Any]],
    seed43: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if len(selected) != 2:
        raise ValueError("exactly two candidates must be replicated")
    rows: list[dict[str, Any]] = []
    for candidate in selected:
        losses = [
            float(seed42[candidate.name]["best_validation_loss"]),
            float(seed43[candidate.name]["best_validation_loss"]),
        ]
        rows.append({
            "candidate": candidate.name,
            "layers": candidate.layers,
            "kernel_size": candidate.kernel_size,
            "trainable_parameters": int(
                seed42[candidate.name]["contract"]["trainable_parameters"]
            ),
            "validation_loss_by_seed": {"42": losses[0], "43": losses[1]},
            "mean_validation_loss": statistics.mean(losses),
            "sample_std_validation_loss": statistics.stdev(losses),
            "best_epoch_by_seed": {
                "42": int(seed42[candidate.name]["best_epoch"]),
                "43": int(seed43[candidate.name]["best_epoch"]),
            },
        })
    rows.sort(key=lambda row: row["mean_validation_loss"])
    relative_gap = 100.0 * (
        rows[1]["mean_validation_loss"] - rows[0]["mean_validation_loss"]
    ) / rows[1]["mean_validation_loss"]
    if relative_gap < 1.0:
        recommended = min(rows, key=lambda row: row["trainable_parameters"])
        rule = "mean losses differ by <1%; prefer the lower-complexity candidate"
    else:
        recommended = rows[0]
        rule = "prefer the lower two-seed mean validation loss"
    return {
        "schema_version": 1,
        "scientific_status": "convgru_two_seed_validation_replication",
        "primary_metric": "normalized sea-weighted 2016 validation loss",
        "ranking_by_two_seed_mean": rows,
        "relative_mean_gap_percent": relative_gap,
        "provisional_recommendation": recommended["candidate"],
        "recommendation_rule": rule,
        "interpretation": (
            "Provisional Processor choice only. Review per-variable land/sea validation metrics "
            "before freezing it for joint-model training."
        ),
        "test_years_read": [],
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    print(f"Saved: {path}", flush=True)


def publish_run(candidate: Candidate, seed: int) -> None:
    """Copy lightweight, reproducible run evidence into the Git results tree."""
    source = candidate.output(seed)
    destination = RESULT_DIR / "runs" / f"{candidate.name.lower()}_seed{seed}"
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("develop_summary.json", "history.json"):
        path = source / name
        if not path.is_file():
            raise FileNotFoundError(path)
        shutil.copy2(path, destination / name)


def publish_readme_and_manifest(result: dict[str, Any]) -> None:
    readme = f"""# V8 ConvGRU overnight selection

This experiment completes a five-candidate local ConvGRU search on a fixed
Processor-only development contract, then repeats the best two candidates with
a second random seed.

- training years: 2013--2015
- validation year: 2016
- locked test year: 2017 (not read)
- history/forecast: 12 h / 6 h
- candidates: L1K3, L2K3, L3K3, L2K5, L3K5
- primary selection metric: normalized sea-weighted validation loss
- replication seeds: 42 and 43

The provisional recommendation is **{result['provisional_recommendation']}**.
This is a Processor-development result, not a final end-to-end test result.
Per-variable land/sea validation metrics must be reviewed before the Processor
configuration is frozen for joint-model training.

`seed42_five_candidate_ranking.json` records the first screen;
`two_seed_comparison.json` records the replication decision; and `runs/`
contains checkpoint-free summaries and histories suitable for Git.
"""
    (RESULT_DIR / "README.md").write_text(readme, encoding="utf-8")
    files = sorted(
        path for path in RESULT_DIR.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )
    manifest = {
        "schema_version": 1,
        "test_years_read": [],
        "files": [
            {
                "path": str(path.relative_to(RESULT_DIR)),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in files
        ],
    }
    write_json(RESULT_DIR / "manifest.json", manifest)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Complete the 5-candidate ConvGRU screen and replicate its best two."
    )
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print missing training commands without running them.",
    )
    args = parser.parse_args()

    print("Stage 1/3: ensure all five seed-42 candidates are converged", flush=True)
    seed42: dict[str, dict[str, Any]] = {}
    for candidate in CANDIDATES:
        summary = ensure_run(
            candidate, seed=42, device=args.device, dry_run=args.dry_run
        )
        if summary is not None:
            seed42[candidate.name] = summary
    if args.dry_run and len(seed42) != len(CANDIDATES):
        print("Dry run complete; selection waits for the missing seed-42 summaries.")
        return 0

    ranking = rank_seed42(seed42)
    write_json(RESULT_DIR / "seed42_five_candidate_ranking.json", ranking)
    for candidate in CANDIDATES:
        publish_run(candidate, 42)
    print("\nSeed-42 ranking:", flush=True)
    for index, row in enumerate(ranking["ranking"], start=1):
        print(
            f"  {index}. {row['candidate']} val={row['best_validation_loss']:.6f} "
            f"params={row['trainable_parameters']:,}",
            flush=True,
        )

    by_name = {candidate.name: candidate for candidate in CANDIDATES}
    selected = [by_name[name] for name in ranking["selected_for_seed43"]]
    print(
        "\nStage 2/3: repeat the best two with seed 43: "
        + ", ".join(candidate.name for candidate in selected),
        flush=True,
    )
    seed43: dict[str, dict[str, Any]] = {}
    for candidate in selected:
        summary = ensure_run(
            candidate, seed=43, device=args.device, dry_run=args.dry_run
        )
        if summary is not None:
            seed43[candidate.name] = summary
    if args.dry_run and len(seed43) != len(selected):
        print("Dry run complete; final comparison waits for seed-43 summaries.")
        return 0

    print("\nStage 3/3: aggregate two-seed stability", flush=True)
    result = compare_two_seeds(selected, seed42, seed43)
    write_json(RESULT_DIR / "two_seed_comparison.json", result)
    for candidate in selected:
        publish_run(candidate, 43)
    publish_readme_and_manifest(result)
    print("\nTwo-seed ranking:", flush=True)
    for index, row in enumerate(result["ranking_by_two_seed_mean"], start=1):
        print(
            f"  {index}. {row['candidate']} mean={row['mean_validation_loss']:.6f} "
            f"std={row['sample_std_validation_loss']:.6f}",
            flush=True,
        )
    print(
        "Provisional recommendation:", result["provisional_recommendation"],
        f"({result['recommendation_rule']})",
        flush=True,
    )
    print("2017 test data read: no", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
