#!/usr/bin/env python3
"""Export the reviewable V9-A refinement record without model checkpoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts" / "v9_a_refinement"
OUTPUT = ROOT / "results" / "v9_a_refinement"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable(value: Any) -> Any:
    """Replace repository-local absolute paths with portable POSIX paths."""
    if isinstance(value, dict):
        return {key: portable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable(item) for item in value]
    if isinstance(value, str):
        try:
            path = Path(value)
            if path.is_absolute() and path.is_relative_to(ROOT):
                return str(path.relative_to(ROOT)).replace("\\", "/")
        except (OSError, ValueError):
            pass
    return value


def main() -> int:
    protocol_path = SOURCE / "refinement_protocol.json"
    if not protocol_path.exists():
        raise FileNotFoundError("V9-A refinement has not completed")
    protocol = portable(read_json(protocol_path))
    runs = [
        SOURCE / "A_lr25e6_recon002" / "seed_42" / "train_summary.json",
        SOURCE / "A_lr25e6_recon000" / "seed_42" / "train_summary.json",
        SOURCE / "A_lr25e6_recon000" / "seed_43" / "train_summary.json",
    ]
    if any(not path.exists() for path in runs):
        raise FileNotFoundError("One or more required V9-A refinement summaries are missing")

    summaries = [read_json(path) for path in runs]
    incumbent_mean = float(protocol["incumbent"]["two_seed_mean"])
    refined_mean = float(protocol["replication"]["two_seed_mean"])
    metrics = {
        "protocol": protocol["protocol"],
        "development_years_only": protocol["development_years_only"],
        "selection_metric": "validation_sea_weighted_mse",
        "epoch_zero_validation_loss": float(
            protocol["epoch_zero"]["validation_loss"]
        ),
        "incumbent": protocol["incumbent"],
        "primary_results": protocol["primary_results"],
        "replication": protocol["replication"],
        "relative_mse_improvement_over_incumbent_percent": (
            100.0 * (incumbent_mean - refined_mean) / incumbent_mean
        ),
        "selected_for_frozen_2024_confirmation": protocol[
            "selected_for_frozen_2024_confirmation"
        ],
        "selected_primary_checkpoint": protocol["selected_primary_checkpoint"],
        "selected_primary_checkpoint_sha256": protocol[
            "selected_primary_checkpoint_sha256"
        ],
        "confirmation_year_read": False,
        "locked_test_year_read": False,
    }
    histories = {
        "selection_metric": "validation_sea_weighted_mse",
        "runs": [
            {
                "variant": summary["variant"],
                "seed": summary["seed"],
                "learning_rate": summary["contract"]["learning_rate"],
                "reconstruction_loss_weight": summary[
                    "reconstruction_loss_weight"
                ],
                "best_validation_loss": summary["best_validation_loss"],
                "best_epoch": summary["best_epoch"],
                "epochs_completed": summary["epochs_completed"],
                "elapsed_seconds": summary["elapsed_seconds"],
                "history": summary["history"],
            }
            for summary in summaries
        ],
    }

    checkpoints = []
    for summary_path, summary in zip(runs, summaries, strict=True):
        checkpoint = summary_path.parent / "best.pt"
        checkpoints.append(
            {
                "variant": summary["variant"],
                "seed": summary["seed"],
                "path": str(checkpoint.relative_to(ROOT)).replace("\\", "/"),
                "size_bytes": checkpoint.stat().st_size,
                "sha256": sha256(checkpoint),
                "tracked_by_git": False,
            }
        )
    checkpoint_manifest = {
        "policy": "Checkpoints remain local; Git records paths, sizes, and SHA-256 only.",
        "selected_variant": protocol["selected_for_frozen_2024_confirmation"],
        "selected_primary_seed": 42,
        "checkpoints": checkpoints,
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT / "metrics.json", metrics)
    write_json(OUTPUT / "histories.json", histories)
    write_json(OUTPUT / "checkpoint_manifest.json", checkpoint_manifest)
    write_json(OUTPUT / "refinement_protocol.json", protocol)

    improvement = metrics["relative_mse_improvement_over_incumbent_percent"]
    readme = f"""# V9-A warm-start refinement (2023 development only)

This controlled diagnostic was run after the V9 output-form screen showed that the
field/autoregressive candidate converged unusually early. It measures the untouched
V7-B warm start, then changes only fine-tuning learning rate and reconstruction weight.

- Training years: `2020--2022`
- Validation and model-selection year: `2023`
- Confirmation year `2024`: **not read**
- Locked final-test year `2025`: **not read**
- Epoch-0 validation MSE: `{metrics['epoch_zero_validation_loss']:.6f}`
- Original V9-A two-seed mean: `{incumbent_mean:.6f}`
- Refined V9-A two-seed mean: `{refined_mean:.6f}`
- Relative validation-MSE improvement: `{improvement:.2f}%`

## Frozen candidate for one-time 2024 confirmation

`A_lr25e6_recon000`: field output, autoregressive forecasting, learning rate
`2.5e-5`, reconstruction-loss weight `0`, primary seed `42`.

- Best epoch: `14`
- Primary validation MSE: `0.310793`
- Replication validation MSE: `0.310198`
- Checkpoint SHA-256: `{protocol['selected_primary_checkpoint_sha256']}`

The checkpoint itself is intentionally not tracked. `checkpoint_manifest.json` records
all local checkpoint paths, sizes, and hashes. `histories.json` preserves every epoch,
and `metrics.json` is the compact scientific summary.
"""
    (OUTPUT / "README.md").write_text(readme, encoding="utf-8")
    print(f"Exported V9-A refinement record to {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
