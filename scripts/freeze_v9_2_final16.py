#!/usr/bin/env python3
"""Freeze the completed V9.2 Final16 production model and its dependencies."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_v7 import load_config, resolve  # noqa: E402


EXPECTED_CHECKPOINT_SHA256 = "ea0afde51397bd88bc6aeca45e452cabadfd956821112c6828eb826fdc03e86f"
EXPECTED_YEARS = list(range(2010, 2026))
EXPECTED_INPUTS = ["msl", "u10", "v10", "i10fg", "t2m", "tp"]
EXPECTED_TARGETS = ["msl", "u10", "v10", "t2m", "tp"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", default="artifacts/v9_2_event_aware_final16/seed_42/final.pt"
    )
    parser.add_argument("--config", default="configs/v9_2_event_aware_final16.yaml")
    parser.add_argument(
        "--output", default="artifacts/v9_2_event_aware_final16/frozen_manifest.json"
    )
    parser.add_argument(
        "--publish", default="results/v9_2_final16/frozen_manifest.json"
    )
    args = parser.parse_args()

    checkpoint_path = resolve(args.checkpoint)
    config_path = resolve(args.config)
    config = load_config(config_path)
    dependencies = [
        config_path,
        resolve("configs/v9_2_event_aware_strong.yaml"),
        resolve("configs/v9_2_event_aware.yaml"),
        resolve("configs/v9_1_pressure_6var.yaml"),
        resolve("configs/v9_1_pressure_ablation_base.yaml"),
        resolve(config["data"]["normalization_stats"]),
        resolve(config["data"]["station_registry"]),
        resolve(config["data"]["static_fields"]),
        resolve(config["data"]["cache_identity"]),
        resolve("scripts/predict_v9_2_final16.py"),
        resolve("scripts/freeze_v9_2_final16.py"),
        resolve("docs/V9_2_FINAL16_MODEL_CARD.md"),
    ]
    evidence = [
        resolve("results/v9_2_2018_development_decision/development_evaluation_2018.json"),
        resolve("results/v9_2_2018_development_decision/formal_train_summary.json"),
        resolve("results/v9_2_final16/training_summary.json"),
        resolve("results/v9_2_final16/final16_summary_full.json"),
        resolve("results/v9_2_final16_operational_era5t_20260801_20260808/metrics.json"),
        resolve("results/v9_2_marine_only_stress_20260816_20260819/stress_summary.json"),
    ]
    required = [checkpoint_path, *dependencies, *evidence]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"V9.2 Final16 freeze inputs are missing: {missing}")
    checkpoint_hash = sha256(checkpoint_path)
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(
            f"Final16 checkpoint SHA-256 changed: {checkpoint_hash}; expected {EXPECTED_CHECKPOINT_SHA256}"
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    contract = checkpoint.get("model_contract", {})
    checks = {
        "status_is_production_refit": contract.get("scientific_status")
        == "production_refit_no_internal_holdout",
        "years_are_2010_2025": contract.get("train_years") == EXPECTED_YEARS,
        "history_is_12h": contract.get("history_hours") == 12,
        "forecast_is_6h": contract.get("forecast_hours") == 6,
        "station_count_is_390": contract.get("station_count") == 390,
        "input_order_matches": contract.get("input_variables") == EXPECTED_INPUTS,
        "target_order_matches": contract.get("target_variables") == EXPECTED_TARGETS,
        "no_internal_holdout": bool(contract.get("no_internal_validation_or_test")),
    }
    if not all(checks.values()):
        raise ValueError(f"V9.2 Final16 checkpoint contract failed: {checks}")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest = {
        "schema_version": 1,
        "model": "V9.2 Final16",
        "status": "frozen_production_candidate",
        "frozen_at_git_commit": commit,
        "definition": "2010-2025 production refit; 390 points; 12 h history; six mask-aware inputs; +1..+6 h five-variable Adriatic grid forecast",
        "checkpoint": record(checkpoint_path),
        "checkpoint_contract": contract,
        "contract_checks": checks,
        "dependencies": [record(path) for path in dependencies],
        "evidence": [record(path) for path in evidence],
        "deployment": {
            "inference_script": "scripts/predict_v9_2_final16.py",
            "physical_source": "239 DPC/MeteoHub stations with per-variable value, mask, and age",
            "marine_source": "151 Open-Meteo ICON-2I support points with per-variable value, mask, and age",
            "output_variables": EXPECTED_TARGETS,
            "checkpoint_location": "local external artifact; do not commit to ordinary Git",
        },
        "scientific_limits": {
            "independent_accuracy_claim_from_final16_refit": False,
            "event_skill_evidence": "2018 development holdout evaluation",
            "ordinary_operational_evidence": "2026-08-01--08 ERA5T evaluation",
            "outage_robustness_evidence": "2026-08-16--19 no-truth marine-only stress test",
        },
    }
    encoded = json.dumps(manifest, indent=2) + "\n"
    for destination in (resolve(args.output), resolve(args.publish)):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(encoded, encoding="utf-8")
        print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
