from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "stage3_runner", ROOT / "scripts" / "run_v8_stage3.py"
)
assert SPEC and SPEC.loader
stage3_runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = stage3_runner
SPEC.loader.exec_module(stage3_runner)


def _summary(loss: float, source: float, degradation: float) -> dict[str, object]:
    return {
        "best_validation_loss": loss,
        "source_best_validation_loss": source,
        "best_epoch": 5,
        "reconstruction": {
            "degradation_percent_vs_stage2": degradation,
            "preservation_gate_passed": degradation <= 3.0,
        },
    }


def test_compare_reports_two_seed_improvement_and_passed_gate() -> None:
    stage3a = {
        "seed42": _summary(0.30, 0.35, 0.5),
        "seed43": _summary(0.31, 0.35, 0.7),
    }
    stage3b = {
        "seed42": _summary(0.28, 0.30, 1.5),
        "seed43": _summary(0.29, 0.31, 2.0),
    }
    result = stage3_runner.compare(stage3a, stage3b)
    assert result["test_years_read"] == []
    assert result["all_stage3b_reconstruction_gates_passed"] is True
    assert result["provisional_checkpoint_seed"] == 42
    assert result["runs"][0]["stage3b_skill_vs_stage2_percent"] > 0


def test_compare_blocks_recommendation_when_reconstruction_gate_fails() -> None:
    stage3a = {
        "seed42": _summary(0.30, 0.35, 0.5),
        "seed43": _summary(0.31, 0.35, 0.7),
    }
    stage3b = {
        "seed42": _summary(0.28, 0.30, 3.5),
        "seed43": _summary(0.29, 0.31, 2.0),
    }
    result = stage3_runner.compare(stage3a, stage3b)
    assert result["all_stage3b_reconstruction_gates_passed"] is False
    assert result["provisional_checkpoint_seed"] is None
