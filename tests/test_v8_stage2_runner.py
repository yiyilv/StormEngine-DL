from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "stage2_runner", ROOT / "scripts" / "run_v8_stage2.py"
)
assert SPEC and SPEC.loader
stage2_runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = stage2_runner
SPEC.loader.exec_module(stage2_runner)


def _summary(loss: float, epoch: int = 20) -> dict[str, object]:
    return {"best_validation_loss": loss, "best_epoch": epoch}


def test_close_two_seed_result_prefers_smaller_spatial_candidate() -> None:
    values = {
        "ph064_lat096_seed42": _summary(0.2000),
        "ph064_lat096_seed43": _summary(0.2002),
        "ph096_lat096_seed42": _summary(0.1995),
        "ph096_lat096_seed43": _summary(0.1997),
    }
    result = stage2_runner.compare(values)
    assert result["relative_mean_gap_percent"] < 1.0
    assert result["provisional_spatial_recommendation"] == "PH64-LAT96"
    assert result["test_years_read"] == []


def test_clear_two_seed_result_prefers_lower_validation_mean() -> None:
    values = {
        "ph064_lat096_seed42": _summary(0.2200),
        "ph064_lat096_seed43": _summary(0.2210),
        "ph096_lat096_seed42": _summary(0.1900),
        "ph096_lat096_seed43": _summary(0.1910),
    }
    result = stage2_runner.compare(values)
    assert result["relative_mean_gap_percent"] >= 1.0
    assert result["provisional_spatial_recommendation"] == "PH96-LAT96"
