from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "v9_2025_final_test", ROOT / "scripts" / "evaluate_v9_2025_final_test.py"
)
assert SPEC and SPEC.loader
final_test = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = final_test
SPEC.loader.exec_module(final_test)


def comparison(mean_skill: float, wind_wins: int) -> dict[str, object]:
    aggregate = {
        "full": {},
        "land": {},
        "sea": {variable: {"skill": mean_skill} for variable in final_test.common.TARGETS},
    }
    remaining = wind_wins
    by_lead = {}
    for lead in range(1, 7):
        sea = {}
        for variable in final_test.common.TARGETS:
            value = mean_skill
            if variable in ("u10", "v10"):
                value = 0.1 if remaining > 0 else -0.1
                remaining -= int(remaining > 0)
            sea[variable] = {"skill": value}
        by_lead[str(lead)] = {"full": {}, "land": {}, "sea": sea}
    return {"aggregate": aggregate, "by_lead_hour": by_lead}


def reconstruction(ratio: float) -> dict[str, object]:
    def value(rmse: float) -> dict[str, object]:
        return {
            "aggregate": {
                "full": {},
                "land": {},
                "sea": {variable: {"rmse": rmse} for variable in final_test.common.TARGETS},
            }
        }
    return {"v9_a": value(ratio), "v7_b": value(1.0)}


def test_final_test_contract_and_unlock_are_frozen() -> None:
    config = final_test.common.read_yaml(ROOT / "configs" / "v9_2025_final_test.yaml")
    final_test.validate_protocol(config)
    assert config["preflight_years"] == [2023]
    assert config["confirmation_years"] == [2024]
    assert config["final_test_years"] == [2025]
    unlock = final_test.validate_unlock_evidence(config)
    assert unlock["confirmation_passed"] is True
    assert unlock["scientific_status"] == "one_time_2024_confirmation"


def test_final_decision_is_locked_and_terminal() -> None:
    config = final_test.common.read_yaml(ROOT / "configs" / "v9_2025_final_test.yaml")
    passed = final_test.final_decision(
        {"v9_a_vs_v7_b": comparison(0.02, 7)}, reconstruction(1.02), config
    )
    assert passed["supports_v9_replacement_claim"] is True
    assert passed["no_post_test_tuning"] is True
    failed = final_test.final_decision(
        {"v9_a_vs_v7_b": comparison(-0.01, 12)}, reconstruction(1.0), config
    )
    assert failed["supports_v9_replacement_claim"] is False


def test_final_protocol_rejects_year_or_gate_changes() -> None:
    config = final_test.common.read_yaml(ROOT / "configs" / "v9_2025_final_test.yaml")
    config["final_test_years"] = [2024]
    with pytest.raises(ValueError, match="chronology"):
        final_test.validate_protocol(config)
    config = final_test.common.read_yaml(ROOT / "configs" / "v9_2025_final_test.yaml")
    config["decision"]["maximum_mean_sea_reconstruction_degradation_percent"] = 4.0
    with pytest.raises(ValueError, match="3 percent"):
        final_test.validate_protocol(config)
