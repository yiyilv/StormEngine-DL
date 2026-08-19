from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "v9_2024_confirmation", ROOT / "scripts" / "evaluate_v9_2024_confirmation.py"
)
assert SPEC and SPEC.loader
confirmation = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = confirmation
SPEC.loader.exec_module(confirmation)


def comparison(mean_skill: float, wind_wins: int) -> dict[str, object]:
    aggregate = {
        "full": {},
        "land": {},
        "sea": {variable: {"skill": mean_skill} for variable in confirmation.TARGETS},
    }
    remaining = wind_wins
    by_lead = {}
    for lead in range(1, 7):
        sea = {}
        for variable in confirmation.TARGETS:
            value = mean_skill
            if variable in ("u10", "v10"):
                value = 0.1 if remaining > 0 else -0.1
                remaining -= int(remaining > 0)
            sea[variable] = {"skill": value}
        by_lead[str(lead)] = {"full": {}, "land": {}, "sea": sea}
    return {"aggregate": aggregate, "by_lead_hour": by_lead}


def reconstruction(candidate_ratio: float) -> dict[str, object]:
    def value(rmse: float) -> dict[str, object]:
        return {
            "aggregate": {
                "full": {},
                "land": {},
                "sea": {variable: {"rmse": rmse} for variable in confirmation.TARGETS},
            }
        }
    return {"v9_a": value(candidate_ratio), "v7_b": value(1.0)}


def test_confirmation_contract_is_locked_before_2024() -> None:
    config = confirmation.read_yaml(ROOT / "configs" / "v9_2024_confirmation.yaml")
    confirmation.validate_protocol(config)
    assert config["preflight_years"] == [2023]
    assert config["confirmation_years"] == [2024]
    assert config["locked_test_years"] == [2025]
    evidence = confirmation.validate_freeze_evidence(config)
    assert evidence["candidate"] == "A_lr25e6_recon000"


def test_confirmation_decision_requires_all_three_gates() -> None:
    config = confirmation.read_yaml(ROOT / "configs" / "v9_2024_confirmation.yaml")
    passed = confirmation.confirmation_decision(
        {"v9_a_vs_v7_b": comparison(0.02, 7)}, reconstruction(1.02), config
    )
    assert passed["passed"] is True
    assert passed["next_action"] == "unlock_one_time_2025_final_test"

    weak_skill = confirmation.confirmation_decision(
        {"v9_a_vs_v7_b": comparison(-0.01, 12)}, reconstruction(1.0), config
    )
    assert weak_skill["passed"] is False
    weak_wind = confirmation.confirmation_decision(
        {"v9_a_vs_v7_b": comparison(0.02, 6)}, reconstruction(1.0), config
    )
    assert weak_wind["passed"] is False
    weak_reconstruction = confirmation.confirmation_decision(
        {"v9_a_vs_v7_b": comparison(0.02, 12)}, reconstruction(1.031), config
    )
    assert weak_reconstruction["passed"] is False
    assert weak_reconstruction["next_action"] == "stop_v9_keep_2025_locked"


def test_protocol_rejects_year_or_gate_changes() -> None:
    config = confirmation.read_yaml(ROOT / "configs" / "v9_2024_confirmation.yaml")
    config["confirmation_years"] = [2025]
    with pytest.raises(ValueError, match="chronology"):
        confirmation.validate_protocol(config)

    config = confirmation.read_yaml(ROOT / "configs" / "v9_2024_confirmation.yaml")
    config["decision"]["minimum_positive_sea_wind_component_leads"] = 6
    with pytest.raises(ValueError, match="7/12"):
        confirmation.validate_protocol(config)
