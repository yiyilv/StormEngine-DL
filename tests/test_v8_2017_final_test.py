from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "v8_2017_final_test", ROOT / "scripts" / "evaluate_v8_2017_final_test.py"
)
assert SPEC and SPEC.loader
final_test = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = final_test
SPEC.loader.exec_module(final_test)


def _comparison(skill_value: float, wind_wins: int) -> dict[str, object]:
    aggregate = {
        "full": {},
        "land": {},
        "sea": {
            variable: {"skill": skill_value}
            for variable in final_test.TARGETS
        },
    }
    by_lead = {}
    remaining = wind_wins
    for lead in range(1, 7):
        regions = {"full": {}, "land": {}, "sea": {}}
        for variable in final_test.TARGETS:
            value = skill_value
            if variable in ("u10", "v10"):
                value = 0.1 if remaining > 0 else -0.1
                remaining -= int(remaining > 0)
            regions["sea"][variable] = {"skill": value}
        by_lead[str(lead)] = regions
    return {"aggregate": aggregate, "by_lead_hour": by_lead}


def test_final_test_contract_is_locked_to_2017() -> None:
    config = final_test.read_yaml(ROOT / "configs" / "v8_2017_final_test.yaml")
    assert config["test_years"] == [2017]
    assert config["threshold_years"] == list(range(2010, 2016))
    assert config["models"] == ["v7_b", "v8_stage3a_seed42"]
    assert config["decision"]["minimum_positive_sea_wind_component_leads"] == 7


def test_final_decision_requires_mean_skill_and_wind_majority() -> None:
    config = final_test.read_yaml(ROOT / "configs" / "v8_2017_final_test.yaml")
    key = "v8_stage3a_seed42_vs_v7_b"
    passed = final_test.final_decision({key: _comparison(0.02, 7)}, config)
    assert passed["supports_v8_replacement_claim"] is True
    assert passed["positive_sea_wind_component_leads_total"] == 7

    weak_mean = final_test.final_decision({key: _comparison(-0.01, 12)}, config)
    assert weak_mean["supports_v8_replacement_claim"] is False
    weak_wind = final_test.final_decision({key: _comparison(0.02, 6)}, config)
    assert weak_wind["supports_v8_replacement_claim"] is False


def test_unlock_evidence_is_the_frozen_passing_2016_result() -> None:
    config = final_test.read_yaml(ROOT / "configs" / "v8_2017_final_test.yaml")
    evidence = final_test.validate_unlock_evidence(config)
    assert evidence["both_stage3a_seeds_passed"] is True
    assert evidence["scientific_status"] == "frozen_2016_validation_benchmark"
