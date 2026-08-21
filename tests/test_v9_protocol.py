from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_trainer():
    spec = importlib.util.spec_from_file_location(
        "train_v9_output_form", ROOT / "scripts" / "train_v9_output_form.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v9_protocol_locks_confirmation_and_test_years() -> None:
    module = load_trainer()
    config = module.load_config(ROOT / "configs" / "v9_dev_output_form.yaml")
    module.require_development_protocol(config)
    assert config["data"]["train_years"] == [2020, 2021, 2022]
    assert config["data"]["validation_years"] == [2023]
    assert config["data"]["confirmation_years"] == [2024]
    assert config["data"]["test_years"] == [2025]
    assert config["development"]["v7b_checkpoint_sha256"] == (
        "2149960f7c71ebb88324aee033ed11fe903f2c34b4064eafdaa5ee95c677f991"
    )
    assert config["confirmation"] == {
        "year": 2024,
        "candidate_reference": "frozen_v7_b",
        "minimum_mean_sea_rmse_skill": 0.0,
        "minimum_positive_sea_wind_component_leads": 7,
        "maximum_reconstruction_degradation_percent": 3.0,
        "failure_action": "stop_without_reading_2025",
    }
    assert config["final_test"] == {
        "year": 2025,
        "run_once": True,
        "no_post_test_tuning": True,
    }


def test_v9_protocol_rejects_confirmation_leakage() -> None:
    module = load_trainer()
    config = module.load_config(ROOT / "configs" / "v9_dev_output_form.yaml")
    config["data"]["validation_years"] = [2024]
    with pytest.raises(ValueError, match="not frozen"):
        module.require_development_protocol(config)


def test_v9_a_refinement_is_development_only_and_controlled() -> None:
    module = load_trainer()
    config = module.load_config(ROOT / "configs" / "v9_a_refinement.yaml")
    module.require_development_protocol(config)
    assert config["data"]["train_years"] == [2020, 2021, 2022]
    assert config["data"]["validation_years"] == [2023]
    assert config["data"]["confirmation_years"] == [2024]
    assert config["data"]["test_years"] == [2025]
    assert config["training"]["output_dir"] == "artifacts/v9_a_refinement"
    assert config["refinement"]["candidates"] == [
        {
            "name": "A_lr25e6_recon002",
            "learning_rate": 0.000025,
            "reconstruction_loss_weight": 0.02,
        },
        {
            "name": "A_lr25e6_recon000",
            "learning_rate": 0.000025,
            "reconstruction_loss_weight": 0.0,
        },
    ]


def test_v9_1_pressure_protocol_reserves_2018_and_2019() -> None:
    module = load_trainer()
    control = module.load_config(ROOT / "configs" / "v9_1_control_5var.yaml")
    pressure = module.load_config(ROOT / "configs" / "v9_1_pressure_6var.yaml")
    module.require_development_protocol(control)
    module.require_development_protocol(pressure)
    assert control["data"]["train_years"] == list(range(2010, 2018))
    assert control["data"]["validation_years"] == [2018]
    assert control["data"]["test_years"] == [2019]
    assert control["data"]["input_variables"] == ["u10", "v10", "i10fg", "t2m", "tp"]
    assert pressure["data"]["input_variables"] == [
        "msl", "u10", "v10", "i10fg", "t2m", "tp"
    ]
    assert pressure["data"]["variable_capabilities"]["msl"]["include_virtual"] is True
    assert pressure["data"]["expected_variable_capability_counts"] == {"msl": 164}
