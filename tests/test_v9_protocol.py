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


def test_v9_protocol_rejects_confirmation_leakage() -> None:
    module = load_trainer()
    config = module.load_config(ROOT / "configs" / "v9_dev_output_form.yaml")
    config["data"]["validation_years"] = [2024]
    with pytest.raises(ValueError, match="not frozen"):
        module.require_development_protocol(config)

