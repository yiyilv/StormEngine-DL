from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "overnight", ROOT / "scripts" / "run_convgru_overnight_selection.py"
)
assert SPEC and SPEC.loader
overnight = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = overnight
SPEC.loader.exec_module(overnight)


def summary(candidate, seed: int, loss: float, parameters: int = 1000):
    return {
        "scientific_status": "processor_family_development_only",
        "stopped_early": True,
        "seed": seed,
        "train_years": [2013, 2014, 2015],
        "validation_years": [2016],
        "test_years_read": [],
        "best_epoch": 20,
        "completed_epochs": 30,
        "best_validation_loss": loss,
        "train_batches_per_epoch": 1,
        "validation_batches_per_epoch": 1,
        "validation_metrics": {"sea": {"msl": {"rmse": loss}}},
        "contract": {
            "version": "x",
            "task": "dense",
            "processor": {
                "family": "convgru",
                "latent_channels": 96,
                "layers": candidate.layers,
                "kernel_size": candidate.kernel_size,
            },
            "trainable_parameters": parameters,
        },
    }


def test_five_candidate_ranking_selects_lowest_two():
    losses = [0.15, 0.14, 0.13, 0.12, 0.11]
    values = {
        candidate.name: summary(candidate, 42, loss)
        for candidate, loss in zip(overnight.CANDIDATES, losses)
    }
    result = overnight.rank_seed42(values)
    assert result["selected_for_seed43"] == ["L3K5", "L2K5"]
    assert result["test_years_read"] == []


def test_two_seed_close_result_prefers_smaller_model():
    selected = [overnight.CANDIDATES[2], overnight.CANDIDATES[4]]
    first, second = selected
    seed42 = {
        first.name: summary(first, 42, 0.1000, 1000),
        second.name: summary(second, 42, 0.0998, 2000),
    }
    seed43 = {
        first.name: summary(first, 43, 0.1000, 1000),
        second.name: summary(second, 43, 0.0998, 2000),
    }
    result = overnight.compare_two_seeds(selected, seed42, seed43)
    assert result["relative_mean_gap_percent"] < 1.0
    assert result["provisional_recommendation"] == first.name


def test_common_contract_detects_non_search_change():
    candidate = overnight.CANDIDATES[0]
    original = summary(candidate, 42, 0.1)
    changed = copy.deepcopy(original)
    changed["contract"]["processor"]["latent_channels"] = 48
    assert overnight._common_contract(original) != overnight._common_contract(changed)
