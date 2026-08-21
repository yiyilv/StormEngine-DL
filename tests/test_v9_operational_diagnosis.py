from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from diagnose_v9_operational_era5t import (  # noqa: E402
    availability_relationship,
    normalize_points,
    source_coverage,
)
from stormengine_dl.data import NormalizationStats  # noqa: E402
from stormengine_dl.data.normalization import VariableStat  # noqa: E402


def test_source_coverage_separates_physical_and_marine() -> None:
    mask = np.zeros((2, 3, 2), dtype=bool)
    mask[:, 0, 0] = True
    mask[0, 1, 1] = True
    mask[:, 2, :] = True
    age = np.zeros(mask.shape, dtype=np.float32)
    age[mask] = np.arange(mask.sum(), dtype=np.float32)
    result = source_coverage(mask, age, ("u10", "t2m"), physical_count=2)
    assert result["physical_dpc"]["station_count"] == 2
    assert result["marine_open_meteo"]["station_count"] == 1
    assert result["physical_dpc"]["variables"]["u10"]["valid_cells"] == 2
    assert result["marine_open_meteo"]["overall_valid_fraction"] == 1.0


def test_availability_relationship_reports_negative_improvement_association() -> None:
    records = [
        {
            "physical_valid_fraction": coverage,
            "v9_a_sea_rmse": error,
            "v7_b_sea_rmse": error + 0.1,
        }
        for coverage, error in ((0.1, 3.0), (0.2, 2.0), (0.3, 1.0))
    ]
    result = availability_relationship(records)
    assert result["windows"] == 3
    assert result["models"]["v9_a"]["pearson_correlation"] < -0.99
    assert len(result["models"]["v7_b"]["coverage_tertiles"]) == 3


def test_availability_relationship_handles_short_smoke_run() -> None:
    records = [
        {
            "physical_valid_fraction": 0.2,
            "v9_a_sea_rmse": 1.0,
            "v7_b_sea_rmse": 1.1,
        },
        {
            "physical_valid_fraction": 0.3,
            "v9_a_sea_rmse": 0.9,
            "v7_b_sea_rmse": 1.0,
        },
    ]
    result = availability_relationship(records)
    assert len(result["models"]["v9_a"]["coverage_tertiles"]) == 2


def test_normalize_points_uses_frozen_variable_statistics() -> None:
    stats = NormalizationStats(
        {
            "u10": VariableStat(mean=2.0, std=2.0, count=1),
            "t2m": VariableStat(mean=10.0, std=5.0, count=1),
        }
    )
    values = np.asarray([[[4.0, 15.0]]], dtype=np.float32)
    normalized = normalize_points(values, ["u10", "t2m"], stats)
    np.testing.assert_allclose(normalized, np.asarray([[[1.0, 1.0]]]))
