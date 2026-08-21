from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

from stormengine_dl.data.normalization import NormalizationStats, VariableStat


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "v8_2016_benchmark", ROOT / "scripts" / "evaluate_v8_2016_benchmark.py"
)
assert SPEC and SPEC.loader
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


def _metrics(rmse: float) -> dict[str, object]:
    regions = {
        region: {
            variable: {"mae": rmse / 2, "rmse": rmse}
            for variable in benchmark.TARGETS
        }
        for region in ("full", "land", "sea")
    }
    return {
        "aggregate": regions,
        "by_lead_hour": {str(lead): regions for lead in range(1, 7)},
    }


def test_skill_preserves_region_variable_and_lead_hierarchy() -> None:
    result = benchmark.skill(_metrics(2.0), _metrics(4.0))
    assert result["aggregate"]["sea"]["u10"]["skill"] == 0.5
    assert result["by_lead_hour"]["6"]["land"]["tp"]["skill"] == 0.5


def test_benchmark_contract_declares_no_2017_read() -> None:
    config = benchmark.read_yaml(ROOT / "configs" / "v8_2016_benchmark.yaml")
    assert config["threshold_years"] == list(range(2010, 2016))
    assert config["validation_years"] == [2016]
    assert 2017 not in config["threshold_years"] + config["validation_years"]
    assert config["acceptance"]["minimum_positive_wind_leads_per_component"] == 4


def test_event_thresholds_exclude_undeclared_years() -> None:
    class Dataset:
        all_times = np.asarray(
            ["2010-01-01T00", "2015-01-01T00", "2017-01-01T00"],
            dtype="datetime64[ns]",
        )
        target_grids = np.zeros((3, 5, 1, 2), dtype=np.float32)

    dataset = Dataset()
    # msl, u10, v10, t2m, tp. The excluded 2017 row is deliberately extreme.
    dataset.target_grids[:, 0] = np.asarray([[[1000, 1002]], [[998, 1004]], [[100, 100]]])
    dataset.target_grids[:, 1] = np.asarray([[[3, 4]], [[6, 8]], [[100, 100]]])
    dataset.target_grids[:, 2] = np.asarray([[[4, 3]], [[8, 6]], [[100, 100]]])
    dataset.target_grids[:, 4] = np.asarray([[[0.2, 1]], [[2, 4]], [[100, 100]]])
    normalization = NormalizationStats(
        {name: VariableStat(0.0, 1.0, 1) for name in benchmark.TARGETS}
    )
    thresholds = benchmark.derive_event_thresholds(
        dataset,
        list(benchmark.TARGETS),
        normalization,
        np.zeros((1, 2), dtype=np.float32),
        [2010, 2015],
        {
            "strong_wind_quantile": 0.95,
            "wet_precipitation_minimum_mm": 0.1,
            "wet_precipitation_quantile": 0.95,
            "fixed_precipitation_threshold_mm": 5.0,
            "low_msl_quantile": 0.05,
        },
    )
    assert thresholds["strong_wind_q95_ms"] < 20
    assert thresholds["heavy_precipitation_wet_q95_mm"] < 5
    assert thresholds["low_msl_q05_hpa"] > 900


def test_publish_writes_git_ready_manifest(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text("{}\n", encoding="utf-8")
    result = {
        "contract": {"samples": 100, "window_stride_hours": 3},
        "acceptance": {
            "both_seeds_passed": True,
            "seed_results": [
                {
                    "seed": 42,
                    "mean_sea_rmse_skill_vs_stage2": 0.1,
                    "positive_sea_wind_leads_vs_fair_persistence": {"u10": 5, "v10": 6},
                    "reconstruction_gate_passed": True,
                    "passed": True,
                }
            ],
        },
    }
    destination = tmp_path / "published"
    benchmark.publish_result(source, destination, result)
    manifest = __import__("json").loads(
        (destination / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["test_years_read"] == []
    assert {row["path"] for row in manifest["files"]} == {"benchmark.json", "README.md"}
