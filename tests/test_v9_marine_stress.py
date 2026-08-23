import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from stress_test_v9_2_marine_only import predicted_event_summary


def test_predicted_event_summary_uses_physical_six_hour_definitions() -> None:
    prediction = torch.zeros(2, 6, 5, 2, 2)
    prediction[0, :, 4, 0, 0] = 6.0
    prediction[1, :, 1, 1, 1] = 16.0
    sea = torch.ones(2, 2, dtype=torch.bool)
    thresholds = {
        "rain_6h_mm": 10.0,
        "storm_rain_6h_mm": 30.0,
        "extreme_rain_6h_mm": 50.0,
        "strong_wind_ms": 15.0,
        "extreme_wind_ms": 20.0,
    }
    result = predicted_event_summary(prediction, sea, thresholds)
    assert result["events"]["heavy_rain_6h_30mm"]["forecast_cases"] == 1
    assert result["events"]["strong_wind_6h_15ms"]["forecast_cases"] == 1
    assert result["events"]["storm_any_6h"]["forecast_cases"] == 2
