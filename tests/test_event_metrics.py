import math

import torch

from stormengine_dl.event_metrics import (
    EventMetricAccumulator,
    PhysicalSixHourEventAccumulator,
)


def test_upper_event_metrics_and_leads() -> None:
    prediction = torch.tensor([[[[2.0, 0.0]], [[3.0, 0.0]]]])
    target = torch.tensor([[[[2.0, 2.0]], [[0.0, 0.0]]]])
    accumulator = EventMetricAccumulator(2, direction="above", threshold=1.0)
    accumulator.update(prediction, target, torch.tensor([[True, True]]))
    result = accumulator.compute()
    aggregate = result["aggregate"]
    assert aggregate["hits"] == 1
    assert aggregate["misses"] == 1
    assert aggregate["false_alarms"] == 1
    assert aggregate["pod"] == 0.5
    assert aggregate["far"] == 0.5
    assert math.isclose(aggregate["csi"], 1 / 3)
    assert math.isclose(aggregate["event_conditioned_rmse"], math.sqrt(2.0))
    assert result["by_lead_hour"]["1"]["peak_cases"] == 1
    assert result["by_lead_hour"]["2"]["peak_cases"] == 0


def test_lower_event_uses_minimum_peak_bias() -> None:
    prediction = torch.tensor([[[[998.0, 1005.0]]]])
    target = torch.tensor([[[[996.0, 1004.0]]]])
    accumulator = EventMetricAccumulator(1, direction="below", threshold=1000.0)
    accumulator.update(prediction, target, torch.tensor([[True, True]]))
    result = accumulator.compute()["aggregate"]
    assert result["hits"] == 1
    assert result["peak_intensity_bias"] == 2.0


def test_region_mask_excludes_land_cells() -> None:
    prediction = torch.tensor([[[[10.0, 10.0]]]])
    target = torch.tensor([[[[10.0, 0.0]]]])
    accumulator = EventMetricAccumulator(1, direction="above", threshold=5.0)
    accumulator.update(prediction, target, torch.tensor([[False, True]]))
    result = accumulator.compute()["aggregate"]
    assert result["hits"] == 0
    assert result["false_alarms"] == 1


def test_six_hour_physical_events_use_accumulated_rain_and_maximum_wind() -> None:
    variables = ["msl", "u10", "v10", "t2m", "tp"]
    prediction = torch.zeros((2, 6, 5, 1, 2))
    target = torch.zeros_like(prediction)

    # Forecast case 0: one correctly detected 12 mm rain cell and one missed
    # 16 m/s wind cell.
    prediction[0, :, 4, 0, 0] = 2.0
    target[0, :, 4, 0, 0] = 2.0
    prediction[0, 2, 1, 0, 1] = 14.0
    target[0, 2, 1, 0, 1] = 16.0

    # Forecast case 1: one correctly detected compound storm and one spatial
    # false alarm above the 20 m/s extreme-wind threshold.
    prediction[1, :, 4, 0, 1] = 6.0
    target[1, :, 4, 0, 1] = 6.0
    prediction[1, 1, 1, 0, 1] = 16.0
    target[1, 1, 1, 0, 1] = 16.0
    prediction[1, 0, 1, 0, 0] = 21.0

    accumulator = PhysicalSixHourEventAccumulator(6)
    accumulator.update(prediction, target, torch.tensor([[True, True]]), variables)
    result = accumulator.compute()

    rain = result["events"]["rain_6h_10mm"]
    assert rain["grid_cell"]["hits"] == 2
    assert rain["grid_cell"]["misses"] == 0
    assert rain["components"]["tp_6h_mm"]["event_conditioned_rmse"] == 0.0

    wind = result["events"]["strong_wind_6h_15ms"]
    assert wind["grid_cell"]["hits"] == 1
    assert wind["grid_cell"]["misses"] == 1
    assert wind["grid_cell"]["false_alarms"] == 1
    assert math.isclose(wind["grid_cell"]["csi"], 1 / 3)
    assert wind["forecast_case"]["hits"] == 1
    assert wind["forecast_case"]["misses"] == 1

    compound = result["events"]["compound_storm_6h"]
    assert compound["grid_cell"]["hits"] == 1
    assert compound["grid_cell"]["misses"] == 0
    assert result["events"]["extreme_wind_6h_20ms"]["grid_cell"][
        "false_alarms"
    ] == 1


def test_six_hour_physical_events_reject_wrong_horizon() -> None:
    try:
        PhysicalSixHourEventAccumulator(5)
    except ValueError as error:
        assert "six forecast hours" in str(error)
    else:
        raise AssertionError("five-hour event evaluation should be rejected")
