import math

import torch

from stormengine_dl.event_metrics import EventMetricAccumulator


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
