from __future__ import annotations

import torch

from stormengine_dl.training import physical_six_hour_event_loss


VARIABLES = ["msl", "u10", "v10", "t2m", "tp"]
NORMALIZATION = {"u10": (0.0, 1.0), "v10": (0.0, 1.0), "tp": (0.0, 1.0)}
THRESHOLDS = {
    "storm_rain_6h_mm": 30.0,
    "extreme_rain_6h_mm": 50.0,
    "strong_wind_ms": 15.0,
    "extreme_wind_ms": 20.0,
}


def objective(prediction: torch.Tensor, target: torch.Tensor):
    return physical_six_hour_event_loss(
        prediction,
        target,
        VARIABLES,
        NORMALIZATION,
        torch.ones(2, 2, dtype=torch.bool),
        thresholds=THRESHOLDS,
        classification_weight=0.05,
        intensity_weight=0.10,
    )


def test_original_thresholds_are_strict_and_six_hour() -> None:
    prediction = torch.zeros(1, 6, 5, 2, 2, requires_grad=True)
    target = torch.zeros_like(prediction)
    target[:, :, VARIABLES.index("tp"), 0, 0] = 50.0 / 6.0
    _, exact = objective(prediction, target)
    assert exact["event_positive_rain_cells"].item() == 0
    target[:, 0, VARIABLES.index("tp"), 0, 0] += 0.1
    _, above = objective(prediction, target)
    assert above["event_positive_rain_cells"].item() == 1


def test_matching_extreme_has_lower_loss_and_gradients() -> None:
    target = torch.zeros(1, 6, 5, 2, 2)
    target[:, :, VARIABLES.index("tp"), 0, 0] = 10.0
    target[:, :, VARIABLES.index("u10"), 0, 1] = 22.0
    poor = torch.zeros_like(target, requires_grad=True)
    good = target.clone().requires_grad_(True)
    poor_loss, components = objective(poor, target)
    good_loss, _ = objective(good, target)
    assert components["event_positive_rain_cells"].item() == 1
    assert components["event_positive_wind_cells"].item() == 1
    assert good_loss < poor_loss
    poor_loss.backward()
    assert poor.grad is not None
    assert torch.isfinite(poor.grad).all()
    assert poor.grad.abs().sum() > 0


def test_requires_exactly_six_forecast_hours() -> None:
    values = torch.zeros(1, 5, 5, 2, 2)
    try:
        objective(values, values)
    except ValueError as error:
        assert "six forecast hours" in str(error)
    else:
        raise AssertionError("five-hour event objective was accepted")
