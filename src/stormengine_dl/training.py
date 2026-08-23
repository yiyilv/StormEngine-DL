"""Training and evaluation helpers for StormEngine V6 experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F


def sea_weight_map(land_sea_mask: torch.Tensor, sea_weight: float = 2.0) -> torch.Tensor:
    """Build the V6 mean-normalized spatial loss weights.

    ``land_sea_mask`` is one on land and zero on sea. The returned map has mean
    one, keeping the loss scale comparable when ``sea_weight`` changes.
    """
    if sea_weight <= 0:
        raise ValueError("sea_weight must be positive")
    mask = land_sea_mask.float()
    weights = 1.0 + (float(sea_weight) - 1.0) * (1.0 - mask)
    return weights / weights.mean().clamp_min(torch.finfo(weights.dtype).eps)


def weighted_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    spatial_weights: torch.Tensor,
) -> torch.Tensor:
    """Mean squared error with a broadcastable ``[H, W]`` spatial map."""
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes must match")
    if spatial_weights.shape != prediction.shape[-2:]:
        raise ValueError("spatial_weights must match prediction [H, W]")
    return ((prediction - target).square() * spatial_weights).mean()


def _balanced_focal_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    spatial_mask: torch.Tensor,
    *,
    gamma: float,
) -> torch.Tensor:
    """Focal BCE with separate positive/negative means.

    Separating the two classes prevents rare physical extremes from disappearing
    in the much larger background class. A batch containing only one class is
    still valid and contributes that class's loss.
    """
    selected_logits = logits[spatial_mask]
    selected_target = target[spatial_mask].to(logits.dtype)
    if selected_logits.numel() == 0:
        raise ValueError("event spatial mask selects no grid cells")
    probability = torch.sigmoid(selected_logits)
    base = F.binary_cross_entropy_with_logits(
        selected_logits, selected_target, reduction="none"
    )
    focal = torch.where(
        selected_target > 0.5,
        (1.0 - probability).pow(gamma),
        probability.pow(gamma),
    ) * base
    positive = selected_target > 0.5
    terms = []
    if positive.any():
        terms.append(focal[positive].mean())
    if (~positive).any():
        terms.append(focal[~positive].mean())
    return torch.stack(terms).mean()


def physical_six_hour_event_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    variables: Sequence[str],
    normalization: Mapping[str, tuple[float, float]],
    spatial_mask: torch.Tensor,
    *,
    thresholds: Mapping[str, float],
    classification_weight: float,
    intensity_weight: float,
    focal_gamma: float = 2.0,
    rain_temperature_mm: float = 5.0,
    wind_temperature_ms: float = 2.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Joint event classification and event-conditioned intensity loss.

    The labels exactly follow the original six-hour StormEngine definitions:
    extreme rain (>50 mm), extreme wind (>20 m/s), and compound storm
    (>30 mm and >15 m/s). Inputs remain continuous normalized fields, so this
    objective does not add an inference-only classification head.
    """
    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError("prediction and target must have shape [B, 6, C, H, W]")
    if prediction.shape[1] != 6:
        raise ValueError("physical event training requires exactly six forecast hours")
    required = {"u10", "v10", "tp"}
    if not required.issubset(variables):
        raise ValueError(f"physical event training requires {sorted(required)}")
    if spatial_mask.shape != prediction.shape[-2:]:
        raise ValueError("spatial_mask must match prediction [H, W]")
    for name in ("storm_rain_6h_mm", "extreme_rain_6h_mm", "strong_wind_ms", "extreme_wind_ms"):
        if name not in thresholds:
            raise ValueError(f"missing event threshold: {name}")

    def physical(values: torch.Tensor, variable: str) -> torch.Tensor:
        mean, std = normalization[variable]
        return values[:, :, variables.index(variable)] * float(std) + float(mean)

    pred_u = physical(prediction, "u10")
    pred_v = physical(prediction, "v10")
    pred_tp_hourly = physical(prediction, "tp")
    true_u = physical(target, "u10")
    true_v = physical(target, "v10")
    true_tp_hourly = physical(target, "tp")

    # Softplus retains a gradient for slightly negative precipitation forecasts;
    # labels retain the exact physical max(tp, 0) definition used in evaluation.
    pred_tp_6h = (F.softplus(pred_tp_hourly * 5.0) / 5.0).sum(dim=1)
    true_tp_6h = true_tp_hourly.clamp_min(0.0).sum(dim=1)
    # A tiny epsilon avoids the undefined derivative of hypot at u=v=0.
    # It is many orders below the 15/20 m/s physical thresholds.
    pred_wind = torch.sqrt(pred_u.square() + pred_v.square() + 1e-12).amax(dim=1)
    true_wind = torch.sqrt(true_u.square() + true_v.square() + 1e-12).amax(dim=1)

    rain50 = true_tp_6h > float(thresholds["extreme_rain_6h_mm"])
    wind20 = true_wind > float(thresholds["extreme_wind_ms"])
    storm = (
        (true_tp_6h > float(thresholds["storm_rain_6h_mm"]))
        & (true_wind > float(thresholds["strong_wind_ms"]))
    )
    rain50_logits = (
        pred_tp_6h - float(thresholds["extreme_rain_6h_mm"])
    ) / rain_temperature_mm
    wind20_logits = (
        pred_wind - float(thresholds["extreme_wind_ms"])
    ) / wind_temperature_ms
    storm_rain_logits = (
        pred_tp_6h - float(thresholds["storm_rain_6h_mm"])
    ) / rain_temperature_mm
    storm_wind_logits = (
        pred_wind - float(thresholds["strong_wind_ms"])
    ) / wind_temperature_ms
    # AND is represented by the limiting condition. torch.minimum is
    # piecewise differentiable and preserves the physical decision boundary.
    storm_logits = torch.minimum(storm_rain_logits, storm_wind_logits)

    expanded_mask = spatial_mask.bool()[None].expand(prediction.shape[0], -1, -1)
    classification = torch.stack(
        (
            _balanced_focal_bce(rain50_logits, rain50, expanded_mask, gamma=focal_gamma),
            _balanced_focal_bce(wind20_logits, wind20, expanded_mask, gamma=focal_gamma),
            _balanced_focal_bce(storm_logits, storm, expanded_mask, gamma=focal_gamma),
        )
    ).mean()

    rain_event = rain50 & expanded_mask
    wind_event = wind20 & expanded_mask
    intensity_terms = []
    if rain_event.any():
        intensity_terms.append(
            ((pred_tp_6h[rain_event] - true_tp_6h[rain_event]) / 50.0).square().mean()
        )
    if wind_event.any():
        intensity_terms.append(
            ((pred_wind[wind_event] - true_wind[wind_event]) / 20.0).square().mean()
        )
    intensity = (
        torch.stack(intensity_terms).mean()
        if intensity_terms
        else prediction.sum() * 0.0
    )
    total = classification_weight * classification + intensity_weight * intensity
    components = {
        "event_loss": total,
        "event_classification_loss": classification,
        "event_intensity_loss": intensity,
        "event_positive_rain_cells": rain_event.sum().to(prediction.dtype),
        "event_positive_wind_cells": wind_event.sum().to(prediction.dtype),
        "event_positive_compound_cells": (storm & expanded_mask).sum().to(prediction.dtype),
    }
    return total, components


@dataclass
class RegionMetricAccumulator:
    """Streaming denormalized MAE/RMSE for full, land, and sea domains."""

    variable_names: tuple[str, ...]

    def __post_init__(self) -> None:
        self._absolute: dict[str, torch.Tensor] = {}
        self._squared: dict[str, torch.Tensor] = {}
        self._count: dict[str, int] = {}

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        land_sea_mask: torch.Tensor,
    ) -> None:
        if prediction.shape != target.shape or prediction.ndim != 5:
            raise ValueError("prediction and target must have shape [B, T, C, H, W]")
        masks = {
            "full": torch.ones_like(land_sea_mask, dtype=torch.bool),
            "land": land_sea_mask >= 0.5,
            "sea": land_sea_mask < 0.5,
        }
        error = prediction.detach().double().cpu() - target.detach().double().cpu()
        for region, mask in masks.items():
            selected = mask.detach().cpu().bool()[None, None, None]
            absolute = (error.abs() * selected).sum(dim=(0, 1, 3, 4))
            squared = (error.square() * selected).sum(dim=(0, 1, 3, 4))
            count = int(prediction.shape[0] * prediction.shape[1] * mask.sum().item())
            if region not in self._absolute:
                self._absolute[region] = torch.zeros_like(absolute)
                self._squared[region] = torch.zeros_like(squared)
                self._count[region] = 0
            self._absolute[region] += absolute
            self._squared[region] += squared
            self._count[region] += count

    def compute(self) -> dict[str, dict[str, dict[str, float]]]:
        output: dict[str, dict[str, dict[str, float]]] = {}
        for region in ("full", "land", "sea"):
            count = max(1, self._count.get(region, 0))
            mae = self._absolute[region] / count
            rmse = torch.sqrt(self._squared[region] / count)
            output[region] = {
                name: {"mae": float(mae[index]), "rmse": float(rmse[index])}
                for index, name in enumerate(self.variable_names)
            }
        return output


@dataclass
class ForecastMetricAccumulator:
    """Aggregate and lead-hour denormalized metrics for forecast tensors."""

    variable_names: tuple[str, ...]
    forecast_hours: int

    def __post_init__(self) -> None:
        if self.forecast_hours < 1:
            raise ValueError("forecast_hours must be positive")
        self._aggregate = RegionMetricAccumulator(self.variable_names)
        self._by_lead = [
            RegionMetricAccumulator(self.variable_names) for _ in range(self.forecast_hours)
        ]

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        land_sea_mask: torch.Tensor,
    ) -> None:
        if prediction.shape[1] != self.forecast_hours:
            raise ValueError("prediction lead dimension does not match forecast_hours")
        self._aggregate.update(prediction, target, land_sea_mask)
        for lead, accumulator in enumerate(self._by_lead):
            accumulator.update(
                prediction[:, lead : lead + 1],
                target[:, lead : lead + 1],
                land_sea_mask,
            )

    def compute(self) -> dict[str, object]:
        return {
            "aggregate": self._aggregate.compute(),
            "by_lead_hour": {
                str(lead + 1): accumulator.compute()
                for lead, accumulator in enumerate(self._by_lead)
            },
        }
