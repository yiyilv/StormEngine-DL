"""Event-focused verification for gridded forecasts."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


@dataclass
class _EventStats:
    direction: str

    def __post_init__(self) -> None:
        if self.direction not in {"above", "below"}:
            raise ValueError("direction must be 'above' or 'below'")
        self.hits = 0
        self.misses = 0
        self.false_alarms = 0
        self.correct_negatives = 0
        self.event_squared_error = 0.0
        self.event_cells = 0
        self.peak_bias_sum = 0.0
        self.peak_cases = 0

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        region_mask: torch.Tensor,
        threshold: float,
    ) -> None:
        if prediction.shape != target.shape or prediction.ndim != 3:
            raise ValueError("prediction and target must have shape [B, T, P]")
        if region_mask.ndim != 1 or region_mask.shape[0] != prediction.shape[-1]:
            raise ValueError("region_mask must have shape [P]")
        prediction = prediction.detach().double().cpu()
        target = target.detach().double().cpu()
        selected_prediction = prediction[:, :, region_mask]
        selected_target = target[:, :, region_mask]
        if self.direction == "above":
            predicted_event = selected_prediction >= threshold
            observed_event = selected_target >= threshold
        else:
            predicted_event = selected_prediction <= threshold
            observed_event = selected_target <= threshold
        self.hits += int((predicted_event & observed_event).sum())
        self.misses += int((~predicted_event & observed_event).sum())
        self.false_alarms += int((predicted_event & ~observed_event).sum())
        self.correct_negatives += int((~predicted_event & ~observed_event).sum())
        error = selected_prediction - selected_target
        self.event_squared_error += float((error.square() * observed_event).sum())
        self.event_cells += int(observed_event.sum())

        cases = observed_event.any(dim=-1)
        if cases.any():
            if self.direction == "above":
                predicted_peak = selected_prediction.amax(dim=-1)
                observed_peak = selected_target.amax(dim=-1)
            else:
                predicted_peak = selected_prediction.amin(dim=-1)
                observed_peak = selected_target.amin(dim=-1)
            self.peak_bias_sum += float(((predicted_peak - observed_peak) * cases).sum())
            self.peak_cases += int(cases.sum())

    def compute(self) -> dict[str, float | int | None]:
        detected = self.hits + self.false_alarms
        observed = self.hits + self.misses
        union = self.hits + self.misses + self.false_alarms
        event_rmse = (
            None
            if self.event_cells == 0
            else float((self.event_squared_error / self.event_cells) ** 0.5)
        )
        peak_bias = (
            None if self.peak_cases == 0 else float(self.peak_bias_sum / self.peak_cases)
        )
        return {
            "hits": self.hits,
            "misses": self.misses,
            "false_alarms": self.false_alarms,
            "correct_negatives": self.correct_negatives,
            "observed_event_cells": self.event_cells,
            "pod": _safe_ratio(self.hits, observed),
            "far": _safe_ratio(self.false_alarms, detected),
            "csi": _safe_ratio(self.hits, union),
            "event_conditioned_rmse": event_rmse,
            "peak_intensity_bias": peak_bias,
            "peak_cases": self.peak_cases,
        }


class EventMetricAccumulator:
    """Aggregate and lead-hour event metrics over one fixed spatial region."""

    def __init__(self, forecast_hours: int, *, direction: str, threshold: float) -> None:
        if forecast_hours < 1:
            raise ValueError("forecast_hours must be positive")
        if not torch.isfinite(torch.tensor(threshold)):
            raise ValueError("threshold must be finite")
        self.forecast_hours = forecast_hours
        self.direction = direction
        self.threshold = float(threshold)
        self.aggregate = _EventStats(direction)
        self.by_lead = [_EventStats(direction) for _ in range(forecast_hours)]

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        region_mask: torch.Tensor,
    ) -> None:
        if prediction.shape != target.shape or prediction.ndim != 4:
            raise ValueError("prediction and target must have shape [B, T, H, W]")
        if prediction.shape[1] != self.forecast_hours:
            raise ValueError("prediction lead dimension does not match forecast_hours")
        flat_prediction = prediction.flatten(start_dim=2)
        flat_target = target.flatten(start_dim=2)
        flat_mask = region_mask.detach().cpu().bool().flatten()
        self.aggregate.update(flat_prediction, flat_target, flat_mask, self.threshold)
        for lead, accumulator in enumerate(self.by_lead):
            accumulator.update(
                flat_prediction[:, lead : lead + 1],
                flat_target[:, lead : lead + 1],
                flat_mask,
                self.threshold,
            )

    def compute(self) -> dict[str, object]:
        return {
            "threshold": self.threshold,
            "direction": self.direction,
            "aggregate": self.aggregate.compute(),
            "by_lead_hour": {
                str(index + 1): value.compute()
                for index, value in enumerate(self.by_lead)
            },
        }
