"""Event-focused verification for gridded forecasts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

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


@dataclass
class _BinaryCounts:
    """Categorical counts for one binary event definition."""

    def __post_init__(self) -> None:
        self.hits = 0
        self.misses = 0
        self.false_alarms = 0
        self.correct_negatives = 0

    def update(self, predicted: torch.Tensor, observed: torch.Tensor) -> None:
        if predicted.shape != observed.shape:
            raise ValueError("predicted and observed event masks must have the same shape")
        predicted = predicted.bool()
        observed = observed.bool()
        self.hits += int((predicted & observed).sum())
        self.misses += int((~predicted & observed).sum())
        self.false_alarms += int((predicted & ~observed).sum())
        self.correct_negatives += int((~predicted & ~observed).sum())

    def compute(self) -> dict[str, float | int | None]:
        detected = self.hits + self.false_alarms
        observed = self.hits + self.misses
        union = self.hits + self.misses + self.false_alarms
        return {
            "hits": self.hits,
            "misses": self.misses,
            "false_alarms": self.false_alarms,
            "correct_negatives": self.correct_negatives,
            "pod": _safe_ratio(self.hits, observed),
            "far": _safe_ratio(self.false_alarms, detected),
            "csi": _safe_ratio(self.hits, union),
        }


class _PhysicalWindowEventStats:
    """Grid-cell and forecast-case diagnostics for one six-hour event."""

    def __init__(self, component_names: Sequence[str]) -> None:
        if not component_names:
            raise ValueError("at least one diagnostic component is required")
        self.component_names = tuple(component_names)
        self.grid_cell = _BinaryCounts()
        self.forecast_case = _BinaryCounts()
        self.event_cells = 0
        self.event_cases = 0
        self.squared_error = {name: 0.0 for name in self.component_names}
        self.peak_bias_sum = {name: 0.0 for name in self.component_names}

    def update(
        self,
        predicted_event: torch.Tensor,
        observed_event: torch.Tensor,
        region_mask: torch.Tensor,
        components: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        if predicted_event.shape != observed_event.shape or predicted_event.ndim != 3:
            raise ValueError("window event masks must have shape [B, H, W]")
        if region_mask.shape != predicted_event.shape[-2:]:
            raise ValueError("region_mask must match the event spatial shape")
        if set(components) != set(self.component_names):
            raise ValueError("diagnostic components do not match the event definition")

        selected_region = region_mask.detach().cpu().bool().flatten()
        predicted = predicted_event.detach().cpu().bool().flatten(start_dim=1)[:, selected_region]
        observed = observed_event.detach().cpu().bool().flatten(start_dim=1)[:, selected_region]
        self.grid_cell.update(predicted, observed)
        predicted_cases = predicted.any(dim=1)
        observed_cases = observed.any(dim=1)
        self.forecast_case.update(predicted_cases, observed_cases)
        self.event_cells += int(observed.sum())
        self.event_cases += int(observed_cases.sum())

        for name, (prediction, target) in components.items():
            if prediction.shape != predicted_event.shape or target.shape != observed_event.shape:
                raise ValueError(f"component {name} must have shape [B, H, W]")
            selected_prediction = (
                prediction.detach().double().cpu().flatten(start_dim=1)[:, selected_region]
            )
            selected_target = target.detach().double().cpu().flatten(start_dim=1)[:, selected_region]
            error = selected_prediction - selected_target
            self.squared_error[name] += float((error.square() * observed).sum())
            if observed_cases.any():
                predicted_peak = selected_prediction.amax(dim=1)
                observed_peak = selected_target.amax(dim=1)
                self.peak_bias_sum[name] += float(
                    ((predicted_peak - observed_peak) * observed_cases).sum()
                )

    def compute(self) -> dict[str, object]:
        components: dict[str, dict[str, float | None]] = {}
        for name in self.component_names:
            components[name] = {
                "event_conditioned_rmse": (
                    None
                    if self.event_cells == 0
                    else float((self.squared_error[name] / self.event_cells) ** 0.5)
                ),
                "peak_intensity_bias": (
                    None
                    if self.event_cases == 0
                    else float(self.peak_bias_sum[name] / self.event_cases)
                ),
            }
        return {
            "grid_cell": self.grid_cell.compute(),
            "forecast_case": self.forecast_case.compute(),
            "observed_event_cells": self.event_cells,
            "observed_event_cases": self.event_cases,
            "components": components,
        }


class PhysicalSixHourEventAccumulator:
    """Evaluate the original StormEngine six-hour physical event definitions.

    Hourly precipitation is clipped to its physical lower bound and summed over
    the complete +1...+6 h forecast window. Wind speed is derived from u10/v10
    and reduced with the maximum over the same window. Metrics are computed over
    both grid cells and whole forecast cases within one fixed spatial region.
    """

    DEFAULT_THRESHOLDS = {
        "rain_6h_mm": 10.0,
        "storm_rain_6h_mm": 30.0,
        "extreme_rain_6h_mm": 50.0,
        "strong_wind_ms": 15.0,
        "extreme_wind_ms": 20.0,
    }

    def __init__(
        self,
        forecast_hours: int,
        *,
        thresholds: Mapping[str, float] | None = None,
        region_name: str = "sea",
    ) -> None:
        if forecast_hours != 6:
            raise ValueError("original physical event evaluation requires exactly six forecast hours")
        merged = dict(self.DEFAULT_THRESHOLDS)
        if thresholds is not None:
            unknown = set(thresholds) - set(merged)
            if unknown:
                raise ValueError(f"unknown physical event thresholds: {sorted(unknown)}")
            merged.update({name: float(value) for name, value in thresholds.items()})
        if not all(torch.isfinite(torch.tensor(value)) and value > 0 for value in merged.values()):
            raise ValueError("physical event thresholds must be finite and positive")
        self.forecast_hours = forecast_hours
        self.thresholds = merged
        self.region_name = region_name
        self.events = {
            "rain_6h_10mm": _PhysicalWindowEventStats(("tp_6h_mm",)),
            "heavy_rain_6h_30mm": _PhysicalWindowEventStats(("tp_6h_mm",)),
            "extreme_rain_6h_50mm": _PhysicalWindowEventStats(("tp_6h_mm",)),
            "strong_wind_6h_15ms": _PhysicalWindowEventStats(("max_wind_speed_ms",)),
            "extreme_wind_6h_20ms": _PhysicalWindowEventStats(("max_wind_speed_ms",)),
            "storm_any_6h": _PhysicalWindowEventStats(
                ("tp_6h_mm", "max_wind_speed_ms")
            ),
            "compound_storm_6h": _PhysicalWindowEventStats(
                ("tp_6h_mm", "max_wind_speed_ms")
            ),
            "extreme_weather_6h": _PhysicalWindowEventStats(
                ("tp_6h_mm", "max_wind_speed_ms")
            ),
        }

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        region_mask: torch.Tensor,
        variables: Sequence[str],
    ) -> None:
        if prediction.shape != target.shape or prediction.ndim != 5:
            raise ValueError("prediction and target must have shape [B, T, C, H, W]")
        if prediction.shape[1] != self.forecast_hours:
            raise ValueError("prediction lead dimension must contain +1...+6 h")
        if prediction.shape[2] != len(variables):
            raise ValueError("variable names do not match the channel dimension")
        required = {"u10", "v10", "tp"}
        if not required.issubset(variables):
            raise ValueError(f"physical events require variables {sorted(required)}")

        u10 = variables.index("u10")
        v10 = variables.index("v10")
        tp = variables.index("tp")
        predicted_tp = prediction[:, :, tp].clamp_min(0.0).sum(dim=1)
        observed_tp = target[:, :, tp].clamp_min(0.0).sum(dim=1)
        predicted_wind = torch.hypot(prediction[:, :, u10], prediction[:, :, v10]).amax(dim=1)
        observed_wind = torch.hypot(target[:, :, u10], target[:, :, v10]).amax(dim=1)

        rain = predicted_tp > self.thresholds["rain_6h_mm"]
        observed_rain = observed_tp > self.thresholds["rain_6h_mm"]
        heavy_rain = predicted_tp > self.thresholds["storm_rain_6h_mm"]
        observed_heavy_rain = observed_tp > self.thresholds["storm_rain_6h_mm"]
        extreme_rain = predicted_tp > self.thresholds["extreme_rain_6h_mm"]
        observed_extreme_rain = observed_tp > self.thresholds["extreme_rain_6h_mm"]
        strong_wind = predicted_wind > self.thresholds["strong_wind_ms"]
        observed_strong_wind = observed_wind > self.thresholds["strong_wind_ms"]
        extreme_wind = predicted_wind > self.thresholds["extreme_wind_ms"]
        observed_extreme_wind = observed_wind > self.thresholds["extreme_wind_ms"]

        precipitation = {"tp_6h_mm": (predicted_tp, observed_tp)}
        wind = {"max_wind_speed_ms": (predicted_wind, observed_wind)}
        combined = {**precipitation, **wind}
        updates = {
            "rain_6h_10mm": (rain, observed_rain, precipitation),
            "heavy_rain_6h_30mm": (heavy_rain, observed_heavy_rain, precipitation),
            "extreme_rain_6h_50mm": (extreme_rain, observed_extreme_rain, precipitation),
            "strong_wind_6h_15ms": (strong_wind, observed_strong_wind, wind),
            "extreme_wind_6h_20ms": (extreme_wind, observed_extreme_wind, wind),
            "storm_any_6h": (
                heavy_rain | strong_wind,
                observed_heavy_rain | observed_strong_wind,
                combined,
            ),
            "compound_storm_6h": (
                heavy_rain & strong_wind,
                observed_heavy_rain & observed_strong_wind,
                combined,
            ),
            "extreme_weather_6h": (
                extreme_rain | extreme_wind,
                observed_extreme_rain | observed_extreme_wind,
                combined,
            ),
        }
        for name, (predicted_event, observed_event, components) in updates.items():
            self.events[name].update(
                predicted_event, observed_event, region_mask, components
            )

    def compute(self) -> dict[str, object]:
        return {
            "forecast_window_hours": self.forecast_hours,
            "region": self.region_name,
            "threshold_operator": ">",
            "thresholds": dict(self.thresholds),
            "postprocessing": {
                "tp_6h_mm": "sum(max(tp_hourly_mm, 0)) over forecast leads +1...+6",
                "max_wind_speed_ms": "max(sqrt(u10^2 + v10^2)) over forecast leads +1...+6",
            },
            "definitions": {
                "rain_6h_10mm": "tp_6h_mm > 10",
                "heavy_rain_6h_30mm": "tp_6h_mm > 30",
                "extreme_rain_6h_50mm": "tp_6h_mm > 50",
                "strong_wind_6h_15ms": "max_wind_speed_ms > 15",
                "extreme_wind_6h_20ms": "max_wind_speed_ms > 20",
                "storm_any_6h": "heavy_rain_6h_30mm OR strong_wind_6h_15ms",
                "compound_storm_6h": "heavy_rain_6h_30mm AND strong_wind_6h_15ms",
                "extreme_weather_6h": "extreme_rain_6h_50mm OR extreme_wind_6h_20ms",
            },
            "events": {name: stats.compute() for name, stats in self.events.items()},
        }
