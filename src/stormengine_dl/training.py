"""Training and evaluation helpers for StormEngine V6 experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch


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
