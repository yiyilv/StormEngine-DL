"""Sparse point observations to a regular latent grid."""

from __future__ import annotations

import torch
from torch import nn


class SetConvEncoder(nn.Module):
    """Encode a sequence of irregular station sets onto a fixed grid.

    Args:
        point_values: ``[B, T, N, C]`` normalized station variables.
        point_coords: ``[B, N, 2]`` normalized ``(lat, lon)`` coordinates.
        point_mask: ``[B, T, N]`` where one marks an available station.

    Returns:
        Latent grids with shape ``[B, T, latent_channels, H, W]``.
    """

    def __init__(
        self,
        value_channels: int,
        point_hidden: int,
        latent_channels: int,
        height: int,
        width: int,
        sigma: float = 0.10,
        point_static_channels: int = 0,
    ) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.sigma = sigma
        self.point_static_channels = point_static_channels

        self.point_mlp = nn.Sequential(
            nn.Linear(value_channels + 2 + point_static_channels, point_hidden),
            nn.ReLU(),
            nn.Linear(point_hidden, latent_channels),
            nn.ReLU(),
        )

        grid_lat = torch.linspace(0.0, 1.0, height)
        grid_lon = torch.linspace(0.0, 1.0, width)
        lat, lon = torch.meshgrid(grid_lat, grid_lon, indexing="ij")
        self.register_buffer("grid_coords", torch.stack((lat, lon), dim=-1))

    def forward(
        self,
        point_values: torch.Tensor,
        point_coords: torch.Tensor,
        point_mask: torch.Tensor | None = None,
        point_static: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, steps, stations, _ = point_values.shape
        if point_coords.shape != (batch, stations, 2):
            raise ValueError("point_coords must have shape [B, N, 2]")

        coords = point_coords[:, None].expand(-1, steps, -1, -1)
        feature_parts = [coords, point_values]
        if self.point_static_channels and point_static is None:
            raise ValueError("point_static is required when point_static_channels is positive")
        if point_static is not None:
            if point_static.shape != (batch, stations, self.point_static_channels):
                raise ValueError(
                    f"point_static must have shape [B, N, {self.point_static_channels}]"
                )
            feature_parts.append(point_static[:, None].expand(-1, steps, -1, -1))
        features = self.point_mlp(torch.cat(feature_parts, dim=-1))

        delta = coords[:, :, :, None, None, :] - self.grid_coords[None, None, None]
        distance_sq = delta.square().sum(dim=-1)
        weights = torch.exp(-distance_sq / (2.0 * self.sigma**2))

        if point_mask is not None:
            weights = weights * point_mask[:, :, :, None, None].to(weights.dtype)

        numerator = torch.einsum("btnhw,btnc->btchw", weights, features)
        denominator = weights.sum(dim=2, keepdim=False).clamp_min(1e-8)
        return numerator / denominator[:, :, None]
