"""Mask-aware V7 spatial encoder and forecast model."""

from __future__ import annotations

import torch
from torch import nn

from .decoder import FieldDecoder
from .processor import ConvGRUProcessor


V7_MODEL_CONTRACT = "stormengine-v7-mask-aware-v1"


class MaskAwareSetConvEncoder(nn.Module):
    """Encode values, per-variable masks, and optional normalized ages."""

    def __init__(
        self,
        value_channels: int,
        point_hidden: int,
        latent_channels: int,
        height: int,
        width: int,
        *,
        include_age: bool,
        sigma: float = 0.10,
        point_static_channels: int = 0,
    ) -> None:
        super().__init__()
        self.value_channels = value_channels
        self.include_age = include_age
        self.height = height
        self.width = width
        self.sigma = sigma
        self.point_static_channels = point_static_channels
        feature_channels = 2 + 2 * value_channels + point_static_channels
        if include_age:
            feature_channels += value_channels
        self.point_mlp = nn.Sequential(
            nn.Linear(feature_channels, point_hidden),
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
        value_mask: torch.Tensor,
        observation_age: torch.Tensor | None = None,
        point_static: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, steps, stations, channels = point_values.shape
        expected = (batch, steps, stations, channels)
        if channels != self.value_channels:
            raise ValueError(f"Expected {self.value_channels} value channels, got {channels}")
        if value_mask.shape != expected:
            raise ValueError("value_mask must have the same shape as point_values")
        if point_coords.shape != (batch, stations, 2):
            raise ValueError("point_coords must have shape [B, N, 2]")
        if self.include_age and observation_age is None:
            raise ValueError("observation_age is required by this V7 contract")
        if observation_age is not None and observation_age.shape != expected:
            raise ValueError("observation_age must have the same shape as point_values")
        coords = point_coords[:, None].expand(-1, steps, -1, -1)
        mask = value_mask.to(point_values.dtype)
        feature_parts = [coords, point_values * mask, mask]
        if self.include_age:
            assert observation_age is not None
            feature_parts.append(observation_age * mask)
        if self.point_static_channels and point_static is None:
            raise ValueError("point_static is required by this V7 contract")
        if point_static is not None:
            if point_static.shape != (batch, stations, self.point_static_channels):
                raise ValueError(
                    f"point_static must have shape [B, N, {self.point_static_channels}]"
                )
            feature_parts.append(point_static[:, None].expand(-1, steps, -1, -1))
        features = self.point_mlp(torch.cat(feature_parts, dim=-1))
        delta = coords[:, :, :, None, None, :] - self.grid_coords[None, None, None]
        weights = torch.exp(-delta.square().sum(dim=-1) / (2.0 * self.sigma**2))
        station_present = value_mask.any(dim=-1)
        weights = weights * station_present[:, :, :, None, None].to(weights.dtype)
        numerator = torch.einsum("btnhw,btnc->btchw", weights, features)
        denominator = weights.sum(dim=2).clamp_min(1e-8)
        return numerator / denominator[:, :, None]


class StormEngineV7ForecastModel(nn.Module):
    contract_version = V7_MODEL_CONTRACT

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        *,
        include_age: bool,
        point_hidden: int = 64,
        latent_channels: int = 64,
        height: int = 31,
        width: int = 33,
        sigma: float = 0.10,
        processor_layers: int = 2,
        kernel_size: int = 3,
        static_channels: int = 0,
        point_static_channels: int = 0,
    ) -> None:
        super().__init__()
        self.include_age = include_age
        self.encoder = MaskAwareSetConvEncoder(
            input_channels,
            point_hidden,
            latent_channels,
            height,
            width,
            include_age=include_age,
            sigma=sigma,
            point_static_channels=point_static_channels,
        )
        self.processor = ConvGRUProcessor(latent_channels, processor_layers, kernel_size)
        self.decoder = FieldDecoder(latent_channels, output_channels, static_channels)

    def forward(
        self,
        point_values: torch.Tensor,
        point_coords: torch.Tensor,
        value_mask: torch.Tensor,
        forecast_steps: int,
        *,
        observation_age: torch.Tensor | None = None,
        static_fields: torch.Tensor | None = None,
        point_static: torch.Tensor | None = None,
    ) -> torch.Tensor:
        encoded = self.encoder(
            point_values, point_coords, value_mask, observation_age, point_static
        )
        future = self.processor(encoded, forecast_steps)
        return self.decoder(future, static_fields)


def require_v7_checkpoint_contract(
    checkpoint: dict[str, object], expected_contract: dict[str, object]
) -> None:
    actual = checkpoint.get("model_contract")
    if actual != expected_contract:
        raise ValueError(
            "Checkpoint model contract is missing or incompatible with this V7 experiment"
        )
