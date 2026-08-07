"""End-to-end StormEngine forecasting model."""

from __future__ import annotations

import torch
from torch import nn

from .decoder import FieldDecoder
from .encoder import SetConvEncoder
from .processor import ConvGRUProcessor


class StormEngineForecastModel(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        point_hidden: int = 64,
        latent_channels: int = 64,
        height: int = 31,
        width: int = 33,
        sigma: float = 0.10,
        processor_layers: int = 2,
        kernel_size: int = 3,
        static_channels: int = 0,
    ) -> None:
        super().__init__()
        self.encoder = SetConvEncoder(
            value_channels=input_channels,
            point_hidden=point_hidden,
            latent_channels=latent_channels,
            height=height,
            width=width,
            sigma=sigma,
        )
        self.processor = ConvGRUProcessor(
            channels=latent_channels,
            layers=processor_layers,
            kernel_size=kernel_size,
        )
        self.decoder = FieldDecoder(
            latent_channels=latent_channels,
            output_channels=output_channels,
            static_channels=static_channels,
        )

    def forward(
        self,
        point_values: torch.Tensor,
        point_coords: torch.Tensor,
        forecast_steps: int,
        point_mask: torch.Tensor | None = None,
        static_fields: torch.Tensor | None = None,
    ) -> torch.Tensor:
        encoded = self.encoder(point_values, point_coords, point_mask)
        future = self.processor(encoded, forecast_steps)
        return self.decoder(future, static_fields)

