"""Encoder-decoder model for isolating sparse spatial reconstruction skill."""

from __future__ import annotations

import torch
from torch import nn

from .decoder import FieldDecoder
from .encoder import SetConvEncoder


class StormEngineReconstructionModel(nn.Module):
    """Map one sparse station snapshot directly to a dense current-time grid."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        point_hidden: int = 64,
        latent_channels: int = 64,
        height: int = 31,
        width: int = 33,
        sigma: float = 0.10,
        static_channels: int = 0,
        point_static_channels: int = 0,
    ) -> None:
        super().__init__()
        self.encoder = SetConvEncoder(
            input_channels,
            point_hidden,
            latent_channels,
            height,
            width,
            sigma,
            point_static_channels,
        )
        self.decoder = FieldDecoder(latent_channels, output_channels, static_channels)

    def forward(
        self,
        point_values: torch.Tensor,
        point_coords: torch.Tensor,
        point_mask: torch.Tensor | None = None,
        static_fields: torch.Tensor | None = None,
        point_static: torch.Tensor | None = None,
    ) -> torch.Tensor:
        encoded = self.encoder(point_values, point_coords, point_mask, point_static)
        return self.decoder(encoded, static_fields)
