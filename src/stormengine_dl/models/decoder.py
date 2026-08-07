"""Decode future latent grids into physical-variable grids."""

from __future__ import annotations

import torch
from torch import nn


class FieldDecoder(nn.Module):
    def __init__(
        self,
        latent_channels: int,
        output_channels: int,
        static_channels: int = 0,
    ) -> None:
        super().__init__()
        in_channels = latent_channels + static_channels
        hidden = max(32, latent_channels)
        self.static_channels = static_channels
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, output_channels, 1),
        )

    def forward(
        self,
        future_latents: torch.Tensor,
        static_fields: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, steps, channels, height, width = future_latents.shape
        values = future_latents.reshape(batch * steps, channels, height, width)

        if self.static_channels:
            if static_fields is None:
                raise ValueError("static_fields are required by this decoder")
            static = static_fields[:, None].expand(-1, steps, -1, -1, -1)
            static = static.reshape(batch * steps, self.static_channels, height, width)
            values = torch.cat((values, static), dim=1)

        output = self.network(values)
        return output.reshape(batch, steps, output.shape[1], height, width)

