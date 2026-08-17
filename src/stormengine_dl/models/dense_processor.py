"""Processor-only dense-grid forecasting models."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .processor import ConvGRUProcessor


class FactorizedAttentionBlock(nn.Module):
    """Apply temporal then spatial self-attention to ``[B,T,P,D]`` tokens."""

    def __init__(self, dimension: int, heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        hidden = int(round(dimension * mlp_ratio))
        self.temporal_norm = nn.LayerNorm(dimension)
        self.temporal_attention = nn.MultiheadAttention(
            dimension, heads, dropout=dropout, batch_first=True
        )
        self.spatial_norm = nn.LayerNorm(dimension)
        self.spatial_attention = nn.MultiheadAttention(
            dimension, heads, dropout=dropout, batch_first=True
        )
        self.feed_forward_norm = nn.LayerNorm(dimension)
        self.feed_forward = nn.Sequential(
            nn.Linear(dimension, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dimension),
            nn.Dropout(dropout),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, steps, patches, dimension = values.shape
        temporal = values.permute(0, 2, 1, 3).reshape(batch * patches, steps, dimension)
        normalized = self.temporal_norm(temporal)
        temporal = temporal + self.temporal_attention(
            normalized, normalized, normalized, need_weights=False
        )[0]
        values = temporal.reshape(batch, patches, steps, dimension).permute(0, 2, 1, 3)

        spatial = values.reshape(batch * steps, patches, dimension)
        normalized = self.spatial_norm(spatial)
        spatial = spatial + self.spatial_attention(
            normalized, normalized, normalized, need_weights=False
        )[0]
        values = spatial.reshape(batch, steps, patches, dimension)
        return values + self.feed_forward(self.feed_forward_norm(values))


class FactorizedViTProcessor(nn.Module):
    """Patch-based temporal/spatial Transformer with direct multi-step output."""

    def __init__(
        self,
        channels: int,
        *,
        height: int,
        width: int,
        history_steps: int,
        forecast_steps: int,
        patch_size: int = 4,
        dimension: int = 128,
        layers: int = 4,
        heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dimension % heads:
            raise ValueError("Transformer dimension must be divisible by attention heads")
        if min(height, width, history_steps, forecast_steps, patch_size, layers) < 1:
            raise ValueError("grid, sequence, patch, and layer sizes must be positive")
        self.channels = channels
        self.height = height
        self.width = width
        self.history_steps = history_steps
        self.forecast_steps = forecast_steps
        self.patch_size = patch_size
        self.padded_height = math.ceil(height / patch_size) * patch_size
        self.padded_width = math.ceil(width / patch_size) * patch_size
        self.patch_rows = self.padded_height // patch_size
        self.patch_columns = self.padded_width // patch_size
        patch_count = self.patch_rows * self.patch_columns

        self.patch_embedding = nn.Conv2d(
            channels, dimension, kernel_size=patch_size, stride=patch_size
        )
        self.spatial_embedding = nn.Parameter(torch.zeros(1, 1, patch_count, dimension))
        self.history_embedding = nn.Parameter(torch.zeros(1, history_steps, 1, dimension))
        self.future_embedding = nn.Parameter(torch.zeros(1, forecast_steps, 1, dimension))
        self.history_blocks = nn.ModuleList(
            FactorizedAttentionBlock(dimension, heads, mlp_ratio, dropout)
            for _ in range(layers)
        )
        self.future_blocks = nn.ModuleList(
            FactorizedAttentionBlock(dimension, heads, mlp_ratio, dropout)
            for _ in range(max(1, layers // 2))
        )
        self.output_norm = nn.LayerNorm(dimension)
        self.patch_output = nn.Linear(dimension, channels * patch_size * patch_size)
        nn.init.trunc_normal_(self.spatial_embedding, std=0.02)
        nn.init.trunc_normal_(self.history_embedding, std=0.02)
        nn.init.trunc_normal_(self.future_embedding, std=0.02)

    def _unpatch(self, values: torch.Tensor) -> torch.Tensor:
        batch, steps, _, _ = values.shape
        patch = self.patch_size
        values = self.patch_output(self.output_norm(values))
        values = values.reshape(
            batch, steps, self.patch_rows, self.patch_columns,
            self.channels, patch, patch,
        )
        values = values.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        values = values.reshape(
            batch, steps, self.channels, self.padded_height, self.padded_width
        )
        return values[..., : self.height, : self.width]

    def forward(self, encoded_history: torch.Tensor, forecast_steps: int) -> torch.Tensor:
        batch, history, channels, height, width = encoded_history.shape
        if (history, channels, height, width) != (
            self.history_steps, self.channels, self.height, self.width
        ):
            raise ValueError("encoded history does not match the ViT Processor contract")
        if forecast_steps != self.forecast_steps:
            raise ValueError("forecast_steps does not match the configured ViT horizon")
        pad = (0, self.padded_width - width, 0, self.padded_height - height)
        values = F.pad(encoded_history.reshape(batch * history, channels, height, width), pad)
        values = self.patch_embedding(values).flatten(2).transpose(1, 2)
        values = values.reshape(batch, history, -1, values.shape[-1])
        values = values + self.spatial_embedding + self.history_embedding
        for block in self.history_blocks:
            values = block(values)

        # The final history tokens contain the complete temporal context after
        # temporal attention. Future embeddings then represent +1 ... +6 h.
        values = values[:, -1:, :, :] + self.future_embedding
        for block in self.future_blocks:
            values = block(values)
        return self._unpatch(values)


class DenseProcessorForecastModel(nn.Module):
    """Common input/output projections around an interchangeable Processor."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        *,
        latent_channels: int,
        processor: nn.Module,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Conv2d(input_channels, latent_channels, kernel_size=1),
            nn.GELU(),
        )
        self.processor = processor
        self.output_projection = nn.Conv2d(latent_channels, output_channels, kernel_size=1)

    def forward(self, history: torch.Tensor, forecast_steps: int) -> torch.Tensor:
        batch, steps, channels, height, width = history.shape
        encoded = self.input_projection(history.reshape(batch * steps, channels, height, width))
        encoded = encoded.reshape(batch, steps, encoded.shape[1], height, width)
        processed = self.processor(encoded, forecast_steps)
        output = self.output_projection(
            processed.reshape(batch * forecast_steps, processed.shape[2], height, width)
        )
        return output.reshape(batch, forecast_steps, output.shape[1], height, width)


def make_dense_processor_model(
    family: str,
    *,
    input_channels: int,
    output_channels: int,
    latent_channels: int,
    height: int,
    width: int,
    history_steps: int,
    forecast_steps: int,
    processor_layers: int = 2,
    kernel_size: int = 3,
    patch_size: int = 4,
    transformer_dimension: int = 128,
    transformer_heads: int = 4,
    transformer_mlp_ratio: float = 4.0,
    dropout: float = 0.0,
) -> DenseProcessorForecastModel:
    if family == "convgru":
        processor: nn.Module = ConvGRUProcessor(
            latent_channels, layers=processor_layers, kernel_size=kernel_size
        )
    elif family == "factorized_vit":
        processor = FactorizedViTProcessor(
            latent_channels,
            height=height,
            width=width,
            history_steps=history_steps,
            forecast_steps=forecast_steps,
            patch_size=patch_size,
            dimension=transformer_dimension,
            layers=processor_layers,
            heads=transformer_heads,
            mlp_ratio=transformer_mlp_ratio,
            dropout=dropout,
        )
    else:
        raise ValueError(f"unknown Processor family: {family}")
    return DenseProcessorForecastModel(
        input_channels, output_channels,
        latent_channels=latent_channels, processor=processor,
    )
