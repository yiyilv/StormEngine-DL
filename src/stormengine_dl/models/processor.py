"""Temporal processing for latent weather grids."""

from __future__ import annotations

import torch
from torch import nn


class ConvGRUCell(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.channels = channels
        self.gates = nn.Conv2d(2 * channels, 2 * channels, kernel_size, padding=padding)
        self.candidate = nn.Conv2d(2 * channels, channels, kernel_size, padding=padding)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        reset, update = torch.sigmoid(self.gates(torch.cat((x, hidden), dim=1))).chunk(2, dim=1)
        candidate = torch.tanh(self.candidate(torch.cat((x, reset * hidden), dim=1)))
        return (1.0 - update) * hidden + update * candidate


class ConvGRUProcessor(nn.Module):
    """Consume past latent grids and autoregressively produce future grids."""

    def __init__(self, channels: int, layers: int = 2, kernel_size: int = 3) -> None:
        super().__init__()
        self.channels = channels
        self.cells = nn.ModuleList(
            ConvGRUCell(channels, kernel_size=kernel_size) for _ in range(layers)
        )

    def _step(self, x: torch.Tensor, states: list[torch.Tensor]) -> list[torch.Tensor]:
        next_states: list[torch.Tensor] = []
        for cell, hidden in zip(self.cells, states):
            hidden = cell(x, hidden)
            next_states.append(hidden)
            x = hidden
        return next_states

    def forward(self, encoded_history: torch.Tensor, forecast_steps: int) -> torch.Tensor:
        batch, history, channels, height, width = encoded_history.shape
        if channels != self.channels:
            raise ValueError("encoded channel count does not match processor channels")

        states = [
            encoded_history.new_zeros(batch, channels, height, width)
            for _ in self.cells
        ]
        for index in range(history):
            states = self._step(encoded_history[:, index], states)

        predictions = []
        current = states[-1]
        for _ in range(forecast_steps):
            states = self._step(current, states)
            current = states[-1]
            predictions.append(current)
        return torch.stack(predictions, dim=1)

