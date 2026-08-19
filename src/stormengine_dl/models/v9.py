"""V9 controlled forecast forms built on the frozen V7-B data contract."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .decoder import FieldDecoder
from .mask_aware import MaskAwareSetConvEncoder
from .processor import ConvGRUProcessor, DirectHorizonConvGRUProcessor


V9_MODEL_CONTRACT = "stormengine-v9-output-form-v1"
VALID_TEMPORAL_MODES = {"autoregressive", "direct"}
VALID_OUTPUT_MODES = {"field", "residual"}


class StormEngineV9ForecastModel(nn.Module):
    """Forecast absolute fields or increments with either temporal formulation."""

    contract_version = V9_MODEL_CONTRACT

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        forecast_steps: int,
        *,
        temporal_mode: str,
        output_mode: str,
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
        if temporal_mode not in VALID_TEMPORAL_MODES:
            raise ValueError(f"Unsupported temporal_mode: {temporal_mode}")
        if output_mode not in VALID_OUTPUT_MODES:
            raise ValueError(f"Unsupported output_mode: {output_mode}")
        self.temporal_mode = temporal_mode
        self.output_mode = output_mode
        self.forecast_steps = forecast_steps
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
        if temporal_mode == "autoregressive":
            self.processor: nn.Module = ConvGRUProcessor(
                latent_channels, processor_layers, kernel_size
            )
        else:
            self.processor = DirectHorizonConvGRUProcessor(
                latent_channels,
                forecast_steps,
                processor_layers,
                kernel_size,
            )
        self.decoder = FieldDecoder(latent_channels, output_channels, static_channels)
        self.reconstruction_decoder = (
            FieldDecoder(latent_channels, output_channels, static_channels)
            if output_mode == "residual"
            else None
        )

    def forward_with_reconstruction(
        self,
        point_values: torch.Tensor,
        point_coords: torch.Tensor,
        value_mask: torch.Tensor,
        forecast_steps: int,
        *,
        observation_age: torch.Tensor | None = None,
        static_fields: torch.Tensor | None = None,
        point_static: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if forecast_steps != self.forecast_steps:
            raise ValueError(
                f"V9 model was built for {self.forecast_steps} forecast steps, not {forecast_steps}"
            )
        encoded = self.encoder(
            point_values, point_coords, value_mask, observation_age, point_static
        )
        future = self.processor(encoded, forecast_steps)
        forecast_component = self.decoder(future, static_fields)
        if self.output_mode == "field":
            return forecast_component, None
        assert self.reconstruction_decoder is not None
        current = self.reconstruction_decoder(encoded[:, -1:], static_fields)
        return current + forecast_component, current[:, 0]

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        prediction, _ = self.forward_with_reconstruction(*args, **kwargs)
        return prediction


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def warm_start_from_v7b(
    model: StormEngineV9ForecastModel,
    checkpoint_path: Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, object]:
    """Load only shape-compatible V7-B tensors and record the exact transfer."""

    actual_sha256 = sha256(checkpoint_path)
    if expected_sha256 and actual_sha256 != expected_sha256:
        raise ValueError(
            f"V7-B checkpoint SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = checkpoint["model_state_dict"]
    target = model.state_dict()
    loaded: list[str] = []
    skipped: list[str] = []
    for key, value in source.items():
        if key in target and target[key].shape == value.shape:
            target[key] = value
            loaded.append(key)
        else:
            skipped.append(key)
    if model.reconstruction_decoder is not None:
        for key, value in source.items():
            if not key.startswith("decoder."):
                continue
            target_key = "reconstruction_decoder." + key.removeprefix("decoder.")
            if target_key in target and target[target_key].shape == value.shape:
                target[target_key] = value
                loaded.append(f"{key}->{target_key}")
    model.load_state_dict(target)
    return {
        "checkpoint": str(checkpoint_path),
        "sha256": actual_sha256,
        "loaded_tensor_count": len(loaded),
        "loaded_tensors": loaded,
        "skipped_source_tensors": skipped,
    }
