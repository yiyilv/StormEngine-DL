"""Mask-aware spatial pretraining for the V8 forecast pipeline."""

from __future__ import annotations

import torch
from torch import nn

from .decoder import FieldDecoder
from .mask_aware import MaskAwareSetConvEncoder, StormEngineV7ForecastModel


V8_RECONSTRUCTION_CONTRACT = "stormengine-v8-mask-aware-reconstruction-v1"


class MaskAwareReconstructionModel(nn.Module):
    """Reconstruct a simultaneous dense grid without a temporal processor."""

    contract_version = V8_RECONSTRUCTION_CONTRACT

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
        static_channels: int = 0,
        point_static_channels: int = 0,
    ) -> None:
        super().__init__()
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
        self.decoder = FieldDecoder(latent_channels, output_channels, static_channels)

    def forward(
        self,
        point_values: torch.Tensor,
        point_coords: torch.Tensor,
        value_mask: torch.Tensor,
        *,
        observation_age: torch.Tensor | None = None,
        static_fields: torch.Tensor | None = None,
        point_static: torch.Tensor | None = None,
    ) -> torch.Tensor:
        encoded = self.encoder(
            point_values, point_coords, value_mask, observation_age, point_static
        )
        return self.decoder(encoded, static_fields)


def load_spatial_pretraining(
    forecast_model: StormEngineV7ForecastModel,
    checkpoint: dict[str, object],
    *,
    expected_contract: dict[str, object] | None = None,
) -> dict[str, object]:
    """Load only compatible Encoder/Decoder weights into a forecast model."""

    contract = checkpoint.get("model_contract")
    if not isinstance(contract, dict) or contract.get("version") != V8_RECONSTRUCTION_CONTRACT:
        raise ValueError("Checkpoint is not a V8 mask-aware reconstruction checkpoint")
    if expected_contract is not None and contract != expected_contract:
        raise ValueError("Reconstruction checkpoint contract is incompatible")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("Reconstruction checkpoint has no model_state_dict")
    encoder = {
        key.removeprefix("encoder."): value
        for key, value in state.items()
        if key.startswith("encoder.")
    }
    decoder = {
        key.removeprefix("decoder."): value
        for key, value in state.items()
        if key.startswith("decoder.")
    }
    if not encoder or not decoder:
        raise ValueError("Reconstruction checkpoint lacks Encoder/Decoder weights")
    forecast_model.encoder.load_state_dict(encoder, strict=True)
    forecast_model.decoder.load_state_dict(decoder, strict=True)
    return contract
