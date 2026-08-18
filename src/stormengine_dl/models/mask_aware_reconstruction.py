"""Mask-aware spatial pretraining for the V8 forecast pipeline."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

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


def freeze_spatial_modules(forecast_model: StormEngineV7ForecastModel) -> tuple[str, ...]:
    """Freeze pretrained spatial modules and expose only Processor parameters.

    Stage 2 deliberately optimizes the temporal Processor while preserving the
    exact Stage-1 Encoder/Decoder solution.  Returning the trainable parameter
    names makes that contract easy to assert before an expensive run.
    """

    forecast_model.encoder.requires_grad_(False)
    forecast_model.decoder.requires_grad_(False)
    forecast_model.processor.requires_grad_(True)
    forecast_model.encoder.eval()
    forecast_model.decoder.eval()
    return tuple(
        name for name, parameter in forecast_model.named_parameters()
        if parameter.requires_grad
    )


def set_processor_only_training_mode(
    forecast_model: StormEngineV7ForecastModel, training: bool
) -> None:
    """Set Stage-2 mode without re-enabling training mode on frozen modules."""

    forecast_model.train(training)
    if training:
        forecast_model.encoder.eval()
        forecast_model.decoder.eval()


def restore_spatial_training_checkpoint(
    resume_path: Path,
    output: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    contract: dict[str, object],
    device: torch.device,
) -> tuple[int, float, int, list[dict[str, float | int]]]:
    """Restore full training state and preserve the earlier selected model.

    A continuation may write to a new directory to keep a published run
    immutable. In that case the source run's selected ``best.pt`` is copied
    alongside the new continuation before optimization resumes.
    """

    resume_path = resume_path.resolve()
    if not resume_path.is_file():
        raise FileNotFoundError(resume_path)
    saved = torch.load(resume_path, map_location=device, weights_only=False)
    if saved.get("model_contract") != contract:
        raise ValueError("Resume checkpoint contract is incompatible")

    source_best = resume_path.parent / "best.pt"
    if not source_best.is_file():
        raise FileNotFoundError(
            f"Resume requires the selected checkpoint beside last.pt: {source_best}"
        )
    selected = torch.load(source_best, map_location="cpu", weights_only=False)
    if selected.get("model_contract") != contract:
        raise ValueError("Selected checkpoint beside resume file is incompatible")
    if float(selected["best_validation_loss"]) != float(saved["best_validation_loss"]):
        raise ValueError("best.pt and resume checkpoint disagree on best validation loss")

    destination_best = output.resolve() / "best.pt"
    if source_best.resolve() != destination_best:
        if destination_best.exists():
            raise FileExistsError(
                f"Refusing to mix continuation with an existing checkpoint: {destination_best}"
            )
        shutil.copy2(source_best, destination_best)

    model.load_state_dict(saved["model_state_dict"])
    optimizer.load_state_dict(saved["optimizer_state_dict"])
    scheduler.load_state_dict(saved["scheduler_state_dict"])
    scaler.load_state_dict(saved["scaler_state_dict"])
    history = list(saved["history"])
    epoch = int(saved["epoch"])
    if len(history) != epoch:
        raise ValueError("Resume history length does not match its completed epoch")
    return (
        epoch,
        float(saved["best_validation_loss"]),
        int(saved["epochs_without_improvement"]),
        history,
    )
