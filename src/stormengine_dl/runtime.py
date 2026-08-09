"""Shared experiment construction and tensor helpers for training and evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .data import CachedEra5SequenceDataset, Era5SequenceDataset, NormalizationStats
from .models import StormEngineForecastModel


def resolve_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def training_cache_dir(repo_root: Path, data: dict[str, Any]) -> Path | None:
    if not data.get("training_cache"):
        return None
    value = Path(data["training_cache"]).expanduser()
    return value.resolve() if value.is_absolute() else resolve_path(repo_root, data["era5_root"]) / value


def require_training_cache(repo_root: Path, data: dict[str, Any], config_path: Path) -> Path | None:
    cache_dir = training_cache_dir(repo_root, data)
    if cache_dir is not None and not (cache_dir / "metadata.json").is_file():
        raise FileNotFoundError(
            f"Training cache is missing at {cache_dir}. Run scripts/build_training_cache.py "
            f"--config {config_path} first."
        )
    return cache_dir


def make_dataset(
    repo_root: Path,
    data: dict[str, Any],
    years: list[int],
    dropout: float = 0.0,
) -> CachedEra5SequenceDataset | Era5SequenceDataset:
    cache_dir = training_cache_dir(repo_root, data)
    if cache_dir is not None and (cache_dir / "metadata.json").is_file():
        return CachedEra5SequenceDataset(
            cache_dir,
            years=years,
            history_hours=int(data["history_hours"]),
            forecast_hours=int(data["forecast_hours"]),
            window_stride_hours=int(data.get("window_stride_hours", 1)),
            station_dropout=dropout,
            input_variables=data["input_variables"],
            target_variables=data["target_variables"],
        )
    return Era5SequenceDataset.from_station_registry(
        manifest_path=resolve_path(repo_root, data["era5_manifest"]),
        data_root=resolve_path(repo_root, data["era5_root"]),
        station_registry_path=resolve_path(repo_root, data["station_registry"]),
        station_profile=data["station_profile"],
        input_variables=data["input_variables"],
        target_variables=data["target_variables"],
        history_hours=int(data["history_hours"]),
        forecast_hours=int(data["forecast_hours"]),
        window_stride_hours=int(data.get("window_stride_hours", 1)),
        cache_months=int(data.get("cache_months", 2)),
        years=years,
        station_dropout=dropout,
        normalization_path=resolve_path(repo_root, data["normalization_stats"]),
    )


def make_model(config: dict[str, Any]) -> StormEngineForecastModel:
    data, domain, model = config["data"], config["domain"], config["model"]
    return StormEngineForecastModel(
        input_channels=len(data["input_variables"]),
        output_channels=len(data["target_variables"]),
        point_hidden=int(model["point_hidden"]),
        latent_channels=int(model["latent_channels"]),
        height=int(domain["height"]),
        width=int(domain["width"]),
        sigma=float(model["gaussian_sigma"]),
        processor_layers=int(model["processor_layers"]),
        kernel_size=int(model["kernel_size"]),
        static_channels=int(model["static_channels"]),
        point_static_channels=int(model["point_static_channels"]),
    )


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        # CUDA is the production path. CPU is a predictable fallback; MPS can
        # still be selected explicitly on a compatible macOS installation.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this PyTorch installation")
    return device


def move_batch(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=device.type == "cuda")
        for key, value in batch.items()
        if key != "start_index"
    }


def forecast(
    model: StormEngineForecastModel,
    batch: dict[str, torch.Tensor],
    static_fields: torch.Tensor,
) -> torch.Tensor:
    return model(
        batch["point_values"],
        batch["point_coords"],
        forecast_steps=batch["target"].shape[1],
        point_mask=batch["point_mask"],
        static_fields=static_fields.expand(batch["target"].shape[0], -1, -1, -1),
        point_static=batch["point_static"],
    )


def denormalize_channels(
    values: torch.Tensor,
    variables: list[str],
    stats: NormalizationStats,
) -> torch.Tensor:
    result = values.detach().float().cpu().clone()
    for channel, variable in enumerate(variables):
        stat = stats.variables[variable]
        result[:, :, channel] = result[:, :, channel] * stat.std + stat.mean
    return result
