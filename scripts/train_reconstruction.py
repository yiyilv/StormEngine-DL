#!/usr/bin/env python3
"""Train Encoder+Decoder on simultaneous sparse-to-grid reconstruction."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from stormengine_dl import StormEngineReconstructionModel
from stormengine_dl.data import (
    CachedEra5SequenceDataset,
    CachedReconstructionDataset,
    NormalizationStats,
    StaticFields,
)
from stormengine_dl.runtime import denormalize_channels, make_dataset, resolve_path, select_device
from stormengine_dl.training import RegionMetricAccumulator, sea_weight_map, weighted_mse


TARGETS = ["msl", "u10", "v10", "t2m"]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_reconstruction_dataset(
    repo_root: Path, data: dict[str, Any], years: list[int], dropout: float
) -> CachedReconstructionDataset:
    source = make_dataset(repo_root, data, years, dropout)
    if not isinstance(source, CachedEra5SequenceDataset):
        raise RuntimeError("reconstruction training requires the hourly memory-mapped cache")
    return CachedReconstructionDataset(source, TARGETS)


def _make_model(config: dict[str, Any]) -> StormEngineReconstructionModel:
    data, domain, model = config["data"], config["domain"], config["model"]
    return StormEngineReconstructionModel(
        input_channels=len(data["input_variables"]),
        output_channels=len(TARGETS),
        point_hidden=int(model["point_hidden"]),
        latent_channels=int(model["latent_channels"]),
        height=int(domain["height"]),
        width=int(domain["width"]),
        sigma=float(model["gaussian_sigma"]),
        static_channels=int(model["static_channels"]),
        point_static_channels=int(model["point_static_channels"]),
    )


def _move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=device.type == "cuda")
        for key, value in batch.items()
        if key != "start_index"
    }


def _forward(
    model: StormEngineReconstructionModel,
    batch: dict[str, torch.Tensor],
    static_fields: torch.Tensor,
) -> torch.Tensor:
    return model(
        batch["point_values"],
        batch["point_coords"],
        batch["point_mask"],
        static_fields.expand(batch["target"].shape[0], -1, -1, -1),
        batch["point_static"],
    )


def _loss_epoch(
    model: StormEngineReconstructionModel,
    loader: DataLoader[dict[str, torch.Tensor]],
    static_fields: torch.Tensor,
    weights: torch.Tensor,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    gradient_clip: float,
    max_batches: int | None,
    label: str,
    progress_every: int,
) -> float:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    completed = 0
    expected = min(len(loader), max_batches) if max_batches is not None else len(loader)
    started = time.time()
    for index, raw_batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        batch = _move(raw_batch, device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        amp = device.type == "cuda" and scaler.is_enabled()
        with torch.set_grad_enabled(training), torch.autocast(device_type=device.type, enabled=amp):
            loss = weighted_mse(_forward(model, batch, static_fields), batch["target"], weights)
        if optimizer is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        total += float(loss.detach())
        completed += 1
        if progress_every > 0 and (completed % progress_every == 0 or completed == expected):
            elapsed = time.time() - started
            eta = elapsed / completed * max(0, expected - completed)
            print(
                f"  {label}: {completed:,}/{expected:,} loss={total / completed:.6f} "
                f"elapsed={elapsed / 60:.1f}m ETA={eta / 60:.1f}m",
                flush=True,
            )
    if completed == 0:
        raise RuntimeError("data loader produced no batches")
    return total / completed


def _validation_metrics(
    model: StormEngineReconstructionModel,
    loader: DataLoader[dict[str, torch.Tensor]],
    static_fields: torch.Tensor,
    device: torch.device,
    normalization: NormalizationStats,
    max_batches: int | None,
) -> dict[str, object]:
    model.eval()
    accumulator = RegionMetricAccumulator(tuple(TARGETS))
    land_mask = static_fields[0, 0].cpu()
    with torch.no_grad():
        for index, raw_batch in enumerate(loader):
            if max_batches is not None and index >= max_batches:
                break
            batch = _move(raw_batch, device)
            accumulator.update(
                denormalize_channels(_forward(model, batch, static_fields), TARGETS, normalization),
                denormalize_channels(batch["target"], TARGETS, normalization),
                land_mask,
            )
    return accumulator.compute()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/era5_2010_2017.yaml")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu", "mps"), default="auto")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument("--output-dir", default="artifacts/v6_reconstruction")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config_path = resolve_path(repo_root, args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data, training = config["data"], config["training"]
    _set_seed(int(config.get("seed", 42)))
    device = select_device(args.device)
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    train_dataset = _make_reconstruction_dataset(
        repo_root, data, list(data["train_years"]), float(training.get("station_dropout", 0.0))
    )
    validation_dataset = _make_reconstruction_dataset(
        repo_root, data, list(data["validation_years"]), 0.0
    )
    batch_size = int(training["batch_size"])
    options = {
        "batch_size": batch_size,
        "num_workers": int(training.get("num_workers", 0)),
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **options)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **options)
    model = _make_model(config).to(device)
    static = StaticFields.load(resolve_path(repo_root, data["static_fields"]))
    static_fields = static.as_tensor().unsqueeze(0).to(device)
    weights = sea_weight_map(static_fields[0, 0], float(training.get("sea_weight", 2.0))).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=int(training.get("lr_scheduler_patience", 5)),
        factor=float(training.get("lr_scheduler_factor", 0.5)),
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=bool(training.get("mixed_precision", True)) and device.type == "cuda"
    )
    output_dir = resolve_path(repo_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = int(args.epochs if args.epochs is not None else training["epochs"])
    patience = int(training.get("early_stopping_patience", 12))
    progress_every = int(training.get("progress_every_batches", 100))
    best_validation = float("inf")
    no_improve = 0
    history: list[dict[str, float]] = []

    print(
        f"Reconstruction windows: train={len(train_dataset)} validation={len(validation_dataset)} "
        f"targets={TARGETS}",
        flush=True,
    )
    for epoch in range(epochs):
        started = time.time()
        train_loss = _loss_epoch(
            model, train_loader, static_fields, weights, device,
            optimizer=optimizer, scaler=scaler,
            gradient_clip=float(training.get("gradient_clip", 1.0)),
            max_batches=args.max_train_batches, label=f"epoch {epoch + 1} train",
            progress_every=progress_every,
        )
        validation_loss = _loss_epoch(
            model, validation_loader, static_fields, weights, device,
            optimizer=None, scaler=scaler, gradient_clip=0.0,
            max_batches=args.max_eval_batches, label=f"epoch {epoch + 1} validation",
            progress_every=progress_every,
        )
        scheduler.step(validation_loss)
        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "seconds": time.time() - started,
        }
        history.append(record)
        improved = validation_loss < best_validation
        if improved:
            best_validation, no_improve = validation_loss, 0
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch, "config": config},
                output_dir / "best.pt",
            )
        else:
            no_improve += 1
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(
            f"Epoch {epoch + 1:03d}/{epochs}: train={train_loss:.6f} "
            f"validation={validation_loss:.6f}{' *' if improved else ''}",
            flush=True,
        )
        if no_improve >= patience:
            print(f"Early stopping after {patience} epochs without improvement", flush=True)
            break

    saved = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(saved["model_state_dict"])
    normalization = NormalizationStats.load(resolve_path(repo_root, data["normalization_stats"]))
    result = {
        "task": "simultaneous sparse-to-grid reconstruction",
        "processor_used": False,
        "targets": TARGETS,
        "best_epoch": int(saved["epoch"]) + 1,
        "best_validation_loss": best_validation,
        "validation_metrics": _validation_metrics(
            model, validation_loader, static_fields, device, normalization, args.max_eval_batches
        ),
    }
    (output_dir / "validation_metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)
    print(f"Artifacts: {output_dir}", flush=True)
    train_dataset.close()
    validation_dataset.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
