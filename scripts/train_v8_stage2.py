#!/usr/bin/env python3
"""V8 Stage 2: train only an L3K3 Processor between frozen spatial modules."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from check_v7 import forward, load_config, make_dataset, make_model, move, resolve  # noqa: E402
from stormengine_dl.data import NormalizationStats, StaticFields  # noqa: E402
from stormengine_dl.models.mask_aware_reconstruction import (  # noqa: E402
    V8_RECONSTRUCTION_CONTRACT,
    freeze_spatial_modules,
    load_spatial_pretraining,
    set_processor_only_training_mode,
)
from stormengine_dl.runtime import denormalize_channels  # noqa: E402
from stormengine_dl.training import (  # noqa: E402
    ForecastMetricAccumulator,
    sea_weight_map,
    weighted_mse,
)


STAGE2_CONTRACT_VERSION = "stormengine-v8-stage2-processor-only-v1"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def module_sha256(module: nn.Module) -> str:
    """Stable tensor hash used to prove Encoder/Decoder remained frozen."""

    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def reset_processor(processor: nn.Module, seed: int) -> None:
    """Give equal-sized candidates identical Processor initialization by seed.

    Encoder construction consumes a different number of random values when
    point_hidden changes.  A dedicated RNG scope prevents that irrelevant fact
    from changing the Processor initialization across spatial candidates.
    """

    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        for child in processor.modules():
            reset = getattr(child, "reset_parameters", None)
            if callable(reset):
                reset()


def expected_spatial_contract(
    config: dict[str, Any], station_count: int
) -> dict[str, object]:
    data, domain, model = config["data"], config["domain"], config["model"]
    targets = list(config["reconstruction"]["target_variables"])
    spatial_keys = (
        "include_age", "point_hidden", "latent_channels", "gaussian_sigma",
        "static_channels", "point_static_channels",
    )
    return {
        "version": V8_RECONSTRUCTION_CONTRACT,
        "task": "simultaneous_sparse_to_grid_reconstruction",
        "input_variables": list(data["input_variables"]),
        "target_variables": targets,
        "include_age": bool(model["include_age"]),
        "station_profile": str(data["station_profile"]),
        "station_count": station_count,
        "cache_identity": Path(data["cache_identity"]).name,
        "static_fields": Path(data["static_fields"]).name,
        "grid_shape": [int(domain["height"]), int(domain["width"])],
        "spatial_model": {key: model[key] for key in spatial_keys},
    }


def stage2_contract(
    config: dict[str, Any],
    *,
    spatial_contract: dict[str, object],
    spatial_checkpoint_sha256: str,
    station_count: int,
) -> dict[str, object]:
    data, model, training, stage2 = (
        config["data"], config["model"], config["training"], config["stage2"]
    )
    return {
        "version": STAGE2_CONTRACT_VERSION,
        "task": "sparse_history_to_future_grid_processor_only",
        "history_hours": int(data["history_hours"]),
        "forecast_hours": int(data["forecast_hours"]),
        "input_variables": list(data["input_variables"]),
        "target_variables": list(data["target_variables"]),
        "station_profile": str(data["station_profile"]),
        "station_count": station_count,
        "cache_identity": Path(data["cache_identity"]).name,
        "window_stride_hours": int(data.get("window_stride_hours", 1)),
        "spatial_pretraining": {
            "checkpoint_sha256": spatial_checkpoint_sha256,
            "contract": spatial_contract,
        },
        "processor": {
            "family": "convgru",
            "layers": int(model["processor_layers"]),
            "kernel_size": int(model["kernel_size"]),
            "latent_channels": int(model["latent_channels"]),
            "initialization": "random_from_stage2_seed",
        },
        "optimization": {
            "trainable_modules": ["processor"],
            "frozen_modules": ["encoder", "decoder"],
            "learning_rate": float(training["learning_rate"]),
            "weight_decay": float(training["weight_decay"]),
            "sea_weight": float(training["sea_weight"]),
            "batch_size": int(stage2["batch_size"]),
        },
    }


def load_pretrained_spatial(
    model: nn.Module,
    checkpoint_path: Path,
    expected: dict[str, object],
) -> dict[str, object]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Stage-1 best checkpoint is missing: {checkpoint_path}. "
            "Copy/retain the local Windows checkpoint before Stage 2."
        )
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    load_spatial_pretraining(model, saved, expected_contract=expected)  # type: ignore[arg-type]
    return saved


def run_epoch(
    model: nn.Module,
    loader: DataLoader[dict[str, torch.Tensor]],
    static: torch.Tensor,
    weights: torch.Tensor,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    gradient_clip: float,
    max_batches: int,
    progress_every: int,
    label: str,
) -> float:
    training = optimizer is not None
    set_processor_only_training_mode(model, training)  # type: ignore[arg-type]
    expected = min(len(loader), max_batches)
    total = 0.0
    completed = 0
    started = time.perf_counter()
    for raw in loader:
        if completed >= max_batches:
            break
        batch = move(raw, device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.autocast(
            device_type=device.type, enabled=scaler.is_enabled()
        ):
            prediction = forward(model, batch, static)  # type: ignore[arg-type]
            loss = weighted_mse(prediction, batch["target"], weights)
        if optimizer is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.processor.parameters(), gradient_clip  # type: ignore[attr-defined]
            )
            scaler.step(optimizer)
            scaler.update()
        total += float(loss.detach())
        completed += 1
        if progress_every > 0 and (
            completed % progress_every == 0 or completed == expected
        ):
            elapsed = time.perf_counter() - started
            eta = elapsed / completed * max(0, expected - completed)
            print(
                f"  {label}: {completed:,}/{expected:,} "
                f"loss={total / completed:.6f} elapsed={elapsed / 60:.1f}m "
                f"ETA={eta / 60:.1f}m",
                flush=True,
            )
    if completed == 0:
        raise RuntimeError("data loader produced no batches")
    return total / completed


def validation_metrics(
    model: nn.Module,
    loader: DataLoader[dict[str, torch.Tensor]],
    static: torch.Tensor,
    device: torch.device,
    variables: list[str],
    forecast_hours: int,
    max_batches: int,
) -> dict[str, object]:
    set_processor_only_training_mode(model, False)  # type: ignore[arg-type]
    normalization = NormalizationStats.load(
        resolve("data/normalization/era5_2010_2015.json")
    )
    accumulator = ForecastMetricAccumulator(tuple(variables), forecast_hours)
    land = static[0, 0].detach().cpu()
    with torch.no_grad():
        for index, raw in enumerate(loader):
            if index >= max_batches:
                break
            batch = move(raw, device)
            accumulator.update(
                denormalize_channels(
                    forward(model, batch, static), variables, normalization  # type: ignore[arg-type]
                ),
                denormalize_channels(batch["target"], variables, normalization),
                land,
            )
    return accumulator.compute()


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_validation: float,
    no_improve: int,
    history: list[dict[str, float | int]],
    contract: dict[str, object],
    frozen_hashes: dict[str, str],
    config: dict[str, Any],
) -> None:
    torch.save(
        {
            "model_contract": contract,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch": epoch,
            "best_validation_loss": best_validation,
            "epochs_without_improvement": no_improve,
            "history": history,
            "frozen_module_sha256": frozen_hashes,
            "config": config,
        },
        path,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "smoke", "pilot", "train"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--spatial-checkpoint")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batches", type=int)
    args = parser.parse_args()

    config = load_config(resolve(args.config))
    data, model_config, training, stage2 = (
        config["data"], config["model"], config["training"], config["stage2"]
    )
    if list(data["test_years"]) != [2017]:
        raise ValueError("V8 Stage 2 requires 2017 to remain the declared locked test year")
    if int(model_config["processor_layers"]) != 3 or int(model_config["kernel_size"]) != 3:
        raise ValueError("The frozen Stage-2 Processor choice must be ConvGRU L3K3")

    seed = int(config["seed"])
    set_seed(seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    # Only train and validation datasets are instantiated.  2017 cannot be
    # accidentally read by this script.
    train = make_dataset(config, data["train_years"], augment=True)
    validation = make_dataset(config, data["validation_years"], augment=False)
    station_count = int(data["station_count"])
    if train.station_ids != validation.station_ids or len(train.station_ids) != station_count:
        raise ValueError("Stage 2 requires one fixed station order in both splits")

    expected_spatial = expected_spatial_contract(config, station_count)
    spatial_checkpoint = resolve(
        args.spatial_checkpoint or stage2["spatial_checkpoint"]
    )
    model = make_model(config)
    spatial_saved = load_pretrained_spatial(model, spatial_checkpoint, expected_spatial)
    reset_processor(model.processor, seed)
    trainable_names = freeze_spatial_modules(model)
    if not trainable_names or any(
        not name.startswith("processor.") for name in trainable_names
    ):
        raise RuntimeError(f"Stage-2 trainable parameter contract failed: {trainable_names}")
    frozen_hashes = {
        "encoder": module_sha256(model.encoder),
        "decoder": module_sha256(model.decoder),
    }
    checkpoint_sha = sha256_file(spatial_checkpoint)
    contract = stage2_contract(
        config,
        spatial_contract=expected_spatial,
        spatial_checkpoint_sha256=checkpoint_sha,
        station_count=station_count,
    )
    model = model.to(device)

    generator = torch.Generator().manual_seed(seed)
    loader_options = {
        "batch_size": int(stage2["batch_size"]),
        "num_workers": int(training.get("num_workers", 0)),
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train, shuffle=True, generator=generator, **loader_options)
    validation_loader = DataLoader(validation, shuffle=False, **loader_options)
    static = StaticFields.load(resolve(data["static_fields"])).as_tensor().unsqueeze(0).to(device)
    weights = sea_weight_map(static[0, 0], float(training["sea_weight"])).to(device)

    if args.mode == "preflight":
        batch = move(next(iter(train_loader)), device)
        with torch.no_grad():
            prediction = forward(model, batch, static)
        result = {
            "mode": "preflight",
            "device": str(device),
            "seed": seed,
            "train_years": list(data["train_years"]),
            "validation_years": list(data["validation_years"]),
            "test_years_read": [],
            "train_samples": len(train),
            "validation_samples": len(validation),
            "point_values": list(batch["point_values"].shape),
            "target": list(batch["target"].shape),
            "prediction": list(prediction.shape),
            "finite": bool(torch.isfinite(prediction).all()),
            "trainable_parameter_names": list(trainable_names),
            "trainable_parameters": sum(
                parameter.numel() for parameter in model.parameters()
                if parameter.requires_grad
            ),
            "spatial_checkpoint": str(spatial_checkpoint),
            "spatial_checkpoint_sha256": checkpoint_sha,
            "stage1_best_epoch": int(spatial_saved["epoch"]),
            "stage1_best_validation_loss": float(spatial_saved["best_validation_loss"]),
            "contract": contract,
        }
        print(json.dumps(result, indent=2), flush=True)
        train.close(); validation.close(); return 0

    optimizer = torch.optim.AdamW(
        model.processor.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=int(stage2.get("lr_scheduler_patience", 5)),
        factor=float(stage2.get("lr_scheduler_factor", 0.5)),
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and bool(training.get("mixed_precision", True)),
    )
    output = resolve(args.output_dir or stage2["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        existing = [
            path for path in (
                output / "best.pt", output / "last.pt", output / "history.json",
                output / f"{args.mode}_summary.json",
            ) if path.exists()
        ]
        if existing:
            raise FileExistsError(
                "Refusing to mix a new Stage-2 run with existing artifacts: "
                + ", ".join(map(str, existing))
            )

    start_epoch, best_validation, no_improve = 0, math.inf, 0
    history: list[dict[str, float | int]] = []
    if args.resume:
        saved = torch.load(resolve(args.resume), map_location=device, weights_only=False)
        if saved.get("model_contract") != contract:
            raise ValueError("Stage-2 resume checkpoint contract is incompatible")
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        scheduler.load_state_dict(saved["scheduler_state_dict"])
        scaler.load_state_dict(saved["scaler_state_dict"])
        if saved.get("frozen_module_sha256") != frozen_hashes:
            raise ValueError("Resume checkpoint does not preserve Stage-1 spatial weights")
        start_epoch = int(saved["epoch"])
        best_validation = float(saved["best_validation_loss"])
        no_improve = int(saved["epochs_without_improvement"])
        history = list(saved["history"])
        if len(history) != start_epoch:
            raise ValueError("Resume history length does not match completed epoch")

    if args.mode == "smoke":
        epochs = int(args.epochs or 1)
        train_batches = int(args.batches or stage2["smoke_train_batches"])
        validation_batches = int(stage2["smoke_validation_batches"])
        patience = epochs + 1
    elif args.mode == "pilot":
        epochs = int(args.epochs or stage2["pilot_epochs"])
        train_batches = int(args.batches or stage2["pilot_train_batches"])
        validation_batches = int(stage2["pilot_validation_batches"])
        patience = epochs + 1
    else:
        epochs = int(args.epochs or stage2["max_epochs"])
        train_batches = len(train_loader)
        validation_batches = len(validation_loader)
        patience = int(stage2["early_stopping_patience"])
    if epochs <= start_epoch:
        raise ValueError("--epochs is the total target and must exceed completed epochs")

    progress_every = int(training.get("progress_every_batches", 100))
    for epoch in range(start_epoch, epochs):
        started = time.perf_counter()
        train.set_epoch(epoch)
        train_loss = run_epoch(
            model, train_loader, static, weights, device,
            optimizer=optimizer, scaler=scaler,
            gradient_clip=float(training["gradient_clip"]),
            max_batches=train_batches, progress_every=progress_every,
            label=f"epoch {epoch + 1} train",
        )
        validation_loss = run_epoch(
            model, validation_loader, static, weights, device,
            optimizer=None, scaler=scaler, gradient_clip=0.0,
            max_batches=validation_batches, progress_every=progress_every,
            label=f"epoch {epoch + 1} validation",
        )
        scheduler.step(validation_loss)
        improved = validation_loss < best_validation
        if improved:
            best_validation, no_improve = validation_loss, 0
        else:
            no_improve += 1
        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "seconds": time.perf_counter() - started,
        }
        history.append(record)
        checkpoint_arguments = {
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "scaler": scaler,
            "epoch": epoch + 1,
            "best_validation": best_validation,
            "no_improve": no_improve,
            "history": history,
            "contract": contract,
            "frozen_hashes": frozen_hashes,
            "config": config,
        }
        save_checkpoint(output / "last.pt", **checkpoint_arguments)
        if improved:
            save_checkpoint(output / "best.pt", **checkpoint_arguments)
        (output / "history.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"Epoch {epoch + 1:03d}/{epochs}: train={train_loss:.6f} "
            f"validation={validation_loss:.6f}{' *' if improved else ''}",
            flush=True,
        )
        if no_improve >= patience:
            print(f"Early stopping after {patience} epochs without improvement", flush=True)
            break

    best = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    final_hashes = {
        "encoder": module_sha256(model.encoder),
        "decoder": module_sha256(model.decoder),
    }
    if final_hashes != frozen_hashes:
        raise RuntimeError("Frozen Encoder/Decoder changed during Stage 2")
    variables = list(data["target_variables"])
    summary = {
        "schema_version": 1,
        "mode": args.mode,
        "scientific_status": (
            "stage2_validation_candidate" if args.mode == "train" else
            "stage2_pilot_only" if args.mode == "pilot" else "stage2_smoke_only"
        ),
        "seed": seed,
        "train_years": list(data["train_years"]),
        "validation_years": list(data["validation_years"]),
        "test_years_read": [],
        "best_epoch": int(best["epoch"]),
        "completed_epochs": len(history),
        "best_validation_loss": float(best["best_validation_loss"]),
        "stopped_early": no_improve >= patience,
        "epochs_without_improvement": no_improve,
        "train_batches_per_epoch": train_batches,
        "validation_batches_per_epoch": validation_batches,
        "spatial_checkpoint_sha256": checkpoint_sha,
        "frozen_module_sha256": final_hashes,
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "validation_metrics": validation_metrics(
            model, validation_loader, static, device, variables,
            int(data["forecast_hours"]), validation_batches,
        ),
        "contract": contract,
    }
    (output / f"{args.mode}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Artifacts: {output}", flush=True)
    train.close(); validation.close(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
