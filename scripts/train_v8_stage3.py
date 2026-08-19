#!/usr/bin/env python3
"""V8 Stage 3 gradual unfreezing on the six-year development split."""

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
from stormengine_dl.data import (  # noqa: E402
    NormalizationStats,
    StaticFields,
    V7CachedReconstructionDataset,
)
from stormengine_dl.models.mask_aware_reconstruction import (  # noqa: E402
    configure_gradual_unfreezing,
    set_gradual_unfreezing_mode,
)
from stormengine_dl.runtime import denormalize_channels  # noqa: E402
from stormengine_dl.training import (  # noqa: E402
    ForecastMetricAccumulator,
    RegionMetricAccumulator,
    sea_weight_map,
    weighted_mse,
)


STAGE2_CONTRACT_VERSION = "stormengine-v8-stage2-processor-only-v1"
STAGE3_CONTRACT_VERSION = "stormengine-v8-stage3-gradual-unfreeze-v1"


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
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def expected_module_names(phase: str) -> tuple[str, ...]:
    return ("processor", "decoder") if phase == "stage3a" else (
        "encoder", "processor", "decoder"
    )


def validate_source_checkpoint(
    saved: dict[str, Any], phase: str, config: dict[str, Any]
) -> dict[str, Any]:
    contract = saved.get("model_contract")
    if not isinstance(contract, dict):
        raise ValueError("Stage 3 source checkpoint has no model contract")
    seed = int(config["seed"])
    if phase == "stage3a":
        if contract.get("version") != STAGE2_CONTRACT_VERSION:
            raise ValueError("Stage 3A must start from a V8 Stage-2 checkpoint")
        spatial = (
            contract.get("spatial_pretraining", {})
            .get("contract", {})
            .get("spatial_model", {})
        )
        processor = contract.get("processor", {})
        checks = {
            "checkpoint seed": int(saved.get("config", {}).get("seed", -1)) == seed,
            "point_hidden": int(spatial.get("point_hidden", -1)) == 64,
            "latent_channels": int(spatial.get("latent_channels", -1)) == 96,
            "processor layers": int(processor.get("layers", -1)) == 3,
            "processor kernel": int(processor.get("kernel_size", -1)) == 3,
            "station count": int(contract.get("station_count", -1)) == 390,
        }
    else:
        checks = {
            "contract version": contract.get("version") == STAGE3_CONTRACT_VERSION,
            "source phase": contract.get("phase") == "stage3a",
            "checkpoint seed": int(contract.get("seed", -1)) == seed,
            "train years": contract.get("train_years") == list(range(2010, 2016)),
            "validation years": contract.get("validation_years") == [2016],
            "test locked": contract.get("test_years_read") == [],
        }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Incompatible {phase} source checkpoint: {failed}")
    state = saved.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("Stage 3 source checkpoint has no model_state_dict")
    return contract


def stage3_contract(
    config: dict[str, Any], phase: str, source_sha256: str,
    source_contract: dict[str, Any], station_count: int,
) -> dict[str, Any]:
    data, model, stage3 = config["data"], config["model"], config["stage3"]
    phase_config = stage3[phase]
    return {
        "version": STAGE3_CONTRACT_VERSION,
        "phase": phase,
        "task": "six_year_sparse_forecast_gradual_unfreezing",
        "seed": int(config["seed"]),
        "history_hours": int(data["history_hours"]),
        "forecast_hours": int(data["forecast_hours"]),
        "input_variables": list(data["input_variables"]),
        "target_variables": list(data["target_variables"]),
        "station_profile": str(data["station_profile"]),
        "station_count": station_count,
        "cache_identity": Path(data["cache_identity"]).name,
        "window_stride_hours": int(data.get("window_stride_hours", 1)),
        "train_years": list(data["train_years"]),
        "validation_years": list(data["validation_years"]),
        "test_years_read": [],
        "model": {
            "point_hidden": int(model["point_hidden"]),
            "latent_channels": int(model["latent_channels"]),
            "gaussian_sigma": float(model["gaussian_sigma"]),
            "processor_layers": int(model["processor_layers"]),
            "kernel_size": int(model["kernel_size"]),
        },
        "source": {
            "checkpoint_sha256": source_sha256,
            "contract": source_contract,
        },
        "optimization": {
            "trainable_modules": list(expected_module_names(phase)),
            "learning_rates": {
                name: float(phase_config[f"{name}_learning_rate"])
                for name in expected_module_names(phase)
            },
            "weight_decay": float(config["training"]["weight_decay"]),
            "sea_weight": float(config["training"]["sea_weight"]),
            "batch_size": int(stage3["batch_size"]),
        },
    }


def optimizer_for(
    model: nn.Module, phase: str, config: dict[str, Any]
) -> torch.optim.AdamW:
    phase_config = config["stage3"][phase]
    groups = []
    for name in expected_module_names(phase):
        module = getattr(model, name)
        parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
        if not parameters:
            raise RuntimeError(f"{phase} has no trainable {name} parameters")
        groups.append({
            "params": parameters,
            "lr": float(phase_config[f"{name}_learning_rate"]),
            "name": name,
        })
    return torch.optim.AdamW(
        groups, weight_decay=float(config["training"]["weight_decay"])
    )


def learning_rates(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    return {
        str(group.get("name", index)): float(group["lr"])
        for index, group in enumerate(optimizer.param_groups)
    }


def run_forecast_epoch(
    model: nn.Module,
    loader: DataLoader[dict[str, torch.Tensor]],
    static: torch.Tensor,
    weights: torch.Tensor,
    device: torch.device,
    phase: str,
    *,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    gradient_clip: float,
    max_batches: int,
    progress_every: int,
    label: str,
) -> float:
    training = optimizer is not None
    set_gradual_unfreezing_mode(model, phase, training)  # type: ignore[arg-type]
    expected = min(len(loader), max_batches)
    total = 0.0
    completed = 0
    started = time.perf_counter()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
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
            torch.nn.utils.clip_grad_norm_(trainable, gradient_clip)
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
                f"  {label}: {completed:,}/{expected:,} loss={total / completed:.6f} "
                f"elapsed={elapsed / 60:.1f}m ETA={eta / 60:.1f}m",
                flush=True,
            )
    if completed == 0:
        raise RuntimeError("data loader produced no batches")
    return total / completed


def reconstruction_forward(
    model: nn.Module, batch: dict[str, torch.Tensor], static: torch.Tensor
) -> torch.Tensor:
    encoded = model.encoder(  # type: ignore[attr-defined]
        batch["point_values"], batch["point_coords"], batch["value_mask"],
        batch["observation_age"], batch["point_static"],
    )
    return model.decoder(  # type: ignore[attr-defined]
        encoded, static.expand(batch["target"].shape[0], -1, -1, -1)
    )


def reconstruction_diagnostic(
    model: nn.Module,
    loader: DataLoader[dict[str, torch.Tensor]],
    static: torch.Tensor,
    weights: torch.Tensor,
    device: torch.device,
    variables: list[str],
    max_batches: int,
) -> dict[str, Any]:
    model.eval()
    normalization = NormalizationStats.load(
        resolve("data/normalization/era5_2010_2015.json")
    )
    accumulator = RegionMetricAccumulator(tuple(variables))
    land = static[0, 0].detach().cpu()
    total = 0.0
    completed = 0
    with torch.no_grad():
        for raw in loader:
            if completed >= max_batches:
                break
            batch = move(raw, device)
            prediction = reconstruction_forward(model, batch, static)
            total += float(weighted_mse(prediction, batch["target"], weights))
            accumulator.update(
                denormalize_channels(prediction, variables, normalization),
                denormalize_channels(batch["target"], variables, normalization),
                land,
            )
            completed += 1
    if completed == 0:
        raise RuntimeError("reconstruction loader produced no batches")
    return {
        "normalized_sea_weighted_loss": total / completed,
        "batches": completed,
        "metrics": accumulator.compute(),
    }


def forecast_metrics(
    model: nn.Module,
    loader: DataLoader[dict[str, torch.Tensor]],
    static: torch.Tensor,
    device: torch.device,
    variables: list[str],
    forecast_hours: int,
    max_batches: int,
) -> dict[str, Any]:
    model.eval()
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
                denormalize_channels(forward(model, batch, static), variables, normalization),  # type: ignore[arg-type]
                denormalize_channels(batch["target"], variables, normalization),
                land,
            )
    return accumulator.compute()


def degradation_percent(current: float, reference: float) -> float:
    if reference <= 0:
        raise ValueError("reconstruction reference loss must be positive")
    return 100.0 * (current / reference - 1.0)


def write_json(path: Path, value: object) -> None:
    path.write_bytes((json.dumps(value, indent=2) + "\n").encode("utf-8"))


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
    history: list[dict[str, Any]],
    contract: dict[str, Any],
    original_reconstruction_reference: dict[str, Any],
    source_reconstruction: dict[str, Any],
    data_loader_generator_state: torch.Tensor,
    config: dict[str, Any],
) -> None:
    torch.save({
        "model_contract": contract,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": epoch,
        "best_validation_loss": best_validation,
        "epochs_without_improvement": no_improve,
        "history": history,
        "original_reconstruction_reference": original_reconstruction_reference,
        "source_reconstruction": source_reconstruction,
        "data_loader_generator_state": data_loader_generator_state,
        "config": config,
    }, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("stage3a", "stage3b"))
    parser.add_argument("mode", choices=("preflight", "smoke", "pilot", "train"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--source-checkpoint")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batches", type=int)
    args = parser.parse_args()

    config = load_config(resolve(args.config))
    data, model_config, training, stage3 = (
        config["data"], config["model"], config["training"], config["stage3"]
    )
    if list(data["train_years"]) != list(range(2010, 2016)):
        raise ValueError("V8 Stage 3 requires 2010--2015 training")
    if list(data["validation_years"]) != [2016] or list(data["test_years"]) != [2017]:
        raise ValueError("V8 Stage 3 requires 2016 validation and locked 2017 test")
    required_model = {
        "point_hidden": 64, "latent_channels": 96,
        "processor_layers": 3, "kernel_size": 3,
    }
    failed_model = [
        key for key, value in required_model.items()
        if int(model_config[key]) != value
    ]
    if failed_model:
        raise ValueError(f"Stage-3 selected-model contract failed: {failed_model}")

    seed = int(config["seed"])
    set_seed(seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    # The test split is deliberately never instantiated in this script.
    train = make_dataset(config, data["train_years"], augment=True)
    validation = make_dataset(config, data["validation_years"], augment=False)
    reconstruction = V7CachedReconstructionDataset(
        validation, list(data["target_variables"])
    )
    station_count = int(data["station_count"])
    if train.station_ids != validation.station_ids or len(train.station_ids) != station_count:
        raise ValueError("Stage 3 requires one fixed 390-station order")

    phase_config = stage3[args.phase]
    source_path = resolve(
        args.source_checkpoint or phase_config["source_checkpoint"]
    )
    if not source_path.is_file():
        raise FileNotFoundError(f"Stage-3 source checkpoint is missing: {source_path}")
    source_saved = torch.load(source_path, map_location="cpu", weights_only=False)
    source_contract = validate_source_checkpoint(source_saved, args.phase, config)
    model = make_model(config)
    model.load_state_dict(source_saved["model_state_dict"], strict=True)
    source_module_hashes = {
        "encoder": module_sha256(model.encoder),
        "processor": module_sha256(model.processor),
        "decoder": module_sha256(model.decoder),
    }
    trainable_names = configure_gradual_unfreezing(model, args.phase)
    allowed = expected_module_names(args.phase)
    if not trainable_names or any(
        not name.startswith(tuple(f"{module}." for module in allowed))
        for name in trainable_names
    ):
        raise RuntimeError(f"Stage-3 trainable parameter contract failed: {trainable_names}")
    source_hash = sha256_file(source_path)
    contract = stage3_contract(
        config, args.phase, source_hash, source_contract, station_count
    )
    model = model.to(device)

    loader_options = {
        "batch_size": int(stage3["batch_size"]),
        "num_workers": int(training.get("num_workers", 0)),
        "pin_memory": device.type == "cuda",
    }
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train, shuffle=True, generator=generator, **loader_options)
    validation_loader = DataLoader(validation, shuffle=False, **loader_options)
    reconstruction_loader = DataLoader(reconstruction, shuffle=False, **loader_options)
    static = StaticFields.load(resolve(data["static_fields"])).as_tensor().unsqueeze(0).to(device)
    weights = sea_weight_map(static[0, 0], float(training["sea_weight"])).to(device)

    if args.mode == "preflight":
        batch = move(next(iter(train_loader)), device)
        reconstruction_batch = move(next(iter(reconstruction_loader)), device)
        with torch.no_grad():
            prediction = forward(model, batch, static)
            reconstructed = reconstruction_forward(model, reconstruction_batch, static)
        result = {
            "mode": "preflight", "phase": args.phase, "device": str(device),
            "seed": seed, "train_years": list(data["train_years"]),
            "validation_years": list(data["validation_years"]), "test_years_read": [],
            "train_samples": len(train), "validation_samples": len(validation),
            "point_values": list(batch["point_values"].shape),
            "target": list(batch["target"].shape),
            "prediction": list(prediction.shape),
            "reconstruction": list(reconstructed.shape),
            "finite": bool(torch.isfinite(prediction).all() and torch.isfinite(reconstructed).all()),
            "trainable_parameter_names": list(trainable_names),
            "trainable_parameters": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
            "optimizer_learning_rates": {
                name: float(phase_config[f"{name}_learning_rate"]) for name in allowed
            },
            "source_checkpoint": str(source_path),
            "source_checkpoint_sha256": source_hash,
            "contract": contract,
        }
        print(json.dumps(result, indent=2), flush=True)
        train.close(); validation.close(); return 0

    optimizer = optimizer_for(model, args.phase, config)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=int(phase_config["lr_scheduler_patience"]),
        factor=float(phase_config["lr_scheduler_factor"]),
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and bool(training.get("mixed_precision", True))
    )
    output = resolve(args.output_dir or phase_config["output_dir"])
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
                "Refusing to mix a new Stage-3 run with existing artifacts: "
                + ", ".join(map(str, existing))
            )

    if args.mode == "smoke":
        epochs = int(args.epochs or 1)
        train_batches = int(args.batches or stage3["smoke_train_batches"])
        validation_batches = int(stage3["smoke_validation_batches"])
        reconstruction_batches = int(stage3["smoke_reconstruction_batches"])
        patience = epochs + 1
    elif args.mode == "pilot":
        epochs = int(args.epochs or stage3["pilot_epochs"])
        train_batches = int(args.batches or stage3["pilot_train_batches"])
        validation_batches = int(stage3["pilot_validation_batches"])
        reconstruction_batches = int(stage3["pilot_reconstruction_batches"])
        patience = epochs + 1
    else:
        epochs = int(args.epochs or phase_config["max_epochs"])
        train_batches = len(train_loader)
        validation_batches = len(validation_loader)
        reconstruction_batches = len(reconstruction_loader)
        patience = int(phase_config["early_stopping_patience"])

    source_reconstruction = reconstruction_diagnostic(
        model, reconstruction_loader, static, weights, device,
        list(data["target_variables"]), reconstruction_batches,
    )
    original_reference = source_saved.get(
        "original_reconstruction_reference", source_reconstruction
    )
    if not isinstance(original_reference, dict):
        raise ValueError("Invalid original reconstruction reference in source checkpoint")

    start_epoch, best_validation, no_improve = 0, math.inf, 0
    history: list[dict[str, Any]] = []
    if args.resume:
        saved = torch.load(resolve(args.resume), map_location=device, weights_only=False)
        if saved.get("model_contract") != contract:
            raise ValueError("Stage-3 resume checkpoint contract is incompatible")
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        scheduler.load_state_dict(saved["scheduler_state_dict"])
        scaler.load_state_dict(saved["scaler_state_dict"])
        generator.set_state(saved["data_loader_generator_state"])
        resume_reference = saved.get("original_reconstruction_reference")
        if not isinstance(resume_reference, dict):
            raise ValueError("Resume checkpoint lacks the reconstruction reference")
        # The resume contract already pins the original source checkpoint hash.
        # Retain the stored diagnostic rather than requiring bitwise-identical
        # floating-point recomputation on a potentially different CUDA session.
        original_reference = resume_reference
        start_epoch = int(saved["epoch"])
        best_validation = float(saved["best_validation_loss"])
        no_improve = int(saved["epochs_without_improvement"])
        history = list(saved["history"])
        if len(history) != start_epoch:
            raise ValueError("Resume history length does not match completed epoch")
    if epochs <= start_epoch:
        raise ValueError("--epochs is the total target and must exceed completed epochs")

    progress_every = int(training.get("progress_every_batches", 100))
    for epoch in range(start_epoch, epochs):
        started = time.perf_counter()
        train.set_epoch(epoch)
        train_loss = run_forecast_epoch(
            model, train_loader, static, weights, device, args.phase,
            optimizer=optimizer, scaler=scaler,
            gradient_clip=float(training["gradient_clip"]),
            max_batches=train_batches, progress_every=progress_every,
            label=f"{args.phase} epoch {epoch + 1} train",
        )
        validation_loss = run_forecast_epoch(
            model, validation_loader, static, weights, device, args.phase,
            optimizer=None, scaler=scaler, gradient_clip=0.0,
            max_batches=validation_batches, progress_every=progress_every,
            label=f"{args.phase} epoch {epoch + 1} validation",
        )
        scheduler.step(validation_loss)
        improved = validation_loss < best_validation
        if improved:
            best_validation, no_improve = validation_loss, 0
        else:
            no_improve += 1
        history.append({
            "epoch": epoch + 1, "train_loss": train_loss,
            "validation_loss": validation_loss,
            "learning_rates": learning_rates(optimizer),
            "seconds": time.perf_counter() - started,
        })
        checkpoint_arguments = {
            "model": model, "optimizer": optimizer, "scheduler": scheduler,
            "scaler": scaler, "epoch": epoch + 1,
            "best_validation": best_validation, "no_improve": no_improve,
            "history": history, "contract": contract,
            "original_reconstruction_reference": original_reference,
            "source_reconstruction": source_reconstruction,
            "data_loader_generator_state": generator.get_state(),
            "config": config,
        }
        save_checkpoint(output / "last.pt", **checkpoint_arguments)
        if improved:
            save_checkpoint(output / "best.pt", **checkpoint_arguments)
        write_json(output / "history.json", history)
        print(
            f"Epoch {epoch + 1:03d}/{epochs}: train={train_loss:.6f} "
            f"validation={validation_loss:.6f} lr={learning_rates(optimizer)}"
            f"{' *' if improved else ''}", flush=True,
        )
        if no_improve >= patience:
            print(f"Early stopping after {patience} epochs without improvement", flush=True)
            break

    best = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    final_module_hashes = {
        "encoder": module_sha256(model.encoder),
        "processor": module_sha256(model.processor),
        "decoder": module_sha256(model.decoder),
    }
    if args.phase == "stage3a" and (
        final_module_hashes["encoder"] != source_module_hashes["encoder"]
    ):
        raise RuntimeError("Frozen Encoder changed during Stage 3A")
    final_reconstruction = reconstruction_diagnostic(
        model, reconstruction_loader, static, weights, device,
        list(data["target_variables"]), reconstruction_batches,
    )
    reference_loss = float(original_reference["normalized_sea_weighted_loss"])
    current_loss = float(final_reconstruction["normalized_sea_weighted_loss"])
    degradation = degradation_percent(current_loss, reference_loss)
    threshold = float(stage3["max_reconstruction_degradation_percent"])
    summary = {
        "schema_version": 1, "phase": args.phase, "mode": args.mode,
        "scientific_status": (
            f"{args.phase}_six_year_validation_candidate" if args.mode == "train"
            else f"{args.phase}_{args.mode}_only"
        ),
        "seed": seed, "train_years": list(data["train_years"]),
        "validation_years": list(data["validation_years"]), "test_years_read": [],
        "best_epoch": int(best["epoch"]), "completed_epochs": len(history),
        "best_validation_loss": float(best["best_validation_loss"]),
        "stopped_early": no_improve >= patience,
        "epochs_without_improvement": no_improve,
        "train_batches_per_epoch": train_batches,
        "validation_batches_per_epoch": validation_batches,
        "source_checkpoint_sha256": source_hash,
        "source_best_validation_loss": float(source_saved["best_validation_loss"]),
        "trainable_parameter_names": list(trainable_names),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "source_module_sha256": source_module_hashes,
        "selected_module_sha256": final_module_hashes,
        "frozen_encoder_verified": (
            final_module_hashes["encoder"] == source_module_hashes["encoder"]
            if args.phase == "stage3a" else None
        ),
        "optimizer_learning_rates_initial": {
            name: float(phase_config[f"{name}_learning_rate"]) for name in allowed
        },
        "forecast_validation_metrics": forecast_metrics(
            model, validation_loader, static, device, list(data["target_variables"]),
            int(data["forecast_hours"]), validation_batches,
        ),
        "reconstruction": {
            "original_stage2_reference": original_reference,
            "source_checkpoint": source_reconstruction,
            "selected_checkpoint": final_reconstruction,
            "degradation_percent_vs_stage2": degradation,
            "allowed_degradation_percent": threshold,
            "preservation_gate_passed": degradation <= threshold,
        },
        "contract": contract,
    }
    write_json(output / f"{args.mode}_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Artifacts: {output}", flush=True)
    train.close(); validation.close(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
