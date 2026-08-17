#!/usr/bin/env python3
"""Fair Processor-only development on dense normalized ERA5 grids."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from stormengine_dl.data import (  # noqa: E402
    CachedEra5SequenceDataset,
    DenseGridForecastDataset,
    StaticFields,
)
from stormengine_dl.models import make_dense_processor_model  # noqa: E402
from stormengine_dl.training import (  # noqa: E402
    ForecastMetricAccumulator,
    sea_weight_map,
    weighted_mse,
)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if "extends" not in config:
        return config
    return _deep_merge(load_config(path.parent / config["extends"]), config)


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def cache_dir(data: dict[str, Any]) -> Path:
    value = Path(data["training_cache"]).expanduser()
    return value.resolve() if value.is_absolute() else resolve(data["era5_root"]) / value


def load_normalization() -> dict[str, dict[str, float]]:
    raw = json.loads(resolve("data/normalization/era5_2010_2015.json").read_text(encoding="utf-8"))
    return raw["variables"]


def denormalize_channels(
    values: torch.Tensor,
    variables: list[str],
    normalization: dict[str, dict[str, float]],
) -> torch.Tensor:
    result = values.detach().float().cpu().clone()
    for channel, variable in enumerate(variables):
        stats = normalization[variable]
        result[:, :, channel] = result[:, :, channel] * float(stats["std"]) + float(stats["mean"])
    return result


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dense_dataset(config: dict[str, Any], years: list[int]) -> DenseGridForecastDataset:
    data = config["data"]
    development = config["processor_development"]
    source = CachedEra5SequenceDataset(
        cache_dir(data),
        years=years,
        history_hours=int(data["history_hours"]),
        forecast_hours=int(data["forecast_hours"]),
        window_stride_hours=int(development["window_stride_hours"]),
        # Processor-only training never reads the legacy sparse point array;
        # only validate the dense target-grid channel contract.
        input_variables=None,
        target_variables=data["target_variables"],
    )
    return DenseGridForecastDataset(source)


def make_model(config: dict[str, Any]) -> torch.nn.Module:
    data, domain = config["data"], config["domain"]
    development = config["processor_development"]
    return make_dense_processor_model(
        str(development["family"]),
        input_channels=len(data["target_variables"]),
        output_channels=len(data["target_variables"]),
        latent_channels=int(development["latent_channels"]),
        height=int(domain["height"]),
        width=int(domain["width"]),
        history_steps=int(data["history_hours"]),
        forecast_steps=int(data["forecast_hours"]),
        processor_layers=int(development["layers"]),
        kernel_size=int(development.get("kernel_size", 3)),
        patch_size=int(development.get("patch_size", 4)),
        transformer_dimension=int(development.get("transformer_dimension", 128)),
        transformer_heads=int(development.get("transformer_heads", 4)),
        transformer_mlp_ratio=float(development.get("transformer_mlp_ratio", 4.0)),
        dropout=float(development.get("dropout", 0.0)),
    )


def contract(config: dict[str, Any], model: torch.nn.Module) -> dict[str, object]:
    data, domain = config["data"], config["domain"]
    development = config["processor_development"]
    keys = (
        "family", "latent_channels", "layers", "kernel_size", "patch_size",
        "transformer_dimension", "transformer_heads", "transformer_mlp_ratio", "dropout",
    )
    return {
        "version": "stormengine-dense-processor-v1",
        "task": "dense_era5_history_to_future",
        "history_hours": int(data["history_hours"]),
        "forecast_hours": int(data["forecast_hours"]),
        "variables": list(data["target_variables"]),
        "grid_shape": [int(domain["height"]), int(domain["width"])],
        "window_stride_hours": int(development["window_stride_hours"]),
        "processor": {key: development[key] for key in keys if key in development},
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


def move(raw: dict[str, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        raw["history"].to(device, non_blocking=device.type == "cuda"),
        raw["target"].to(device, non_blocking=device.type == "cuda"),
    )


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader[dict[str, torch.Tensor]],
    weights: torch.Tensor,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    max_batches: int,
    gradient_clip: float,
    progress_every: int,
    label: str,
) -> float:
    training = optimizer is not None
    model.train(training)
    expected = min(len(loader), max_batches)
    total = 0.0
    completed = 0
    started = time.perf_counter()
    for raw in loader:
        if completed >= max_batches:
            break
        history, target = move(raw, device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.autocast(
            device_type=device.type, enabled=scaler.is_enabled()
        ):
            prediction = model(history, target.shape[1])
            loss = weighted_mse(prediction, target, weights)
        if optimizer is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        total += float(loss.detach())
        completed += 1
        if progress_every and (completed % progress_every == 0 or completed == expected):
            elapsed = time.perf_counter() - started
            eta = elapsed / completed * max(0, expected - completed)
            print(
                f"  {label}: {completed:,}/{expected:,} loss={total / completed:.6f} "
                f"elapsed={elapsed / 60:.1f}m ETA={eta / 60:.1f}m",
                flush=True,
            )
    if not completed:
        raise RuntimeError("data loader produced no batches")
    return total / completed


def validation_metrics(
    model: torch.nn.Module,
    loader: DataLoader[dict[str, torch.Tensor]],
    static: torch.Tensor,
    device: torch.device,
    variables: list[str],
    max_batches: int,
) -> dict[str, object]:
    model.eval()
    normalization = load_normalization()
    accumulator = ForecastMetricAccumulator(tuple(variables), int(loader.dataset.source.forecast_hours))
    with torch.no_grad():
        for index, raw in enumerate(loader):
            if index >= max_batches:
                break
            history, target = move(raw, device)
            prediction = model(history, target.shape[1])
            accumulator.update(
                denormalize_channels(prediction, variables, normalization),
                denormalize_channels(target, variables, normalization),
                static[0, 0].detach().cpu(),
            )
    return accumulator.compute()


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    scaler: torch.amp.GradScaler,
    *,
    epoch: int,
    best_validation: float,
    no_improve: int,
    history: list[dict[str, float | int]],
    model_contract: dict[str, object],
    config: dict[str, Any],
) -> None:
    torch.save({
        "model_contract": model_contract,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": epoch,
        "best_validation_loss": best_validation,
        "epochs_without_improvement": no_improve,
        "history": history,
        "config": config,
    }, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "smoke", "develop"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batches", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    config = load_config(resolve(args.config))
    data, training = config["data"], config["training"]
    development = config["processor_development"]
    seed = int(config["seed"])
    set_seed(seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    train = make_dense_dataset(config, data["train_years"])
    validation = make_dense_dataset(config, data["validation_years"])
    if train.variables != validation.variables:
        raise ValueError("training and validation variable contracts differ")
    options = {
        "batch_size": int(development["batch_size"]),
        "num_workers": int(training.get("num_workers", 0)),
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train, shuffle=True, **options)
    validation_loader = DataLoader(validation, shuffle=False, **options)
    model = make_model(config).to(device)
    model_contract = contract(config, model)
    static = StaticFields.load(resolve(data["static_fields"])).as_tensor().unsqueeze(0).to(device)
    weights = sea_weight_map(static[0, 0], float(training["sea_weight"])).to(device)

    if args.mode == "preflight":
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        history, target = move(next(iter(train_loader)), device)
        prediction = model(history, target.shape[1])
        loss = weighted_mse(prediction, target, weights)
        loss.backward()
        peak = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
        result = {
            "mode": "preflight",
            "device": str(device),
            "family": development["family"],
            "train_samples": len(train),
            "validation_samples": len(validation),
            "batch_size": options["batch_size"],
            "history": list(history.shape),
            "target": list(target.shape),
            "prediction": list(prediction.shape),
            "finite": bool(torch.isfinite(prediction).all()),
            "loss": float(loss.detach()),
            "peak_cuda_allocated_bytes": int(peak),
            "peak_cuda_allocated_gib": peak / 1024**3,
            "contract": model_contract,
        }
        print(json.dumps(result, indent=2), flush=True)
        train.close(); validation.close(); return 0

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=int(training.get("lr_scheduler_patience", 5)),
        factor=float(training.get("lr_scheduler_factor", 0.5)),
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and bool(training.get("mixed_precision", True))
    )
    output = resolve(args.output_dir or development["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        existing = [path for path in (output / "best.pt", output / "last.pt", output / "history.json") if path.exists()]
        if existing:
            raise FileExistsError("Refusing to mix a new run with existing artifacts: " + ", ".join(map(str, existing)))

    start_epoch, best_validation, no_improve = 0, math.inf, 0
    records: list[dict[str, float | int]] = []
    resume_path: Path | None = None
    if args.resume:
        resume_path = resolve(args.resume)
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        if checkpoint.get("model_contract") != model_contract:
            raise ValueError("resume checkpoint Processor contract does not match this config")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint["epoch"])
        best_validation = float(checkpoint["best_validation_loss"])
        no_improve = int(checkpoint["epochs_without_improvement"])
        records = list(checkpoint["history"])

    if args.mode == "smoke":
        epochs = int(args.epochs or 1)
        train_batches = int(args.batches or development["smoke_train_batches"])
        validation_batches = int(development["smoke_validation_batches"])
    else:
        epochs = int(args.epochs or development["max_epochs"])
        train_batches = len(train_loader)
        validation_batches = len(validation_loader)
    if epochs <= start_epoch:
        raise ValueError("target epoch must exceed the checkpoint epoch")

    patience = int(development["early_stopping_patience"])
    for epoch in range(start_epoch, epochs):
        started = time.perf_counter()
        train_loss = run_epoch(
            model, train_loader, weights, device,
            optimizer=optimizer, scaler=scaler, max_batches=train_batches,
            gradient_clip=float(training["gradient_clip"]),
            progress_every=int(training.get("progress_every_batches", 100)),
            label=f"epoch {epoch + 1} train",
        )
        validation_loss = run_epoch(
            model, validation_loader, weights, device,
            optimizer=None, scaler=scaler, max_batches=validation_batches,
            gradient_clip=0.0,
            progress_every=int(training.get("progress_every_batches", 100)),
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
        records.append(record)
        save_checkpoint(
            output / "last.pt", model, optimizer, scheduler, scaler,
            epoch=epoch + 1, best_validation=best_validation, no_improve=no_improve,
            history=records, model_contract=model_contract, config=config,
        )
        if improved:
            save_checkpoint(
                output / "best.pt", model, optimizer, scheduler, scaler,
                epoch=epoch + 1, best_validation=best_validation, no_improve=no_improve,
                history=records, model_contract=model_contract, config=config,
            )
        (output / "history.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
        print(
            f"Epoch {epoch + 1:03d}/{epochs}: train={train_loss:.6f} "
            f"validation={validation_loss:.6f}{' *' if improved else ''}", flush=True,
        )
        if no_improve >= patience:
            print(f"Early stopping after {patience} epochs without improvement", flush=True)
            break

    best = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    summary = {
        "schema_version": 1,
        "mode": args.mode,
        "scientific_status": "processor_family_development_only" if args.mode == "develop" else "smoke_only",
        "seed": seed,
        "train_years": list(data["train_years"]),
        "validation_years": list(data["validation_years"]),
        "test_years_read": [],
        "best_epoch": int(best["epoch"]),
        "completed_epochs": len(records),
        "best_validation_loss": float(best["best_validation_loss"]),
        "stopped_early": no_improve >= patience,
        "early_stopping_patience": patience,
        "resumed_from": str(resume_path) if resume_path else None,
        "train_batches_per_epoch": train_batches,
        "validation_batches_per_epoch": validation_batches,
        "validation_metrics": validation_metrics(
            model, validation_loader, static, device, list(data["target_variables"]), validation_batches
        ),
        "contract": model_contract,
    }
    (output / f"{args.mode}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Artifacts: {output}", flush=True)
    train.close(); validation.close(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
