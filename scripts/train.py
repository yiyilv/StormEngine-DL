#!/usr/bin/env python3
"""Train, resume, and evaluate the end-to-end StormEngine V6 model."""

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

from stormengine_dl import StormEngineForecastModel
from stormengine_dl.data import Era5SequenceDataset, NormalizationStats, StaticFields
from stormengine_dl.training import RegionMetricAccumulator, sea_weight_map, weighted_mse


def _resolve(repo_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_dataset(repo_root: Path, data: dict[str, Any], years: list[int], dropout: float) -> Era5SequenceDataset:
    return Era5SequenceDataset.from_station_registry(
        manifest_path=_resolve(repo_root, data["era5_manifest"]),
        data_root=_resolve(repo_root, data["era5_root"]),
        station_registry_path=_resolve(repo_root, data["station_registry"]),
        station_profile=data["station_profile"],
        input_variables=data["input_variables"],
        target_variables=data["target_variables"],
        history_hours=int(data["history_hours"]),
        forecast_hours=int(data["forecast_hours"]),
        window_stride_hours=int(data.get("window_stride_hours", 1)),
        cache_months=int(data.get("cache_months", 2)),
        years=years,
        station_dropout=dropout,
        normalization_path=_resolve(repo_root, data["normalization_stats"]),
    )


def _make_model(config: dict[str, Any]) -> StormEngineForecastModel:
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


def _move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=device.type == "cuda")
        for key, value in batch.items()
        if key != "start_index"
    }


def _forward(
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


def _run_loss_epoch(
    model: StormEngineForecastModel,
    loader: DataLoader[dict[str, torch.Tensor]],
    static_fields: torch.Tensor,
    weights: torch.Tensor,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    gradient_clip: float = 1.0,
    max_batches: int | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total, batches = 0.0, 0
    for index, raw_batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        batch = _move(raw_batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        amp = device.type == "cuda" and scaler is not None and scaler.is_enabled()
        with torch.set_grad_enabled(training), torch.autocast(device_type=device.type, enabled=amp):
            prediction = _forward(model, batch, static_fields)
            loss = weighted_mse(prediction, batch["target"], weights)
        if training:
            assert scaler is not None
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        total += float(loss.detach())
        batches += 1
    if batches == 0:
        raise RuntimeError("data loader produced no batches")
    return total / batches


def _denormalize_channels(values: torch.Tensor, variables: list[str], stats: NormalizationStats) -> torch.Tensor:
    result = values.detach().float().cpu().clone()
    for channel, variable in enumerate(variables):
        stat = stats.variables[variable]
        result[:, :, channel] = result[:, :, channel] * stat.std + stat.mean
    return result


def _evaluate(
    model: StormEngineForecastModel,
    loader: DataLoader[dict[str, torch.Tensor]],
    static_fields: torch.Tensor,
    device: torch.device,
    variables: list[str],
    normalization: NormalizationStats,
    max_batches: int | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    model.eval()
    accumulator = RegionMetricAccumulator(tuple(variables))
    land_mask = static_fields[0, 0].detach().cpu()
    with torch.no_grad():
        for index, raw_batch in enumerate(loader):
            if max_batches is not None and index >= max_batches:
                break
            batch = _move(raw_batch, device)
            prediction = _forward(model, batch, static_fields)
            accumulator.update(
                _denormalize_channels(prediction, variables, normalization),
                _denormalize_channels(batch["target"], variables, normalization),
                land_mask,
            )
    return accumulator.compute()


def _checkpoint(
    path: Path,
    model: StormEngineForecastModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_val: float,
    no_improve: int,
    history: list[dict[str, float]],
    config: dict[str, Any],
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val,
            "no_improve": no_improve,
            "history": history,
            "config": config,
        },
        path,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu", "mps"), default="auto")
    parser.add_argument("--resume", help="Resume from a last.pt checkpoint")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument("--epochs", type=int, help="Override configured epoch count")
    parser.add_argument("--output-dir", help="Override the configured artifact directory")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config_path = _resolve(repo_root, args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data, training = config["data"], config["training"]
    _set_seed(int(config.get("seed", 42)))

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this PyTorch installation")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_dataset = _make_dataset(
        repo_root, data, data["train_years"], float(training.get("station_dropout", 0.0))
    )
    validation_dataset = _make_dataset(repo_root, data, data["validation_years"], 0.0)
    test_dataset = _make_dataset(repo_root, data, data["test_years"], 0.0)
    batch_size = int(training["batch_size"])
    num_workers = int(training.get("num_workers", 0))
    loader_options = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)

    model = _make_model(config).to(device)
    static = StaticFields.load(_resolve(repo_root, data["static_fields"]))
    static_fields = static.as_tensor().unsqueeze(0).to(device)
    weights = sea_weight_map(static_fields[0, 0], float(training.get("sea_weight", 2.0))).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=int(training.get("lr_scheduler_patience", 5)),
        factor=float(training.get("lr_scheduler_factor", 0.5)),
    )
    amp_enabled = bool(training.get("mixed_precision", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    output_dir = _resolve(
        repo_root,
        args.output_dir or training.get("output_dir", "artifacts/v6_pilot"),
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    start_epoch, best_val, no_improve, history = 0, float("inf"), 0, []
    if args.resume:
        resume_path = _resolve(repo_root, args.resume)
        saved = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        scheduler.load_state_dict(saved["scheduler_state_dict"])
        scaler.load_state_dict(saved["scaler_state_dict"])
        start_epoch = int(saved["epoch"]) + 1
        best_val = float(saved["best_val_loss"])
        no_improve = int(saved["no_improve"])
        history = list(saved["history"])
        print(f"Resumed {resume_path} at epoch {start_epoch + 1}")

    epochs = int(args.epochs if args.epochs is not None else training["epochs"])
    patience = int(training.get("early_stopping_patience", 12))
    print(
        f"Windows: train={len(train_dataset)} validation={len(validation_dataset)} "
        f"test={len(test_dataset)} | batch={batch_size} | AMP={amp_enabled}"
    )
    print(
        f"V6 loss: sea_weight={training.get('sea_weight', 2.0)} "
        f"land={float(weights[static_fields[0, 0] >= 0.5].mean()):.3f} "
        f"sea={float(weights[static_fields[0, 0] < 0.5].mean()):.3f}"
    )

    for epoch in range(start_epoch, epochs):
        started = time.time()
        train_loss = _run_loss_epoch(
            model,
            train_loader,
            static_fields,
            weights,
            device,
            optimizer=optimizer,
            scaler=scaler,
            gradient_clip=float(training.get("gradient_clip", 1.0)),
            max_batches=args.max_train_batches,
        )
        validation_loss = _run_loss_epoch(
            model,
            validation_loader,
            static_fields,
            weights,
            device,
            scaler=scaler,
            max_batches=args.max_eval_batches,
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
        improved = validation_loss < best_val
        if improved:
            best_val, no_improve = validation_loss, 0
        else:
            no_improve += 1
        _checkpoint(
            output_dir / "last.pt", model, optimizer, scheduler, scaler,
            epoch, best_val, no_improve, history, config,
        )
        if improved:
            _checkpoint(
                output_dir / "best.pt", model, optimizer, scheduler, scaler,
                epoch, best_val, no_improve, history, config,
            )
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(
            f"Epoch {epoch + 1:03d}/{epochs}: train={train_loss:.6f} "
            f"val={validation_loss:.6f} lr={record['learning_rate']:.2e} "
            f"time={record['seconds']:.1f}s{' *' if improved else ''}"
        )
        if no_improve >= patience:
            print(f"Early stopping after {patience} epochs without improvement")
            break

    best = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    normalization = NormalizationStats.load(_resolve(repo_root, data["normalization_stats"]))
    metrics = _evaluate(
        model,
        test_loader,
        static_fields,
        device,
        list(data["target_variables"]),
        normalization,
        max_batches=args.max_eval_batches,
    )
    result = {
        "best_epoch": int(best["epoch"]) + 1,
        "best_validation_loss": float(best["best_val_loss"]),
        "test_metrics": metrics,
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Artifacts: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
