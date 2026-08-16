#!/usr/bin/env python3
"""Stage-1 V8 mask-aware simultaneous spatial reconstruction training."""

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
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from check_v7 import load_config, make_dataset, resolve  # noqa: E402
from stormengine_dl import MaskAwareReconstructionModel  # noqa: E402
from stormengine_dl.data import (  # noqa: E402
    NormalizationStats,
    StaticFields,
    V7CachedReconstructionDataset,
)
from stormengine_dl.runtime import denormalize_channels  # noqa: E402
from stormengine_dl.training import RegionMetricAccumulator, sea_weight_map, weighted_mse  # noqa: E402
from stormengine_dl.models.mask_aware_reconstruction import (  # noqa: E402
    restore_spatial_training_checkpoint,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_model(config: dict[str, Any], targets: list[str]) -> MaskAwareReconstructionModel:
    data, domain, model = config["data"], config["domain"], config["model"]
    return MaskAwareReconstructionModel(
        len(data["input_variables"]),
        len(targets),
        include_age=bool(model["include_age"]),
        point_hidden=int(model["point_hidden"]),
        latent_channels=int(model["latent_channels"]),
        height=int(domain["height"]),
        width=int(domain["width"]),
        sigma=float(model["gaussian_sigma"]),
        static_channels=int(model["static_channels"]),
        point_static_channels=int(model["point_static_channels"]),
    )


def model_contract(
    config: dict[str, Any], model: MaskAwareReconstructionModel,
    targets: list[str], station_count: int,
) -> dict[str, object]:
    spatial_keys = (
        "include_age", "point_hidden", "latent_channels", "gaussian_sigma",
        "static_channels", "point_static_channels",
    )
    return {
        "version": model.contract_version,
        "task": "simultaneous_sparse_to_grid_reconstruction",
        "input_variables": list(config["data"]["input_variables"]),
        "target_variables": targets,
        "include_age": bool(config["model"]["include_age"]),
        "station_profile": str(config["data"]["station_profile"]),
        "station_count": station_count,
        "cache_identity": Path(config["data"]["cache_identity"]).name,
        "static_fields": Path(config["data"]["static_fields"]).name,
        "grid_shape": [int(config["domain"]["height"]), int(config["domain"]["width"])],
        "spatial_model": {key: config["model"][key] for key in spatial_keys},
    }


def move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=device.type == "cuda")
        for key, value in batch.items()
        if key not in {"start_index", "source_type"}
    }


def forward(
    model: MaskAwareReconstructionModel,
    batch: dict[str, torch.Tensor],
    static: torch.Tensor,
) -> torch.Tensor:
    return model(
        batch["point_values"],
        batch["point_coords"],
        batch["value_mask"],
        observation_age=batch["observation_age"],
        static_fields=static.expand(batch["target"].shape[0], -1, -1, -1),
        point_static=batch["point_static"],
    )


def run_epoch(
    model: MaskAwareReconstructionModel,
    loader: DataLoader[dict[str, torch.Tensor]],
    static: torch.Tensor,
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
        batch = move(raw, device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.autocast(
            device_type=device.type, enabled=scaler.is_enabled()
        ):
            loss = weighted_mse(forward(model, batch, static), batch["target"], weights)
        if optimizer is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        total += float(loss.detach())
        completed += 1
        if progress_every > 0 and (completed % progress_every == 0 or completed == expected):
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


def validation_metrics(
    model: MaskAwareReconstructionModel,
    loader: DataLoader[dict[str, torch.Tensor]],
    static: torch.Tensor,
    device: torch.device,
    targets: list[str],
    max_batches: int,
) -> dict[str, object]:
    model.eval()
    normalization = NormalizationStats.load(resolve("data/normalization/era5_2010_2015.json"))
    land = static[0, 0].detach().cpu()
    accumulator = RegionMetricAccumulator(tuple(targets))
    with torch.no_grad():
        for index, raw in enumerate(loader):
            if index >= max_batches:
                break
            batch = move(raw, device)
            accumulator.update(
                denormalize_channels(forward(model, batch, static), targets, normalization),
                denormalize_channels(batch["target"], targets, normalization),
                land,
            )
    return accumulator.compute()


def save_checkpoint(
    path: Path,
    model: MaskAwareReconstructionModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_validation: float,
    no_improve: int,
    history: list[dict[str, float | int]],
    contract: dict[str, object],
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
            "config": config,
        },
        path,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=("preflight", "smoke", "pilot", "screen", "develop", "train")
    )
    parser.add_argument("--config", default="configs/v8_reconstruction.yaml")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batches", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    config = load_config(resolve(args.config))
    reconstruction = config["reconstruction"]
    targets = list(reconstruction["target_variables"])
    set_seed(int(config["seed"]))
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    train_source = make_dataset(config, config["data"]["train_years"], augment=True)
    validation_source = make_dataset(config, config["data"]["validation_years"], augment=False)
    train = V7CachedReconstructionDataset(train_source, targets)
    validation = V7CachedReconstructionDataset(validation_source, targets)
    expected_stations = int(config["data"]["station_count"])
    if train_source.station_ids != validation_source.station_ids or len(train_source.station_ids) != expected_stations:
        raise ValueError("V8 reconstruction requires the fixed configured station order")

    options = {
        "batch_size": int(config["training"]["batch_size"]),
        "num_workers": int(config["training"].get("num_workers", 0)),
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train, shuffle=True, **options)
    validation_loader = DataLoader(validation, shuffle=False, **options)
    model = make_model(config, targets).to(device)
    contract = model_contract(config, model, targets, len(train_source.station_ids))
    static = StaticFields.load(resolve(config["data"]["static_fields"])).as_tensor().unsqueeze(0).to(device)
    weights = sea_weight_map(static[0, 0], float(config["training"]["sea_weight"])).to(device)

    if args.mode == "preflight":
        batch = move(next(iter(train_loader)), device)
        with torch.no_grad():
            prediction = forward(model, batch, static)
        print(json.dumps({
            "mode": "preflight",
            "device": str(device),
            "train_samples": len(train),
            "validation_samples": len(validation),
            "point_values": list(batch["point_values"].shape),
            "value_mask": list(batch["value_mask"].shape),
            "target": list(batch["target"].shape),
            "prediction": list(prediction.shape),
            "finite": bool(torch.isfinite(prediction).all()),
            "contract": contract,
        }, indent=2))
        train.close(); validation.close(); return 0

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=int(config["training"].get("lr_scheduler_patience", 5)),
        factor=float(config["training"].get("lr_scheduler_factor", 0.5)),
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and bool(config["training"].get("mixed_precision", True))
    )
    output = resolve(args.output_dir or reconstruction["output_dir"])
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
                "Refusing to mix a new run with existing artifacts: "
                + ", ".join(str(path) for path in existing)
            )
    start_epoch, best_validation, no_improve = 0, math.inf, 0
    history: list[dict[str, float | int]] = []
    resume_path: Path | None = None
    if args.resume:
        resume_path = resolve(args.resume)
        start_epoch, best_validation, no_improve, history = restore_spatial_training_checkpoint(
            resume_path, output, model, optimizer, scheduler, scaler, contract, device
        )

    if args.mode == "smoke":
        epochs = int(args.epochs or 1)
        train_batches = int(args.batches or reconstruction["smoke_train_batches"])
        validation_batches = int(reconstruction["smoke_validation_batches"])
    elif args.mode == "pilot":
        epochs = int(args.epochs or reconstruction["pilot_epochs"])
        train_batches = int(args.batches or reconstruction["pilot_train_batches"])
        validation_batches = int(reconstruction["pilot_validation_batches"])
    elif args.mode == "screen":
        if args.resume:
            raise ValueError("Screening candidates must start from the same random initialization")
        epochs = int(args.epochs or reconstruction["screen_epochs"])
        train_batches = int(args.batches or reconstruction["screen_train_batches"])
        validation_batches = int(reconstruction["screen_validation_batches"])
    elif args.mode == "develop":
        if args.resume:
            raise ValueError("Development candidates must start from the same random initialization")
        epochs = int(args.epochs or reconstruction["development_epochs"])
        train_batches = len(train_loader)
        validation_batches = len(validation_loader)
    else:
        epochs = int(args.epochs or reconstruction["max_epochs"])
        train_batches = len(train_loader)
        validation_batches = len(validation_loader)

    if epochs <= start_epoch:
        raise ValueError(
            f"Target epoch count {epochs} must be greater than completed epoch {start_epoch}; "
            "--epochs is the total target, not the number of additional epochs"
        )

    progress_every = int(config["training"].get("progress_every_batches", 100))
    patience = int(reconstruction["early_stopping_patience"])
    for epoch in range(start_epoch, epochs):
        started = time.perf_counter()
        train.set_epoch(epoch)
        train_loss = run_epoch(
            model, train_loader, static, weights, device,
            optimizer=optimizer, scaler=scaler, max_batches=train_batches,
            gradient_clip=float(config["training"]["gradient_clip"]),
            progress_every=progress_every, label=f"epoch {epoch + 1} train",
        )
        validation_loss = run_epoch(
            model, validation_loader, static, weights, device,
            optimizer=None, scaler=scaler, max_batches=validation_batches,
            gradient_clip=0.0, progress_every=progress_every,
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
        save_checkpoint(
            output / "last.pt", model, optimizer, scheduler, scaler, epoch + 1,
            best_validation, no_improve, history, contract, config,
        )
        if improved:
            save_checkpoint(
                output / "best.pt", model, optimizer, scheduler, scaler, epoch + 1,
                best_validation, no_improve, history, contract, config,
            )
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        print(
            f"Epoch {epoch + 1:03d}/{epochs}: train={train_loss:.6f} "
            f"validation={validation_loss:.6f}{' *' if improved else ''}", flush=True,
        )
        if no_improve >= patience:
            print(f"Early stopping after {patience} epochs without improvement", flush=True)
            break

    best = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    result = {
        "schema_version": 1,
        "mode": args.mode,
        "scientific_status": (
            "validation_result" if args.mode == "train" else
            "development_candidate_only" if args.mode == "develop" else
            "candidate_screening_only" if args.mode == "screen" else
            "pilot_only"
        ),
        "processor_used": False,
        "seed": int(config["seed"]),
        "initial_epoch": start_epoch,
        "target_max_epoch": epochs,
        "train_batches_per_epoch": train_batches,
        "validation_batches_per_epoch": validation_batches,
        "resumed_from": str(resume_path) if resume_path is not None else None,
        "train_years": list(config["data"]["train_years"]),
        "validation_years": list(config["data"]["validation_years"]),
        "test_years_read": [],
        "best_epoch": int(best["epoch"]),
        "best_validation_loss": float(best["best_validation_loss"]),
        "validation_metrics": validation_metrics(
            model, validation_loader, static, device, targets, validation_batches
        ),
        "contract": contract,
    }
    (output / f"{args.mode}_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)
    print(f"Artifacts: {output}", flush=True)
    train.close(); validation.close(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
