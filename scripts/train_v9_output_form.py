#!/usr/bin/env python3
"""Train one controlled V9 output/temporal-form candidate on the development split."""

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
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from check_v7 import cache_dir, load_config, make_dataset, move, resolve  # noqa: E402
from stormengine_dl.data import NormalizationStats, StaticFields  # noqa: E402
from stormengine_dl.models.v9 import (  # noqa: E402
    StormEngineV9ForecastModel,
    load_exact_v9_checkpoint,
    warm_start_from_v7b,
    warm_start_from_v7b_expanded_encoder,
)
from stormengine_dl.training import (  # noqa: E402
    physical_six_hour_event_loss,
    sea_weight_map,
    weighted_mse,
)


def require_development_protocol(config: dict[str, Any]) -> None:
    data = config["data"]
    protocol = str(config.get("development_protocol", "v9-output-form-2020-2025"))
    protocols = {
        "v9-output-form-2020-2025": {
            "train_years": [2020, 2021, 2022],
            "validation_years": [2023],
            "confirmation_years": [2024],
            "test_years": [2025],
        },
        "v9.1-pressure-ablation-2015-2019": {
            "train_years": [2015, 2016, 2017],
            "validation_years": [2018],
            "confirmation_years": [],
            "test_years": [2019],
        },
        "v9.2-event-aware-development-2015-2018": {
            "train_years": [2015, 2016, 2017],
            "validation_years": [2018],
            "confirmation_years": [],
            "test_years": [],
        },
    }
    if protocol not in protocols:
        raise ValueError(f"Unknown frozen V9 development protocol: {protocol}")
    expected = protocols[protocol]
    mismatches = {
        key: {"expected": value, "actual": data.get(key)}
        for key, value in expected.items()
        if data.get(key) != value
    }
    if mismatches:
        raise ValueError(f"V9 development split is not frozen: {mismatches}")
    held_out = set(data["confirmation_years"] + data["test_years"])
    if set(data["train_years"] + data["validation_years"]) & held_out:
        raise ValueError("V9 development may not read confirmation or locked-test years")


def make_model(
    config: dict[str, Any], temporal_mode: str, output_mode: str
) -> StormEngineV9ForecastModel:
    data, domain, model = config["data"], config["domain"], config["model"]
    return StormEngineV9ForecastModel(
        len(data["input_variables"]),
        len(data["target_variables"]),
        int(data["forecast_hours"]),
        temporal_mode=temporal_mode,
        output_mode=output_mode,
        include_age=bool(model["include_age"]),
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


def forward(
    model: StormEngineV9ForecastModel,
    batch: dict[str, torch.Tensor],
    static: torch.Tensor,
) -> torch.Tensor:
    prediction, _ = forward_components(model, batch, static)
    return prediction


def forward_components(
    model: StormEngineV9ForecastModel,
    batch: dict[str, torch.Tensor],
    static: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    prediction, current = model.forward_with_reconstruction(
        batch["point_values"],
        batch["point_coords"],
        batch["value_mask"],
        batch["target"].shape[1],
        observation_age=batch["observation_age"],
        static_fields=static.expand(batch["target"].shape[0], -1, -1, -1),
        point_static=batch["point_static"],
    )
    if current is None:
        raise RuntimeError("V9 requires a current-field reconstruction for every candidate")
    return prediction, current


def model_contract(
    config: dict[str, Any],
    model: StormEngineV9ForecastModel,
    station_count: int,
    warm_start: dict[str, object] | None,
) -> dict[str, object]:
    return {
        "version": model.contract_version,
        "temporal_mode": model.temporal_mode,
        "output_mode": model.output_mode,
        "history_hours": int(config["data"]["history_hours"]),
        "forecast_hours": int(config["data"]["forecast_hours"]),
        "input_variables": list(config["data"]["input_variables"]),
        "target_variables": list(config["data"]["target_variables"]),
        "station_count": station_count,
        "cache_identity": Path(config["data"]["cache_identity"]).name,
        "train_years": list(config["data"]["train_years"]),
        "validation_years": list(config["data"]["validation_years"]),
        "confirmation_years": list(config["data"]["confirmation_years"]),
        "locked_test_years": list(config["data"]["test_years"]),
        "reconstruction_loss_weight": float(
            config["training"]["reconstruction_loss_weight"]
        ),
        "learning_rate": float(config["training"]["learning_rate"]),
        "weight_decay": float(config["training"]["weight_decay"]),
        "warm_start": warm_start,
        "development_protocol": str(
            config.get("development_protocol", "v9-output-form-2020-2025")
        ),
        "variable_capabilities": dict(config["data"].get("variable_capabilities", {})),
        "expected_variable_capability_counts": dict(
            config["data"].get("expected_variable_capability_counts", {})
        ),
        "event_aware": dict(config["training"].get("event_aware", {})),
    }


def event_context(
    config: dict[str, Any], static: torch.Tensor, device: torch.device
) -> dict[str, Any] | None:
    settings = config["training"].get("event_aware", {})
    if not bool(settings.get("enabled", False)):
        return None
    stats = NormalizationStats.load(resolve(config["data"]["normalization_stats"]))
    normalization = {
        name: (stats.variables[name].mean, stats.variables[name].std)
        for name in ("u10", "v10", "tp")
    }
    region = str(settings.get("region", "sea"))
    land = static[0, 0]
    masks = {
        "full": torch.ones_like(land, dtype=torch.bool),
        "land": land >= 0.5,
        "sea": land < 0.5,
    }
    if region not in masks:
        raise ValueError("event_aware.region must be full, land, or sea")
    return {
        "settings": settings,
        "normalization": normalization,
        "spatial_mask": masks[region].to(device),
    }


def event_objective(
    prediction: torch.Tensor,
    target: torch.Tensor,
    variables: list[str],
    context: dict[str, Any] | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if context is None:
        zero = prediction.sum() * 0.0
        return zero, {
            "event_loss": zero,
            "event_classification_loss": zero,
            "event_intensity_loss": zero,
            "event_positive_rain_cells": zero,
            "event_positive_wind_cells": zero,
            "event_positive_compound_cells": zero,
        }
    settings = context["settings"]
    return physical_six_hour_event_loss(
        prediction,
        target,
        variables,
        context["normalization"],
        context["spatial_mask"],
        thresholds=settings["thresholds"],
        classification_weight=float(settings["classification_weight"]),
        intensity_weight=float(settings["intensity_weight"]),
        focal_gamma=float(settings.get("focal_gamma", 2.0)),
        rain_temperature_mm=float(settings.get("rain_temperature_mm", 5.0)),
        wind_temperature_ms=float(settings.get("wind_temperature_ms", 2.0)),
    )


def event_window_sample_weights(
    dataset: Any,
    variables: list[str],
    context: dict[str, Any],
    multiplier: float,
) -> tuple[torch.Tensor, dict[str, int | float]]:
    """Find event windows from training targets only; never reads validation/test."""
    settings = context["settings"]
    thresholds = settings["thresholds"]
    normalization = context["normalization"]
    spatial = context["spatial_mask"].detach().cpu().numpy().astype(bool)
    u_index, v_index, tp_index = (variables.index(name) for name in ("u10", "v10", "tp"))
    weights = np.ones(len(dataset), dtype=np.float64)
    event_count = 0
    for item, split_start in enumerate(dataset.window_starts):
        global_start = int(dataset.global_indices[int(split_start)]) + dataset.history_hours
        values = np.asarray(
            dataset.target_grids[global_start : global_start + dataset.forecast_hours],
            dtype=np.float32,
        )
        u = values[:, u_index] * normalization["u10"][1] + normalization["u10"][0]
        v = values[:, v_index] * normalization["v10"][1] + normalization["v10"][0]
        tp = values[:, tp_index] * normalization["tp"][1] + normalization["tp"][0]
        tp_6h = np.maximum(tp, 0.0).sum(axis=0)
        wind = np.hypot(u, v).max(axis=0)
        event = (
            (tp_6h > float(thresholds["extreme_rain_6h_mm"]))
            | (wind > float(thresholds["extreme_wind_ms"]))
            | (
                (tp_6h > float(thresholds["storm_rain_6h_mm"]))
                & (wind > float(thresholds["strong_wind_ms"]))
            )
        )
        if bool(event[spatial].any()):
            weights[item] = multiplier
            event_count += 1
    return torch.from_numpy(weights), {
        "training_windows": len(dataset),
        "event_windows": event_count,
        "event_window_fraction": event_count / max(1, len(dataset)),
        "event_window_multiplier": multiplier,
    }


def evaluate_validation(
    model: StormEngineV9ForecastModel,
    loader: DataLoader,
    device: torch.device,
    static: torch.Tensor,
    weights: torch.Tensor,
    max_batches: int,
    variables: list[str],
    event: dict[str, Any] | None,
) -> dict[str, float | int]:
    model.eval()
    forecast_total = 0.0
    reconstruction_total = 0.0
    count = 0
    event_total = 0.0
    event_classification_total = 0.0
    event_intensity_total = 0.0
    with torch.no_grad():
        for raw in loader:
            if count >= max_batches:
                break
            batch = move(raw, device)
            prediction, current = forward_components(model, batch, static)
            forecast_total += float(weighted_mse(prediction, batch["target"], weights))
            event_loss, event_components = event_objective(
                prediction, batch["target"], variables, event
            )
            event_total += float(event_loss)
            event_classification_total += float(event_components["event_classification_loss"])
            event_intensity_total += float(event_components["event_intensity_loss"])
            reconstruction_total += float(
                weighted_mse(current, batch["current_target"], weights)
            )
            count += 1
    if count == 0:
        raise RuntimeError("V9 validation loader produced no batches")
    return {
        "validation_loss": forecast_total / count,
        "validation_reconstruction_loss": reconstruction_total / count,
        "validation_batches": count,
        "validation_event_loss": event_total / count,
        "validation_event_classification_loss": event_classification_total / count,
        "validation_event_intensity_loss": event_intensity_total / count,
        "validation_total_loss": (forecast_total + event_total) / count,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "baseline", "smoke", "train"))
    parser.add_argument("--config", default="configs/v9_dev_output_form.yaml")
    parser.add_argument("--temporal-mode", choices=("autoregressive", "direct"), required=True)
    parser.add_argument("--output-mode", choices=("field", "residual"), required=True)
    parser.add_argument("--variant-name", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--warm-start")
    parser.add_argument("--initial-checkpoint")
    parser.add_argument("--initial-checkpoint-sha256")
    parser.add_argument("--resume")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-validation-batches", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--reconstruction-loss-weight", type=float)
    args = parser.parse_args()

    config = load_config(resolve(args.config))
    require_development_protocol(config)
    if args.learning_rate is not None:
        config["training"]["learning_rate"] = args.learning_rate
    if args.reconstruction_loss_weight is not None:
        config["training"]["reconstruction_loss_weight"] = (
            args.reconstruction_loss_weight
        )
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device_name = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    # Deliberately instantiate only the development years here. The confirmation
    # and test datasets do not exist in this process and therefore cannot leak.
    train = make_dataset(config, config["data"]["train_years"], augment=True)
    validation = make_dataset(config, config["data"]["validation_years"], augment=False)
    if train.station_ids != validation.station_ids or len(train.station_ids) != 390:
        raise ValueError("V9 requires the fixed 390-point V7-B station contract")
    model = make_model(config, args.temporal_mode, args.output_mode)
    transfer: dict[str, object] | None = None
    if args.warm_start and args.initial_checkpoint:
        raise ValueError("use either --warm-start or --initial-checkpoint, not both")
    if args.initial_checkpoint:
        transfer = load_exact_v9_checkpoint(
            model,
            resolve(args.initial_checkpoint),
            expected_sha256=args.initial_checkpoint_sha256,
            expected_input_variables=list(config["data"]["input_variables"]),
        )
    elif args.warm_start:
        source_inputs = list(
            config["development"].get(
                "warm_start_source_input_variables", config["data"]["input_variables"]
            )
        )
        target_inputs = list(config["data"]["input_variables"])
        if source_inputs == target_inputs:
            transfer = warm_start_from_v7b(
                model,
                resolve(args.warm_start),
                expected_sha256=config["development"].get("v7b_checkpoint_sha256"),
            )
        else:
            transfer = warm_start_from_v7b_expanded_encoder(
                model,
                resolve(args.warm_start),
                source_input_variables=source_inputs,
                target_input_variables=target_inputs,
                expected_sha256=config["development"].get("v7b_checkpoint_sha256"),
            )
    model = model.to(device)
    contract = model_contract(config, model, len(train.station_ids), transfer)
    static = StaticFields.load(resolve(config["data"]["static_fields"])).as_tensor()
    static = static.unsqueeze(0).to(device)
    weights = sea_weight_map(static[0, 0], float(config["training"]["sea_weight"])).to(device)
    event = event_context(config, static, device)
    options = {
        "batch_size": int(config["training"]["batch_size"]),
        "num_workers": int(config["training"].get("num_workers", 0)),
        "pin_memory": device.type == "cuda",
    }
    sampling_summary: dict[str, int | float] | None = None
    sampler = None
    sampling = config["training"].get("event_aware", {}).get("oversampling", {})
    if event is not None and args.mode == "train" and bool(sampling.get("enabled", False)):
        sample_weights, sampling_summary = event_window_sample_weights(
            train,
            list(config["data"]["target_variables"]),
            event,
            float(sampling.get("event_window_multiplier", 8.0)),
        )
        generator = torch.Generator().manual_seed(args.seed)
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(train),
            replacement=True,
            generator=generator,
        )
        print("Event-window sampling:", json.dumps(sampling_summary), flush=True)
    train_loader = DataLoader(train, shuffle=sampler is None, sampler=sampler, **options)
    validation_loader = DataLoader(validation, shuffle=False, **options)

    if args.mode == "preflight":
        batch = move(next(iter(train_loader)), device)
        with torch.no_grad():
            prediction, current = forward_components(model, batch, static)
            reconstruction_loss = weighted_mse(
                current, batch["current_target"], weights
            )
            preflight_event_loss, preflight_event_components = event_objective(
                prediction,
                batch["target"],
                list(config["data"]["target_variables"]),
                event,
            )
        result = {
            "mode": "preflight",
            "variant": args.variant_name,
            "device": str(device),
            "cache": str(cache_dir(config["data"])),
            "train_samples": len(train),
            "validation_samples": len(validation),
            "prediction_shape": list(prediction.shape),
            "prediction_finite": bool(torch.isfinite(prediction).all()),
            "current_reconstruction_shape": list(current.shape),
            "current_reconstruction_finite": bool(torch.isfinite(current).all()),
            "current_reconstruction_loss": float(reconstruction_loss),
            "event_loss": float(preflight_event_loss),
            "event_components": {
                key: float(value) for key, value in preflight_event_components.items()
            },
            "contract": contract,
            "variable_capability_counts": train.variable_capability_counts,
        }
        print(json.dumps(result, indent=2))
        train.close(); validation.close(); return 0

    training = config["training"]
    reconstruction_weight = float(training["reconstruction_loss_weight"])
    if reconstruction_weight < 0:
        raise ValueError("reconstruction_loss_weight must be non-negative")
    output = resolve(training["output_dir"]) / args.variant_name / f"seed_{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    if args.mode == "baseline":
        started = time.perf_counter()
        metrics = evaluate_validation(
            model,
            validation_loader,
            device,
            static,
            weights,
            int(args.max_validation_batches or len(validation_loader)),
            list(config["data"]["target_variables"]),
            event,
        )
        result = {
            "mode": "baseline",
            "variant": args.variant_name,
            "seed": args.seed,
            "selection_metric": "validation_sea_weighted_mse",
            **metrics,
            "elapsed_seconds": time.perf_counter() - started,
            "contract": contract,
        }
        (output / "baseline_summary.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, indent=2))
        train.close(); validation.close(); return 0

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=int(training.get("lr_scheduler_patience", 4)),
        factor=float(training.get("lr_scheduler_factor", 0.5)),
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and bool(training.get("mixed_precision", True)),
    )
    history: list[dict[str, float | int]] = []
    start_epoch = 0
    best = math.inf
    stale = 0
    if args.resume:
        saved = torch.load(resolve(args.resume), map_location=device, weights_only=False)
        if saved.get("model_contract") != contract:
            raise ValueError("Resume checkpoint does not match the exact V9 experiment contract")
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        if "scheduler_state_dict" in saved:
            scheduler.load_state_dict(saved["scheduler_state_dict"])
        if "scaler_state_dict" in saved:
            scaler.load_state_dict(saved["scaler_state_dict"])
        history = list(saved.get("history", []))
        start_epoch = int(saved["epoch"])
        best = float(saved["best_validation_loss"])
        stale = int(saved.get("epochs_without_improvement", 0))

    smoke = args.mode == "smoke"
    epochs = int(args.epochs or (1 if smoke else training["max_epochs"]))
    max_train = int(args.max_train_batches or (2 if smoke else len(train_loader)))
    max_val = int(args.max_validation_batches or (1 if smoke else len(validation_loader)))
    progress_every = int(training.get("progress_every_batches", 100))
    started_all = time.perf_counter()
    for epoch in range(start_epoch, epochs):
        epoch_started = time.perf_counter()
        train.set_epoch(epoch)
        model.train()
        train_total = 0.0
        train_forecast_total = 0.0
        train_reconstruction_total = 0.0
        train_event_total = 0.0
        train_event_classification_total = 0.0
        train_event_intensity_total = 0.0
        train_count = 0
        for raw in train_loader:
            if train_count >= max_train:
                break
            batch = move(raw, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                prediction, current = forward_components(model, batch, static)
                forecast_loss = weighted_mse(prediction, batch["target"], weights)
                reconstruction_loss = weighted_mse(
                    current, batch["current_target"], weights
                )
                event_loss, event_components = event_objective(
                    prediction,
                    batch["target"],
                    list(config["data"]["target_variables"]),
                    event,
                )
                loss = (
                    forecast_loss
                    + reconstruction_weight * reconstruction_loss
                    + event_loss
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip"])
            )
            scaler.step(optimizer); scaler.update()
            train_total += float(loss.detach())
            train_forecast_total += float(forecast_loss.detach())
            train_reconstruction_total += float(reconstruction_loss.detach())
            train_event_total += float(event_loss.detach())
            train_event_classification_total += float(
                event_components["event_classification_loss"].detach()
            )
            train_event_intensity_total += float(
                event_components["event_intensity_loss"].detach()
            )
            train_count += 1
            if progress_every and (train_count % progress_every == 0 or train_count == max_train):
                print(
                    f"epoch {epoch + 1} train {train_count}/{min(max_train, len(train_loader))} "
                    f"loss={train_total / train_count:.6f}",
                    flush=True,
                )
        validation_metrics = evaluate_validation(
            model,
            validation_loader,
            device,
            static,
            weights,
            max_val,
            list(config["data"]["target_variables"]),
            event,
        )
        train_loss = train_total / max(1, train_count)
        train_forecast_loss = train_forecast_total / max(1, train_count)
        train_reconstruction_loss = train_reconstruction_total / max(1, train_count)
        train_event_loss = train_event_total / max(1, train_count)
        train_event_classification_loss = train_event_classification_total / max(1, train_count)
        train_event_intensity_loss = train_event_intensity_total / max(1, train_count)
        validation_loss = float(validation_metrics["validation_total_loss"])
        validation_reconstruction_loss = float(
            validation_metrics["validation_reconstruction_loss"]
        )
        scheduler.step(validation_loss)
        improved = validation_loss < best
        if improved:
            best = validation_loss; stale = 0
        else:
            stale += 1
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_forecast_loss": train_forecast_loss,
            "train_reconstruction_loss": train_reconstruction_loss,
            "train_event_loss": train_event_loss,
            "train_event_classification_loss": train_event_classification_loss,
            "train_event_intensity_loss": train_event_intensity_loss,
            **validation_metrics,
            "validation_selection_loss": validation_loss,
            "validation_reconstruction_loss": validation_reconstruction_loss,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        checkpoint = {
            "model_contract": contract,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch": epoch + 1,
            "best_validation_loss": best,
            "epochs_without_improvement": stale,
            "history": history,
            "config": config,
        }
        prefix = "smoke_" if smoke else ""
        torch.save(checkpoint, output / f"{prefix}last.pt")
        if improved:
            torch.save(checkpoint, output / f"{prefix}best.pt")
        elapsed = time.perf_counter() - epoch_started
        print(
            f"Epoch {epoch + 1}/{epochs}: train={train_loss:.6f} "
            f"forecast={train_forecast_loss:.6f} event={train_event_loss:.6f} "
            f"recon={train_reconstruction_loss:.6f} validation={validation_loss:.6f} "
            f"val_event={validation_metrics['validation_event_loss']:.6f} "
            f"val_recon={validation_reconstruction_loss:.6f} "
            f"best={best:.6f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e} time={elapsed / 60:.1f}m",
            flush=True,
        )
        if not smoke and stale >= int(training["early_stopping_patience"]):
            print(f"Early stopping after {stale} non-improving epochs.", flush=True)
            break

    summary = {
        "variant": args.variant_name,
        "temporal_mode": args.temporal_mode,
        "output_mode": args.output_mode,
        "seed": args.seed,
        "mode": args.mode,
        "selection_metric": "validation_forecast_mse_plus_event_loss",
        "reconstruction_loss_weight": reconstruction_weight,
        "best_validation_loss": best,
        "best_epoch": min(history, key=lambda row: row["validation_selection_loss"])["epoch"],
        "epochs_completed": len(history),
        "elapsed_seconds": time.perf_counter() - started_all,
        "contract": contract,
        "event_window_sampling": sampling_summary,
        "history": history,
    }
    (output / f"{args.mode}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    train.close(); validation.close(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
