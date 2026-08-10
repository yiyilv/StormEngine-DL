#!/usr/bin/env python3
"""Run V7 forward, smoke, benchmark, or capped pilot checks."""

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

from stormengine_dl import StormEngineV7ForecastModel  # noqa: E402
from stormengine_dl.data import MissingnessStrategy, StaticFields, V7CachedSequenceDataset  # noqa: E402
from stormengine_dl.models.mask_aware import require_v7_checkpoint_contract  # noqa: E402
from stormengine_dl.training import sea_weight_map, weighted_mse  # noqa: E402


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
    value = Path(data["training_cache"])
    return value.resolve() if value.is_absolute() else resolve(data["era5_root"]) / value


def strategy(config: dict[str, Any]) -> MissingnessStrategy:
    value = config["missingness"]
    variable_range = value.get("variable_dropout_range")
    station_range = value.get("station_dropout_range")
    outage_range = value.get("outage_duration_hours")
    return MissingnessStrategy(
        variable_dropout=value.get("variable_dropout", {}),
        station_dropout=float(value.get("station_dropout", 0)),
        network_dropout=float(value.get("network_outage_probability", value.get("network_dropout", 0))),
        time_block_probability=float(value.get("time_block_probability", 0)),
        time_block_hours=int(value.get("time_block_hours", 3)),
        age_60_probability=float(value.get("age_60_probability", 0)),
        variable_dropout_range=tuple(map(float, variable_range)) if variable_range else None,
        station_dropout_range=tuple(map(float, station_range)) if station_range else None,
        outage_duration_hours=tuple(map(int, outage_range)) if outage_range else None,
    )


def make_dataset(
    config: dict[str, Any], years: list[int], *, augment: bool
) -> V7CachedSequenceDataset:
    data = config["data"]
    missingness = config["missingness"]
    use_empirical = bool(
        missingness.get(
            "use_real_templates_for_training" if augment else "use_real_templates_for_validation",
            False,
        )
    )
    empirical_path = (
        resolve(missingness["empirical_tensor"])
        if use_empirical
        else None
    )
    empirical_manifest = (
        resolve(missingness["empirical_manifest"])
        if empirical_path is not None
        else None
    )
    return V7CachedSequenceDataset(
        cache_dir(data), resolve(data["station_registry"]), resolve(data["cache_identity"]),
        years=years, input_variables=data["input_variables"],
        target_variables=data["target_variables"],
        strategy=strategy(config) if augment and not use_empirical else MissingnessStrategy({}),
        history_hours=int(data["history_hours"]), forecast_hours=int(data["forecast_hours"]),
        window_stride_hours=int(data.get("window_stride_hours", 1)), seed=int(config["seed"]),
        empirical_mask_path=empirical_path,
        empirical_mask_manifest_path=empirical_manifest,
    )


def make_model(config: dict[str, Any]) -> StormEngineV7ForecastModel:
    data, domain, model = config["data"], config["domain"], config["model"]
    return StormEngineV7ForecastModel(
        len(data["input_variables"]), len(data["target_variables"]),
        include_age=bool(model["include_age"]), point_hidden=int(model["point_hidden"]),
        latent_channels=int(model["latent_channels"]), height=int(domain["height"]),
        width=int(domain["width"]), sigma=float(model["gaussian_sigma"]),
        processor_layers=int(model["processor_layers"]), kernel_size=int(model["kernel_size"]),
        static_channels=int(model["static_channels"]),
        point_static_channels=int(model["point_static_channels"]),
    )


def contract(
    config: dict[str, Any], model: StormEngineV7ForecastModel, station_count: int
) -> dict[str, object]:
    return {
        "version": model.contract_version,
        "input_variables": list(config["data"]["input_variables"]),
        "include_age": bool(config["model"]["include_age"]),
        "observation_age_units": (
            "hours_capped_at_1" if bool(config["model"]["include_age"]) else "not_used"
        ),
        "station_count": station_count,
        "cache_identity": Path(config["data"]["cache_identity"]).name,
        "static_fields": Path(config["data"]["static_fields"]).name,
        "model": dict(config["model"]),
    }


def move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=device.type == "cuda") for key, value in batch.items() if key != "start_index"}


def forward(model: StormEngineV7ForecastModel, batch: dict[str, torch.Tensor], static: torch.Tensor) -> torch.Tensor:
    return model(
        batch["point_values"], batch["point_coords"], batch["value_mask"],
        batch["target"].shape[1], observation_age=batch["observation_age"],
        static_fields=static.expand(batch["target"].shape[0], -1, -1, -1),
        point_static=batch["point_static"],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "forward", "smoke", "benchmark", "pilot", "train"))
    parser.add_argument("--config", default="configs/v7_a.yaml")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batches", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--resume")
    args = parser.parse_args()
    config = load_config(resolve(args.config))
    seed = int(config["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    train = make_dataset(config, config["data"]["train_years"], augment=True)
    validation = make_dataset(config, config["data"]["validation_years"], augment=False)
    if train.station_ids != validation.station_ids or len(train.station_ids) != 239:
        raise ValueError("V7-A requires the same 239 physical stations in every split")
    options = dict(batch_size=int(config["training"]["batch_size"]), num_workers=int(config["training"].get("num_workers", 0)), pin_memory=device.type == "cuda")
    train_loader = DataLoader(train, shuffle=True, **options)
    validation_loader = DataLoader(validation, shuffle=False, **options)
    model = make_model(config).to(device)
    model_contract = contract(config, model, len(train.station_ids))
    static = StaticFields.load(resolve(config["data"]["static_fields"])).as_tensor().unsqueeze(0).to(device)
    weights = sea_weight_map(static[0, 0], float(config["training"]["sea_weight"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and bool(config["training"].get("mixed_precision", True)))
    start_epoch = 0
    best_validation = math.inf
    epochs_without_improvement = 0
    history: list[dict[str, float | int | None]] = []
    if args.resume:
        saved = torch.load(resolve(args.resume), map_location=device, weights_only=False)
        require_v7_checkpoint_contract(saved, model_contract)
        model.load_state_dict(saved["model_state_dict"])
        if "optimizer_state_dict" in saved:
            optimizer.load_state_dict(saved["optimizer_state_dict"])
        if "scaler_state_dict" in saved:
            scaler.load_state_dict(saved["scaler_state_dict"])
        start_epoch = int(saved.get("epoch", 0))
        best_validation = float(saved.get("best_validation_loss", math.inf))
        epochs_without_improvement = int(saved.get("epochs_without_improvement", 0))
        history = list(saved.get("history", []))

    if args.mode in {"preflight", "forward"}:
        batch = move(next(iter(train_loader)), device)
        with torch.no_grad():
            prediction = forward(model, batch, static)
        present = batch["value_mask"].any(dim=-1).sum(dim=-1).float()
        valid_age = batch["observation_age"][batch["value_mask"]]
        result = {"mode": "preflight", "device": str(device), "splits": {"train_samples": len(train), "validation_samples": len(validation), "test_years": list(config["data"]["test_years"])}, "point_values": list(batch["point_values"].shape), "value_mask": list(batch["value_mask"].shape), "observation_age": list(batch["observation_age"].shape), "prediction": list(prediction.shape), "finite": bool(torch.isfinite(prediction).all()), "valid_fraction": float(batch["value_mask"].float().mean()), "stations_present_per_hour": {"min": int(present.min()), "median": float(present.median()), "max": int(present.max())}, "nonzero_age_fraction_of_valid": float((valid_age > 0).float().mean()), "contract": model_contract}
        print(json.dumps(result, indent=2)); train.close(); validation.close(); return 0

    training = config["training"]
    validation_missingness = (
        "empirical_dpc"
        if config["missingness"].get("use_real_templates_for_validation", False)
        else "clean"
    )
    default_batches = training["benchmark_batches"] if args.mode == "benchmark" else training["smoke_train_batches"] if args.mode == "smoke" else training["pilot_train_batches"] if args.mode == "pilot" else len(train_loader)
    max_train = int(args.batches or default_batches)
    max_validation = int(training["smoke_validation_batches"] if args.mode == "smoke" else training["pilot_validation_batches"] if args.mode == "pilot" else len(validation_loader))
    max_epochs = int(args.epochs or (training["pilot_epochs"] if args.mode == "pilot" else training.get("max_epochs", 1) if args.mode == "train" else 1))
    warmup = int(training["benchmark_warmup_batches"] if args.mode == "benchmark" else 0)
    progress_every = int(training.get("progress_every_batches", 100))
    timings: list[float] = []
    output = resolve(training["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    for epoch in range(start_epoch, max_epochs):
        epoch_started = time.perf_counter()
        expected_train_batches = min(len(train_loader), max_train + warmup)
        print(
            f"Epoch {epoch + 1}/{max_epochs} | train: {expected_train_batches} batches",
            flush=True,
        )
        train.set_epoch(epoch)
        model.train(); total = 0.0; count = 0
        for raw in train_loader:
            if count >= max_train + warmup: break
            started = time.perf_counter(); batch = move(raw, device); optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                prediction = forward(model, batch, static); loss = weighted_mse(prediction, batch["target"], weights)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip"]))
            scaler.step(optimizer); scaler.update()
            if device.type == "cuda": torch.cuda.synchronize()
            if count >= warmup: timings.append(time.perf_counter() - started); total += float(loss.detach())
            count += 1
            if progress_every > 0 and (
                count % progress_every == 0 or count == expected_train_batches
            ):
                elapsed = time.perf_counter() - epoch_started
                eta = elapsed / count * max(0, expected_train_batches - count)
                print(
                    f"  train {count}/{expected_train_batches} "
                    f"({100 * count / expected_train_batches:5.1f}%) "
                    f"loss={float(loss.detach()):.5f} elapsed={elapsed / 60:.1f}m "
                    f"ETA={eta / 60:.1f}m",
                    flush=True,
                )
        train_loss = total / max(1, count - warmup)
        model.eval(); val_total = 0.0; val_count = 0
        if args.mode != "benchmark":
            expected_validation_batches = min(len(validation_loader), max_validation)
            validation_started = time.perf_counter()
            print(f"Epoch {epoch + 1}/{max_epochs} | validation: {expected_validation_batches} batches", flush=True)
            with torch.no_grad():
                for raw in validation_loader:
                    if val_count >= max_validation: break
                    batch = move(raw, device); val_total += float(weighted_mse(forward(model, batch, static), batch["target"], weights)); val_count += 1
                    if progress_every > 0 and (
                        val_count % progress_every == 0
                        or val_count == expected_validation_batches
                    ):
                        elapsed = time.perf_counter() - validation_started
                        eta = elapsed / val_count * max(0, expected_validation_batches - val_count)
                        print(
                            f"  validation {val_count}/{expected_validation_batches} "
                            f"({100 * val_count / expected_validation_batches:5.1f}%) "
                            f"elapsed={elapsed / 60:.1f}m ETA={eta / 60:.1f}m",
                            flush=True,
                        )
        validation_loss = val_total / val_count if val_count else None
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "validation_loss": validation_loss})
        if args.mode in {"pilot", "train"}:
            improved = validation_loss is not None and validation_loss < best_validation
            if improved:
                best_validation = float(validation_loss)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            checkpoint = {
                "model_contract": model_contract,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "epoch": epoch + 1,
                "best_validation_loss": best_validation,
                "epochs_without_improvement": epochs_without_improvement,
                "config": config,
                "history": history,
            }
            prefix = "pilot_" if args.mode == "pilot" else ""
            torch.save(checkpoint, output / f"{prefix}last.pt")
            if improved:
                torch.save(checkpoint, output / f"{prefix}best.pt")
            epoch_elapsed = time.perf_counter() - epoch_started
            remaining = max_epochs - epoch - 1
            print(
                f"Epoch {epoch + 1}/{max_epochs} complete | "
                f"train_loss={train_loss:.6f} validation_loss={validation_loss:.6f} "
                f"best={best_validation:.6f} improved={improved} "
                f"elapsed={epoch_elapsed / 60:.1f}m "
                f"max_remaining≈{epoch_elapsed * remaining / 3600:.1f}h",
                flush=True,
            )
            if epochs_without_improvement >= int(training.get("early_stopping_patience", 3)):
                print(
                    f"Early stopping after {epochs_without_improvement} epochs without improvement.",
                    flush=True,
                )
                break
    mean_batch = float(np.mean(timings)) if timings else None
    result = {"mode": args.mode, "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None, "contract": model_contract, "validation_missingness": validation_missingness, "history": history, "timed_batches": len(timings), "mean_batch_seconds": mean_batch, "estimated_epoch_seconds": mean_batch * math.ceil(len(train) / options["batch_size"]) if mean_batch else None, "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else None}
    (output / f"{args.mode}_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2)); train.close(); validation.close(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
