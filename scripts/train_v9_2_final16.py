#!/usr/bin/env python3
"""Refit the frozen V9.2 strong design on all 2010--2025 ERA5 data."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from check_v7 import cache_dir, load_config, make_dataset, move, resolve  # noqa: E402
from train_v9_output_form import (  # noqa: E402
    event_context,
    event_objective,
    event_window_sample_weights,
    forward_components,
    make_model,
    model_contract,
)
from stormengine_dl.data import StaticFields  # noqa: E402
from stormengine_dl.models.v9 import load_exact_v9_checkpoint  # noqa: E402
from stormengine_dl.training import sea_weight_map, weighted_mse  # noqa: E402


YEARS = list(range(2010, 2026))


def validate_contract(config: dict) -> None:
    data = config["data"]
    if config.get("development_protocol") != "v9.2-final-production-2010-2025":
        raise ValueError("final16 production protocol is not frozen")
    if data.get("train_years") != YEARS:
        raise ValueError("final16 must train on exactly 2010--2025")
    for name in ("validation_years", "confirmation_years", "test_years"):
        if data.get(name) != []:
            raise ValueError(f"final16 {name} must be empty")
    if int(config["training"].get("fixed_refit_epochs", 0)) != 1:
        raise ValueError("final16 schedule is frozen to one complete epoch")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "train"))
    parser.add_argument("--config", default="configs/v9_2_event_aware_final16.yaml")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-batches", type=int)
    args = parser.parse_args()

    config = load_config(resolve(args.config))
    validate_contract(config)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    dataset = make_dataset(config, YEARS, augment=True)
    if len(dataset.station_ids) != 390:
        raise ValueError("final16 requires the frozen 390-point station contract")
    model = make_model(config, "autoregressive", "field")
    source = resolve(config["development"]["source_checkpoint"])
    transfer = load_exact_v9_checkpoint(
        model,
        source,
        expected_sha256=config["development"]["source_checkpoint_sha256"],
        expected_input_variables=list(config["data"]["input_variables"]),
    )
    model = model.to(device)
    static = StaticFields.load(resolve(config["data"]["static_fields"])).as_tensor()
    static = static.unsqueeze(0).to(device)
    spatial_weights = sea_weight_map(
        static[0, 0], float(config["training"]["sea_weight"])
    ).to(device)
    event = event_context(config, static, device)
    contract = model_contract(config, model, len(dataset.station_ids), transfer)
    contract["scientific_status"] = config["production_refit"]["scientific_status"]
    contract["fixed_refit_epochs"] = 1
    contract["no_internal_validation_or_test"] = True

    if args.mode == "preflight":
        loader = DataLoader(dataset, batch_size=int(config["training"]["batch_size"]))
        batch = move(next(iter(loader)), device)
        with torch.no_grad():
            prediction, _ = forward_components(model, batch, static)
            event_loss, components = event_objective(
                prediction,
                batch["target"],
                list(config["data"]["target_variables"]),
                event,
            )
        print(json.dumps({
            "mode": "preflight",
            "device": str(device),
            "years": YEARS,
            "samples": len(dataset),
            "cache": str(cache_dir(config["data"])),
            "prediction_shape": list(prediction.shape),
            "prediction_finite": bool(torch.isfinite(prediction).all()),
            "event_loss": float(event_loss),
            "event_components": {key: float(value) for key, value in components.items()},
            "contract": contract,
        }, indent=2), flush=True)
        dataset.close()
        return 0

    output = resolve(config["training"]["output_dir"]) / f"seed_{args.seed}"
    summary_path = output / "final16_summary.json"
    if summary_path.exists():
        raise FileExistsError(f"completed final16 refit already exists: {summary_path}")
    output.mkdir(parents=True, exist_ok=True)
    sample_weights, sampling_summary = event_window_sample_weights(
        dataset,
        list(config["data"]["target_variables"]),
        event,
        float(config["training"]["event_aware"]["oversampling"]["event_window_multiplier"]),
    )
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(dataset),
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        sampler=sampler,
        num_workers=int(config["training"].get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and bool(config["training"].get("mixed_precision", True)),
    )
    maximum = min(len(loader), args.max_train_batches) if args.max_train_batches else len(loader)
    started = time.perf_counter()
    model.train()
    dataset.set_epoch(0)
    totals = {"total": 0.0, "forecast": 0.0, "event": 0.0}
    batches = 0
    for raw in loader:
        if batches >= maximum:
            break
        batch = move(raw, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            prediction, _ = forward_components(model, batch, static)
            forecast_loss = weighted_mse(prediction, batch["target"], spatial_weights)
            event_loss, _ = event_objective(
                prediction,
                batch["target"],
                list(config["data"]["target_variables"]),
                event,
            )
            loss = forecast_loss + event_loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(config["training"]["gradient_clip"])
        )
        scaler.step(optimizer)
        scaler.update()
        totals["total"] += float(loss.detach())
        totals["forecast"] += float(forecast_loss.detach())
        totals["event"] += float(event_loss.detach())
        batches += 1
        every = int(config["training"].get("progress_every_batches", 50))
        if every and (batches % every == 0 or batches == maximum):
            print(
                f"final16 epoch 1/1 batch {batches}/{maximum} "
                f"loss={totals['total'] / batches:.6f}",
                flush=True,
            )

    bounded = args.max_train_batches is not None and args.max_train_batches < len(loader)
    checkpoint = {
        "model_contract": contract,
        "model_state_dict": model.state_dict(),
        "epoch": 1,
        "config": config,
        "scientific_status": "bounded_pipeline_check" if bounded else config["production_refit"]["scientific_status"],
    }
    checkpoint_name = "bounded.pt" if bounded else "final.pt"
    torch.save(checkpoint, output / checkpoint_name)
    summary = {
        "scientific_status": checkpoint["scientific_status"],
        "years_trained": YEARS,
        "epochs": 1,
        "batches": batches,
        "samples": len(dataset),
        "mean_train_loss": totals["total"] / batches,
        "mean_forecast_loss": totals["forecast"] / batches,
        "mean_event_loss": totals["event"] / batches,
        "event_window_sampling": sampling_summary,
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint": checkpoint_name,
        "contract": contract,
        "independent_accuracy_claim": False,
    }
    name = "bounded_summary.json" if bounded else "final16_summary.json"
    (output / name).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    dataset.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
