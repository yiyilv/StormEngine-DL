#!/usr/bin/env python3
"""Run frozen V9.2 Final16 inference from aligned DPC and Open-Meteo tensors."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from check_v7 import load_config, resolve  # noqa: E402
from freeze_v9_2_final16 import EXPECTED_CHECKPOINT_SHA256  # noqa: E402
from evaluate_v9_2_final16_operational_era5t import (  # noqa: E402
    INPUTS,
    TARGETS,
    load_model,
    load_pressure,
    sha256,
)
from train_v9_output_form import forward  # noqa: E402
from stormengine_dl.data import NormalizationStats, StaticFields, load_v7_b_input  # noqa: E402
from stormengine_dl.data.operational_adapter import load_fixed_registry  # noqa: E402
from stormengine_dl.runtime import denormalize_channels  # noqa: E402


def enforce_physical_bounds(prediction: np.ndarray) -> np.ndarray:
    """Apply the frozen operational lower bound to hourly precipitation."""
    result = np.asarray(prediction, np.float32).copy()
    result[:, :, TARGETS.index("tp")] = np.maximum(
        result[:, :, TARGETS.index("tp")], 0.0
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/v9_2_event_aware_final16.yaml")
    parser.add_argument("--dpc-input", required=True)
    parser.add_argument("--dpc-msl", required=True)
    parser.add_argument("--marine-input", required=True)
    parser.add_argument("--marine-msl", required=True)
    parser.add_argument(
        "--checkpoint", default="artifacts/v9_2_event_aware_final16/seed_42/final.pt"
    )
    parser.add_argument("--output", required=True, help="Destination .npz forecast file")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = resolve(args.output)
    metadata_path = output.with_suffix(".json")
    if not args.overwrite and (output.exists() or metadata_path.exists()):
        raise FileExistsError(f"Refusing to overwrite existing prediction output: {output}")
    config = load_config(resolve(args.config))
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    normalization = NormalizationStats.load(resolve(config["data"]["normalization_stats"]))
    registry = load_fixed_registry(resolve(config["data"]["station_registry"]), include_virtual=True)
    base = load_v7_b_input(
        resolve(args.dpc_input),
        resolve(args.marine_input),
        resolve(config["data"]["normalization_stats"]),
        expected_station_ids=registry.station_ids,
    )
    pressure_values, pressure_mask, pressure_age, pressure_metadata = load_pressure(
        resolve(args.dpc_msl),
        resolve(args.marine_msl),
        times=base.times.astype("datetime64[ns]"),
        station_ids=base.station_ids,
        physical_count=base.physical_station_count,
        stats=normalization,
    )
    values = np.concatenate((pressure_values, base.values), axis=-1)
    mask = np.concatenate((pressure_mask, base.value_mask), axis=-1)
    age = np.concatenate((pressure_age, base.observation_age), axis=-1)
    if values.shape[-1] != len(INPUTS) or not np.isfinite(values).all():
        raise ValueError("Assembled six-variable operational input is invalid")

    history = int(config["data"]["history_hours"])
    forecast = int(config["data"]["forecast_hours"])
    available = len(base.times) - history - forecast + 1
    count = available if args.max_windows is None else min(available, args.max_windows)
    if count <= 0:
        raise ValueError("Input period is too short for one 12 h -> 6 h forecast")
    starts = np.arange(count, dtype=np.int64)
    times = base.times.astype("datetime64[ns]")
    forecast_origins = times[starts + history - 1]
    forecast_times = np.stack(
        [times[start + history : start + history + forecast] for start in starts]
    )

    static_data = StaticFields.load(resolve(config["data"]["static_fields"]))
    static = static_data.as_tensor().unsqueeze(0).to(device)
    coordinates = torch.from_numpy(base.coordinates).to(device)
    point_static = torch.from_numpy(base.station_static).to(device)
    checkpoint_path = resolve(args.checkpoint)
    checkpoint_hash = sha256(checkpoint_path)
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(
            f"Checkpoint SHA-256 changed: {checkpoint_hash}; expected {EXPECTED_CHECKPOINT_SHA256}"
        )
    model, model_contract = load_model(config, checkpoint_path, device)
    blocks: list[np.ndarray] = []
    started = time.perf_counter()
    with torch.no_grad():
        for offset in range(0, count, args.batch_size):
            batch_starts = starts[offset : offset + args.batch_size]
            size = len(batch_starts)
            batch = {
                "point_values": torch.from_numpy(
                    np.stack([values[start : start + history] for start in batch_starts])
                ).to(device),
                "value_mask": torch.from_numpy(
                    np.stack([mask[start : start + history] for start in batch_starts])
                ).to(device),
                "observation_age": torch.from_numpy(
                    np.stack([age[start : start + history] for start in batch_starts])
                ).to(device),
                "point_coords": coordinates[None].expand(size, -1, -1),
                "point_static": point_static[None].expand(size, -1, -1),
                "target": torch.empty(size, forecast, len(TARGETS), 31, 33, device=device),
            }
            normalized = forward(model, batch, static)
            if not torch.isfinite(normalized).all():
                raise RuntimeError("V9.2 Final16 produced non-finite output")
            blocks.append(
                denormalize_channels(normalized, list(TARGETS), normalization)
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            print(f"V9.2 Final16 inference {min(offset + size, count)}/{count}", flush=True)
    prediction = enforce_physical_bounds(np.concatenate(blocks, axis=0))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        forecast_origins=forecast_origins,
        forecast_times=forecast_times,
        variable_names=np.asarray(TARGETS),
        values=prediction,
        latitudes=static_data.latitudes,
        longitudes=static_data.longitudes,
    )
    metadata = {
        "schema_version": 1,
        "scientific_status": "frozen_v9_2_final16_operational_prediction_no_truth",
        "checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint_hash},
        "model_contract": model_contract,
        "input_period": [str(times[0]), str(times[-1])],
        "forecast_windows": count,
        "forecast_shape": list(prediction.shape),
        "variables": list(TARGETS),
        "units": {"msl": "hPa", "u10": "m/s", "v10": "m/s", "t2m": "degC", "tp": "mm/hour"},
        "postprocessing": {"tp": "max(raw_hourly_tp_mm, 0)"},
        "input_files": {
            name: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for name, path in {
                "dpc_input": resolve(args.dpc_input),
                "dpc_msl": resolve(args.dpc_msl),
                "marine_input": resolve(args.marine_input),
                "marine_msl": resolve(args.marine_msl),
            }.items()
        },
        "input_coverage": {
            "overall": float(mask.mean()),
            "physical": float(mask[:, : base.physical_station_count].mean()),
            "marine": float(mask[:, base.physical_station_count :].mean()),
            "pressure_sources": pressure_metadata,
        },
        "finite_output": bool(np.isfinite(prediction).all()),
        "elapsed_seconds": time.perf_counter() - started,
        "accuracy_claim": False,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Forecast: {output}")
    print(f"Metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
