#!/usr/bin/env python3
"""Evaluate frozen V7-A on 2017 under clean or reproducible missing inputs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from check_v7 import contract, forward, load_config, make_dataset, make_model, move, resolve, strategy  # noqa: E402
from stormengine_dl.baselines import dense_grid_persistence  # noqa: E402
from stormengine_dl.data import MissingnessStrategy, NormalizationStats, StaticFields  # noqa: E402
from stormengine_dl.models.mask_aware import require_v7_checkpoint_contract  # noqa: E402
from stormengine_dl.runtime import denormalize_channels  # noqa: E402
from stormengine_dl.training import ForecastMetricAccumulator  # noqa: E402


def _last_available(values: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the newest valid value and availability for every batch/station/channel."""
    latest = torch.zeros_like(values[:, 0])
    available = torch.zeros_like(mask[:, 0])
    for hour in range(values.shape[1]):
        take = mask[:, hour]
        latest = torch.where(take, values[:, hour], latest)
        available |= take
    return latest, available


def _masked_idw(
    values: torch.Tensor, mask: torch.Tensor, coords: torch.Tensor,
    height: int, width: int, forecast_hours: int, target_variables: list[str],
    input_variables: list[str],
) -> torch.Tensor:
    """Input-fair IDW persistence which ignores missing station-variable cells."""
    grid_lat = torch.linspace(0, 1, height, device=values.device)
    grid_lon = torch.linspace(0, 1, width, device=values.device)
    lat, lon = torch.meshgrid(grid_lat, grid_lon, indexing="ij")
    grid = torch.stack((lat, lon), -1).reshape(-1, 2)
    distance = (grid[None, :, None] - coords[:, None]).square().sum(-1).clamp_min(1e-6)
    base_weights = distance.reciprocal()
    output = torch.zeros(values.shape[0], len(target_variables), height * width, device=values.device)
    for output_channel, name in enumerate(target_variables):
        if name not in input_variables:
            continue
        channel = input_variables.index(name)
        weights = base_weights * mask[:, None, :, channel]
        denominator = weights.sum(-1).clamp_min(1e-8)
        output[:, output_channel] = (weights * values[:, None, :, channel]).sum(-1) / denominator
    grid_output = output.reshape(values.shape[0], len(target_variables), height, width)
    return dense_grid_persistence(grid_output, forecast_hours)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v7_a.yaml")
    parser.add_argument("--checkpoint", default="artifacts/v7_a_2010_2017/best.pt")
    parser.add_argument("--scenario", choices=("clean", "missing"), default="clean")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    config = load_config(resolve(args.config))
    config["seed"] = args.seed
    test = make_dataset(config, config["data"]["test_years"], augment=args.scenario == "missing")
    if args.scenario == "clean":
        test.strategy = MissingnessStrategy({})
    else:
        test.strategy = strategy(config)
    if len(test.station_ids) != 239:
        raise ValueError("V7-A evaluation requires exactly 239 physical stations")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    loader = DataLoader(test, batch_size=int(config["training"]["batch_size"]), shuffle=False, num_workers=0)
    model = make_model(config).to(device)
    saved = torch.load(resolve(args.checkpoint), map_location=device, weights_only=False)
    expected = contract(config, model, len(test.station_ids))
    require_v7_checkpoint_contract(saved, expected)
    model.load_state_dict(saved["model_state_dict"]); model.eval()
    static_data = StaticFields.load(resolve(config["data"]["static_fields"]))
    static = static_data.as_tensor().unsqueeze(0).to(device)
    normalization = NormalizationStats.load(resolve("data/normalization/era5_2010_2015.json"))
    targets = list(config["data"]["target_variables"]); inputs = list(config["data"]["input_variables"])
    hours = int(config["data"]["forecast_hours"])
    metrics = {name: ForecastMetricAccumulator(tuple(targets), hours) for name in ("v7_a", "dense_persistence", "sparse_idw_persistence")}
    land = torch.from_numpy(static_data.land_sea_mask)
    processed = 0; started = time.time()
    with torch.no_grad():
        for index, raw in enumerate(loader):
            if args.max_batches is not None and index >= args.max_batches: break
            batch = move(raw, device)
            prediction = denormalize_channels(forward(model, batch, static), targets, normalization)
            target = denormalize_channels(batch["target"], targets, normalization)
            starts = raw["start_index"].numpy() + int(config["data"]["history_hours"]) - 1
            dense_now = torch.from_numpy(np.asarray(test.target_grids[starts], dtype=np.float32)).to(device)
            dense = denormalize_channels(dense_now[:, None], targets, normalization)[:, 0]
            latest, available = _last_available(batch["point_values"], batch["value_mask"])
            latest_physical = latest.detach().float().cpu().clone()
            for channel, name in enumerate(inputs):
                stat = normalization.variables[name]
                latest_physical[:, :, channel] = latest_physical[:, :, channel] * stat.std + stat.mean
            sparse = _masked_idw(latest_physical.to(device), available, batch["point_coords"], static.shape[-2], static.shape[-1], hours, targets, inputs)
            # V7-A has no pressure input. Use the train-only ERA5 climatological
            # mean instead of the physically meaningless 0 hPa fallback.
            if "msl" not in inputs and "msl" in targets:
                sparse[:, :, targets.index("msl")] = normalization.variables["msl"].mean
            metrics["v7_a"].update(prediction, target, land)
            metrics["dense_persistence"].update(dense_grid_persistence(dense, hours), target, land)
            metrics["sparse_idw_persistence"].update(sparse, target, land)
            processed += target.shape[0]
            if (index + 1) % 100 == 0: print(f"{index + 1}/{len(loader)} batches", flush=True)
    result = {"schema_version": 1, "model": "V7-A", "scenario": args.scenario, "seed": args.seed, "years": list(config["data"]["test_years"]), "samples": processed, "elapsed_seconds": time.time() - started, "baseline_notes": {"dense_persistence": "last full ERA5 grid; uses more information than V7-A", "sparse_idw_persistence": "latest available sparse inputs; msl uses the 2010-2015 train-only climatological mean because pressure is absent from V7-A inputs"}, "metrics": {name: value.compute() for name, value in metrics.items()}}
    output = resolve(args.output_dir or f"artifacts/v7_a_2010_2017/evaluation_2017_{args.scenario}_seed{args.seed}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output / 'metrics.json'), "samples": processed}, indent=2))
    test.close(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
