#!/usr/bin/env python3
"""Run every available 12-hour real-DPC window through frozen V7-A."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))

from check_v7 import contract, load_config, make_model, resolve  # noqa: E402
from stormengine_dl.data import StaticFields, load_dpc_v7_input  # noqa: E402
from stormengine_dl.data.operational_adapter import load_fixed_registry  # noqa: E402
from stormengine_dl.models.mask_aware import require_v7_checkpoint_contract  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v7_a.yaml")
    parser.add_argument("--checkpoint", default="artifacts/v7_a_2010_2017/best.pt")
    parser.add_argument("--input", default="data_external/meteohub/processed/20260801_20260808/official_hourly_physical.npz")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--output-dir", default="artifacts/v7_a_2010_2017/dpc_replay_20260801_20260808")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--max-windows", type=int, help="Development-only replay cap")
    args = parser.parse_args()

    config = load_config(resolve(args.config)); data = config["data"]
    registry = load_fixed_registry(resolve(data["station_registry"]), include_virtual=False)
    observations = load_dpc_v7_input(
        resolve(args.input), resolve("data/normalization/era5_2010_2015.json"),
        expected_station_ids=registry.station_ids,
    )
    if len(observations.station_ids) != 239 or set(observations.source_type) != {"physical"}:
        raise ValueError("V7-A replay accepts exactly 239 physical stations")
    history = int(data["history_hours"]); forecast = int(data["forecast_hours"])
    # Keep all six forecast hours inside the replay week (169 - 12 - 6 + 1 = 152).
    window_count = len(observations.times) - history - forecast + 1
    if args.max_windows is not None:
        window_count = min(window_count, args.max_windows)
    window_starts = range(0, window_count)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    model = make_model(config).to(device)
    saved = torch.load(resolve(args.checkpoint), map_location=device, weights_only=False)
    require_v7_checkpoint_contract(saved, contract(config, model, 239))
    model.load_state_dict(saved["model_state_dict"]); model.eval()
    static = StaticFields.load(resolve(data["static_fields"])).as_tensor().unsqueeze(0).to(device)
    coords = torch.from_numpy(observations.coordinates).unsqueeze(0).to(device)
    point_static = torch.from_numpy(observations.station_static).unsqueeze(0).to(device)
    records: list[dict[str, object]] = []; predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in window_starts:
            stop = start + history
            values = torch.from_numpy(observations.values[start:stop]).unsqueeze(0).to(device)
            mask = torch.from_numpy(observations.value_mask[start:stop]).unsqueeze(0).to(device)
            age = torch.from_numpy(observations.observation_age[start:stop]).unsqueeze(0).to(device)
            output = model(values, coords, mask, forecast, observation_age=age, static_fields=static, point_static=point_static)
            finite = bool(torch.isfinite(output).all())
            if not finite: raise RuntimeError(f"Non-finite V7-A output at window {start}")
            records.append({
                "window": start,
                "history_start": str(observations.times[start].astype("datetime64[m]")),
                "analysis_time": str(observations.times[stop - 1].astype("datetime64[m]")),
                "valid_fraction": float(mask.float().mean()),
                "stations_present_min": int(mask.any(-1).sum(-1).min()),
                "stations_present_max": int(mask.any(-1).sum(-1).max()),
                "finite": finite,
            })
            if args.save_predictions: predictions.append(output[0].float().cpu().numpy())
    output_dir = resolve(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "purpose": "runtime compatibility replay; no accuracy claim without 2026 targets", "checkpoint": str(resolve(args.checkpoint)), "windows": len(records), "all_finite": all(bool(row["finite"]) for row in records), "variables": list(data["target_variables"]), "forecast_hours": forecast, "records": records}
    (output_dir / "replay_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.save_predictions:
        np.savez_compressed(output_dir / "predictions.npz", predictions=np.stack(predictions), variables=np.asarray(data["target_variables"]))
    print(json.dumps({"output": str(output_dir / 'replay_summary.json'), "windows": len(records), "all_finite": report["all_finite"]}, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
