#!/usr/bin/env python3
"""Evaluate frozen V7-A against future DPC observations at station locations."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))

from check_v7 import contract, load_config, make_model, resolve  # noqa: E402
from stormengine_dl.data import NormalizationStats, StaticFields, load_dpc_v7_input  # noqa: E402
from stormengine_dl.data.operational_adapter import load_fixed_registry  # noqa: E402
from stormengine_dl.models.mask_aware import require_v7_checkpoint_contract  # noqa: E402
from stormengine_dl.runtime import denormalize_channels  # noqa: E402

EVALUATED = ("u10", "v10", "t2m", "tp")


class Metrics:
    def __init__(self, leads: int, variables: int) -> None:
        self.absolute = np.zeros((leads, variables), np.float64)
        self.squared = np.zeros((leads, variables), np.float64)
        self.count = np.zeros((leads, variables), np.int64)

    def update(self, prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> None:
        error = prediction - target
        self.absolute += np.where(mask, np.abs(error), 0).sum(axis=(0, 2))
        self.squared += np.where(mask, error**2, 0).sum(axis=(0, 2))
        self.count += mask.sum(axis=(0, 2))

    def result(self) -> dict[str, object]:
        leads: dict[str, object] = {}
        for lead in range(self.count.shape[0]):
            leads[str(lead + 1)] = {
                name: {
                    "count": int(self.count[lead, channel]),
                    "mae": float(self.absolute[lead, channel] / self.count[lead, channel]) if self.count[lead, channel] else None,
                    "rmse": float(np.sqrt(self.squared[lead, channel] / self.count[lead, channel])) if self.count[lead, channel] else None,
                }
                for channel, name in enumerate(EVALUATED)
            }
        count = self.count.sum(0); absolute = self.absolute.sum(0); squared = self.squared.sum(0)
        aggregate = {
            name: {"count": int(count[channel]), "mae": float(absolute[channel] / count[channel]) if count[channel] else None, "rmse": float(np.sqrt(squared[channel] / count[channel])) if count[channel] else None}
            for channel, name in enumerate(EVALUATED)
        }
        return {"aggregate": aggregate, "by_lead_hour": leads}


def sample_grid(fields: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Bilinearly sample [B,L,C,H,W] fields at normalized [N,lat/lon] points."""
    batch, leads, channels, height, width = fields.shape
    grid = torch.stack((coords[:, 1] * 2 - 1, coords[:, 0] * 2 - 1), -1)
    grid = grid[None, :, None].expand(batch * leads, -1, -1, -1)
    sampled = F.grid_sample(fields.reshape(batch * leads, channels, height, width), grid, mode="bilinear", padding_mode="border", align_corners=True)
    return sampled[:, :, :, 0].reshape(batch, leads, channels, coords.shape[0]).permute(0, 1, 3, 2)


def latest_persistence(values: np.ndarray, mask: np.ndarray, origin: int, history: int, channels: list[int]) -> tuple[np.ndarray, np.ndarray]:
    result = np.zeros((values.shape[1], len(channels)), np.float32); valid = np.zeros_like(result, bool)
    for output_channel, source_channel in enumerate(channels):
        candidates = [origin] if EVALUATED[output_channel] == "tp" else range(origin, origin - history, -1)
        for time_index in candidates:
            take = mask[time_index, :, source_channel] & ~valid[:, output_channel]
            result[take, output_channel] = values[time_index, take, source_channel]; valid[take, output_channel] = True
    return result, valid


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v7_a.yaml")
    parser.add_argument("--checkpoint", default="artifacts/v7_a_2010_2017/best.pt")
    parser.add_argument("--input", default="data_external/meteohub/processed/20260801_20260808/official_hourly_physical.npz")
    parser.add_argument("--output-dir", default="artifacts/v7_a_2010_2017/dpc_observation_evaluation_20260801_20260808")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    config = load_config(resolve(args.config)); data = config["data"]
    registry = load_fixed_registry(resolve(data["station_registry"]), include_virtual=False)
    model_input = load_dpc_v7_input(resolve(args.input), resolve("data/normalization/era5_2010_2015.json"), expected_station_ids=registry.station_ids)
    with np.load(resolve(args.input), allow_pickle=False) as raw:
        raw_names = tuple(str(value) for value in raw["variable_names"].tolist())
        target_channels = [raw_names.index(name) for name in EVALUATED]
        target_values = np.asarray(raw["values"][:, :, target_channels], np.float32)
        target_mask = np.asarray(raw["value_mask"][:, :, target_channels], bool)
    history = int(data["history_hours"]); leads = int(data["forecast_hours"])
    starts = np.arange(0, len(model_input.times) - history - leads + 1)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    model = make_model(config).to(device); saved = torch.load(resolve(args.checkpoint), map_location=device, weights_only=False)
    require_v7_checkpoint_contract(saved, contract(config, model, 239)); model.load_state_dict(saved["model_state_dict"]); model.eval()
    static = StaticFields.load(resolve(data["static_fields"])).as_tensor().unsqueeze(0).to(device)
    coords = torch.from_numpy(model_input.coordinates).to(device); point_static = torch.from_numpy(model_input.station_static).to(device)
    normalization = NormalizationStats.load(resolve("data/normalization/era5_2010_2015.json")); output_names = list(data["target_variables"]); output_channels = [output_names.index(name) for name in EVALUATED]
    model_all = Metrics(leads, len(EVALUATED)); model_comparable = Metrics(leads, len(EVALUATED)); persistence = Metrics(leads, len(EVALUATED)); window_rows=[]
    with torch.no_grad():
        for offset in range(0, len(starts), args.batch_size):
            batch_starts = starts[offset:offset + args.batch_size]; size = len(batch_starts)
            values = torch.from_numpy(np.stack([model_input.values[s:s+history] for s in batch_starts])).to(device)
            masks = torch.from_numpy(np.stack([model_input.value_mask[s:s+history] for s in batch_starts])).to(device)
            ages = torch.from_numpy(np.stack([model_input.observation_age[s:s+history] for s in batch_starts])).to(device)
            prediction = model(values, coords[None].expand(size,-1,-1), masks, leads, observation_age=ages, static_fields=static.expand(size,-1,-1,-1), point_static=point_static[None].expand(size,-1,-1))
            prediction = denormalize_channels(prediction, output_names, normalization).to(device)
            prediction_points = sample_grid(prediction, coords)[..., output_channels].cpu().numpy()
            future_values = np.stack([target_values[s+history:s+history+leads] for s in batch_starts])
            future_mask = np.stack([target_mask[s+history:s+history+leads] for s in batch_starts])
            base_values=[]; base_masks=[]
            for s in batch_starts:
                value, valid = latest_persistence(target_values, target_mask, int(s+history-1), history, list(range(len(EVALUATED))))
                base_values.append(np.broadcast_to(value, (leads, *value.shape))); base_masks.append(np.broadcast_to(valid, (leads, *valid.shape)))
            base_values=np.stack(base_values); base_masks=np.stack(base_masks); common=future_mask & base_masks
            model_all.update(prediction_points, future_values, future_mask); model_comparable.update(prediction_points, future_values, common); persistence.update(base_values, future_values, common)
            for local,s in enumerate(batch_starts):
                error=prediction_points[local]-future_values[local]; valid=future_mask[local]
                window_rows.append({"window":int(s),"analysis_time":str(model_input.times[s+history-1].astype("datetime64[m]")),"input_valid_fraction":float(model_input.value_mask[s:s+history].mean()),"future_target_cells":int(valid.sum()),"model_rmse_all_available":float(np.sqrt((error[valid]**2).mean())) if valid.any() else None})
            print(f"windows {offset + size}/{len(starts)}", flush=True)
    all_result=model_all.result(); comparable_result=model_comparable.result(); persistence_result=persistence.result()
    skill={name: None if persistence_result["aggregate"][name]["rmse"] in (None,0) else 1-comparable_result["aggregate"][name]["rmse"]/persistence_result["aggregate"][name]["rmse"] for name in EVALUATED}
    result={"schema_version":1,"purpose":"external station-space observation evaluation; not a full-grid 2026 evaluation","windows":len(starts),"history_hours":history,"forecast_hours":leads,"station_count":len(model_input.station_ids),"variables":list(EVALUATED),"excluded":{"msl":"no directly comparable DPC sea-level pressure","i10fg":"V7-A does not forecast gust"},"future_leakage_check":"targets begin strictly after each 12-hour input cutoff","model_all_available_observations":all_result,"persistence_comparable_subset":{"model":comparable_result,"persistence":persistence_result,"rmse_skill":skill,"definition":"1 - V7-A RMSE / persistence RMSE; positive is better"}}
    output=resolve(args.output_dir); output.mkdir(parents=True,exist_ok=True); (output/"metrics.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    with (output/"window_diagnostics.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(window_rows[0])); writer.writeheader(); writer.writerows(window_rows)
    print(json.dumps({"output":str(output/"metrics.json"),"windows":len(starts),"aggregate":all_result["aggregate"],"skill":skill},indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
