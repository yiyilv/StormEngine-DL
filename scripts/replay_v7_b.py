#!/usr/bin/env python3
"""Replay frozen V7-B with combined DPC and Open-Meteo inputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src")); sys.path.insert(0,str(ROOT/"scripts"))
from check_v7 import contract, load_config, make_model, resolve  # noqa: E402
from stormengine_dl.data import StaticFields, load_v7_b_input  # noqa: E402
from stormengine_dl.data.operational_adapter import load_fixed_registry  # noqa: E402
from stormengine_dl.models.mask_aware import require_v7_checkpoint_contract  # noqa: E402


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",default="configs/v7_b.yaml"); parser.add_argument("--checkpoint",default="artifacts/v7_b_2010_2017/best.pt"); parser.add_argument("--device",choices=("auto","cuda","cpu"),default="auto"); parser.add_argument("--output-dir",default="artifacts/v7_b_2010_2017/combined_replay_20260801_20260808"); parser.add_argument("--save-predictions",action="store_true"); parser.add_argument("--max-windows",type=int)
    args=parser.parse_args(); config=load_config(resolve(args.config)); data=config["data"]
    registry=load_fixed_registry(resolve(data["station_registry"]),include_virtual=True)
    batch=load_v7_b_input(resolve("data_external/meteohub/processed/20260801_20260808/official_hourly_physical.npz"),resolve("data_external/open_meteo/processed/20260801_20260808/icon2i_hourly_marine.npz"),resolve("data/normalization/era5_2010_2015.json"),expected_station_ids=registry.station_ids)
    history=int(data["history_hours"]); forecast=int(data["forecast_hours"]); count=len(batch.times)-history-forecast+1
    if args.max_windows is not None: count=min(count,args.max_windows)
    device=torch.device("cuda" if args.device=="auto" and torch.cuda.is_available() else ("cpu" if args.device=="auto" else args.device)); model=make_model(config).to(device)
    saved=torch.load(resolve(args.checkpoint),map_location=device,weights_only=False); require_v7_checkpoint_contract(saved,contract(config,model,390)); model.load_state_dict(saved["model_state_dict"]); model.eval()
    static=StaticFields.load(resolve(data["static_fields"])).as_tensor().unsqueeze(0).to(device); coords=torch.from_numpy(batch.coordinates).unsqueeze(0).to(device); point_static=torch.from_numpy(batch.station_static).unsqueeze(0).to(device)
    records=[]; predictions=[]
    with torch.no_grad():
        for start in range(count):
            stop=start+history; mask=torch.from_numpy(batch.value_mask[start:stop]).unsqueeze(0).to(device)
            output=model(torch.from_numpy(batch.values[start:stop]).unsqueeze(0).to(device),coords,mask,forecast,observation_age=torch.from_numpy(batch.observation_age[start:stop]).unsqueeze(0).to(device),static_fields=static,point_static=point_static)
            finite=bool(torch.isfinite(output).all())
            if not finite: raise RuntimeError(f"Non-finite V7-B output at window {start}")
            physical=mask[:,:,:239]; marine=mask[:,:,239:]
            records.append({"window":start,"analysis_time":str(batch.times[stop-1].astype("datetime64[m]")),"physical_valid_fraction":float(physical.float().mean()),"marine_valid_fraction":float(marine.float().mean()),"finite":finite})
            if args.save_predictions: predictions.append(output[0].float().cpu().numpy())
            if (start+1)%25==0 or start+1==count: print(f"windows {start+1}/{count}",flush=True)
    output_dir=resolve(args.output_dir); output_dir.mkdir(parents=True,exist_ok=True); report={"schema_version":1,"model":"V7-B","purpose":"combined DPC plus model-derived Open-Meteo runtime replay","windows":count,"station_count":390,"physical_stations":239,"marine_stations":151,"all_finite":all(row["finite"] for row in records),"records":records}; (output_dir/"replay_summary.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    if args.save_predictions: np.savez_compressed(output_dir/"predictions.npz",predictions=np.stack(predictions),variables=np.asarray(data["target_variables"]))
    print(json.dumps({"output":str(output_dir/"replay_summary.json"),"windows":count,"all_finite":report["all_finite"]},indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())
