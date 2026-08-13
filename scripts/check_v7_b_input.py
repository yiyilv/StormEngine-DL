#!/usr/bin/env python3
"""Validate the real 239+151 V7-B input and run one model forward pass."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from check_v7 import load_config, make_model, resolve  # noqa: E402
from stormengine_dl.data import StaticFields, load_v7_b_input  # noqa: E402
from stormengine_dl.data.operational_adapter import load_fixed_registry  # noqa: E402


def main() -> int:
    config = load_config(resolve("configs/v7_b.yaml")); registry = load_fixed_registry(resolve(config["data"]["station_registry"]), include_virtual=True)
    batch = load_v7_b_input(
        resolve("data_external/meteohub/processed/20260801_20260808/official_hourly_physical.npz"),
        resolve("data_external/open_meteo/processed/20260801_20260808/icon2i_hourly_marine.npz"),
        resolve("data/normalization/era5_2010_2015.json"), expected_station_ids=registry.station_ids,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model = make_model(config).to(device).eval()
    static = StaticFields.load(resolve(config["data"]["static_fields"])).as_tensor().unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(
            torch.from_numpy(batch.values[:12]).unsqueeze(0).to(device),
            torch.from_numpy(batch.coordinates).unsqueeze(0).to(device),
            torch.from_numpy(batch.value_mask[:12]).unsqueeze(0).to(device), 6,
            observation_age=torch.from_numpy(batch.observation_age[:12]).unsqueeze(0).to(device),
            static_fields=static,
            point_static=torch.from_numpy(batch.station_static).unsqueeze(0).to(device),
        )
    result={"times":len(batch.times),"values_shape":list(batch.values.shape),"physical_stations":batch.physical_station_count,"marine_stations":batch.marine_station_count,"variables":list(batch.variable_names),"source_counts":{"physical":batch.source_type.count("physical"),"model_derived_open_meteo":batch.source_type.count("model_derived_open_meteo")},"physical_valid_fraction":float(batch.value_mask[:,:239].mean()),"marine_valid_fraction":float(batch.value_mask[:,239:].mean()),"prediction_shape":list(prediction.shape),"finite":bool(torch.isfinite(prediction).all())}
    print(json.dumps(result,indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())
