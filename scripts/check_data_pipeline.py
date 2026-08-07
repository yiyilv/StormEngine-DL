#!/usr/bin/env python3
"""Validate a configured ERA5-to-station dataset before training."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from stormengine_dl import StormEngineForecastModel
from stormengine_dl.data import Era5SequenceDataset, StaticFields


def _resolve(repo_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config_path = _resolve(repo_root, args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data = config["data"]

    common = dict(
        manifest_path=_resolve(repo_root, data["era5_manifest"]),
        data_root=_resolve(repo_root, data["era5_root"]),
        station_registry_path=_resolve(repo_root, data["station_registry"]),
        station_profile=data["station_profile"],
        input_variables=data["input_variables"],
        target_variables=data["target_variables"],
        history_hours=int(data["history_hours"]),
        forecast_hours=int(data["forecast_hours"]),
        normalization_path=_resolve(repo_root, data["normalization_stats"]),
    )

    datasets: dict[str, Era5SequenceDataset] = {}
    for split, field in (
        ("train", "train_years"),
        ("validation", "validation_years"),
        ("test", "test_years"),
    ):
        dataset = Era5SequenceDataset.from_station_registry(
            **common,
            years=data[field],
        )
        datasets[split] = dataset
        print(
            f"{split:10s}: years={data[field]} months={len(dataset.months):2d} "
            f"windows={len(dataset):7d} time={dataset.times[0]} -> {dataset.times[-1]}"
        )

    sample = datasets["train"][0]
    station_types = sample["point_static"].sum(dim=0).to(torch.int64).tolist()
    print("\nFirst training sample")
    print(f"  point_values: {tuple(sample['point_values'].shape)}")
    print(f"  point_coords: {tuple(sample['point_coords'].shape)}")
    print(f"  point_mask:   {tuple(sample['point_mask'].shape)}")
    print(f"  point_static: {tuple(sample['point_static'].shape)}")
    print(f"  target:       {tuple(sample['target'].shape)}")
    print(f"  station types [physical, virtual]: {station_types}")

    expected_stations = sum(station_types)
    if expected_stations != sample["point_values"].shape[1]:
        raise RuntimeError("station-type features do not match the station dimension")
    for name in ("point_values", "point_coords", "point_mask", "point_static", "target"):
        if not torch.isfinite(sample[name]).all():
            raise RuntimeError(f"non-finite values found in {name}")

    static = StaticFields.load(_resolve(repo_root, data["static_fields"]))
    static_tensor = static.as_tensor().unsqueeze(0)
    model_config = config["model"]
    model = StormEngineForecastModel(
        input_channels=len(data["input_variables"]),
        output_channels=len(data["target_variables"]),
        point_hidden=int(model_config["point_hidden"]),
        latent_channels=int(model_config["latent_channels"]),
        height=int(config["domain"]["height"]),
        width=int(config["domain"]["width"]),
        sigma=float(model_config["gaussian_sigma"]),
        processor_layers=int(model_config["processor_layers"]),
        kernel_size=int(model_config["kernel_size"]),
        static_channels=int(model_config["static_channels"]),
        point_static_channels=int(model_config["point_static_channels"]),
    )
    with torch.no_grad():
        prediction = model(
            sample["point_values"].unsqueeze(0),
            sample["point_coords"].unsqueeze(0),
            forecast_steps=int(data["forecast_hours"]),
            point_mask=sample["point_mask"].unsqueeze(0),
            static_fields=static_tensor,
            point_static=sample["point_static"].unsqueeze(0),
        )
    print("\nStatic fields and model forward")
    print(f"  static fields: {tuple(static_tensor.shape)}")
    print(f"  land fraction: {float(static.land_sea_mask.mean()):.3f}")
    print(f"  distance mean: {float(static.station_distance.mean()):.3f}")
    print(f"  prediction:    {tuple(prediction.shape)}")
    if prediction.shape != sample["target"].unsqueeze(0).shape:
        raise RuntimeError("prediction and target shapes do not match")
    if not torch.isfinite(prediction).all():
        raise RuntimeError("non-finite model prediction")

    print("\nPASS: normalized data, static fields, and full model forward are valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
