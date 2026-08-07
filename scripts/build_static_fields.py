#!/usr/bin/env python3
"""Build the V6-style LSM and nearest-station auxiliary fields."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import xarray as xr
import yaml
from shapely.geometry import Point
from shapely.ops import unary_union
from shapely.prepared import prep

from stormengine_dl.data import StaticFields, build_station_distance_field, load_station_coordinates


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(_resolve(root, args.config).read_text(encoding="utf-8"))
    data = config["data"]

    manifest_path = _resolve(root, data["era5_manifest"])
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        row = next(row for row in csv.DictReader(handle) if row["valid"].lower() == "true")
    with xr.open_dataset(_resolve(root, data["era5_root"]) / row["instant_path"]) as dataset:
        latitudes = np.sort(np.asarray(dataset.latitude.values, dtype=np.float32))
        longitudes = np.sort(np.asarray(dataset.longitude.values, dtype=np.float32))

    try:
        import cartopy.io.shapereader as shpreader
    except ImportError as error:
        raise RuntimeError("Cartopy is required only when rebuilding the land-sea mask") from error
    land_path = shpreader.natural_earth("10m", "physical", "land")
    land = prep(unary_union(list(shpreader.Reader(land_path).geometries())))
    lsm = np.asarray(
        [[float(land.covers(Point(float(lon), float(lat)))) for lon in longitudes] for lat in latitudes],
        dtype=np.float32,
    )

    coordinates, _ = load_station_coordinates(
        _resolve(root, data["station_registry"]), data["station_profile"]
    )
    distance = build_station_distance_field(latitudes, longitudes, coordinates)
    fields = StaticFields(latitudes, longitudes, lsm, distance)
    output = _resolve(root, data["static_fields"])
    fields.save(output)
    metadata = {
        "grid_shape": [int(latitudes.size), int(longitudes.size)],
        "station_profile": data["station_profile"],
        "station_count": int(coordinates.shape[0]),
        "land_source": "Natural Earth 10m physical land via Cartopy",
        "land_fraction": float(lsm.mean()),
        "distance_method": "nearest station; local equirectangular kilometres; normalized by grid maximum",
        "distance_min": float(distance.min()),
        "distance_max": float(distance.max()),
        "distance_mean": float(distance.mean()),
    }
    output.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Static fields: {output}")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
