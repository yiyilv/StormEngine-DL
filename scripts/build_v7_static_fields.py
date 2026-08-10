#!/usr/bin/env python3
"""Build V7 static fields using only the 239 physical coastal stations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stormengine_dl.data import (  # noqa: E402
    StaticFields,
    build_station_distance_field,
    load_station_coordinates,
)


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/static/adriatic_390_fields.npz")
    parser.add_argument("--registry", default="data/stations_registry.csv")
    parser.add_argument("--identity", default="data/manifests/v7_cache_identity_2010_2017.json")
    parser.add_argument("--output", default="data/static/adriatic_physical_239_fields.npz")
    args = parser.parse_args()

    source = StaticFields.load(resolve(args.source))
    coordinates, metadata = load_station_coordinates(resolve(args.registry), "land_only")
    if len(coordinates) != 239:
        raise ValueError(f"expected 239 physical stations, found {len(coordinates)}")
    identity = json.loads(resolve(args.identity).read_text(encoding="utf-8"))
    expected_ids = [
        identity["station_ids"][index]
        for index in identity["physical_station_indices"]
    ]
    station_ids = [row["station_id"] for row in metadata]
    if station_ids != expected_ids:
        raise ValueError("land-only station order does not match the pinned cache identity")

    distance = build_station_distance_field(source.latitudes, source.longitudes, coordinates)
    fields = StaticFields(
        source.latitudes,
        source.longitudes,
        source.land_sea_mask,
        distance,
    )
    output = resolve(args.output)
    fields.save(output)
    details = {
        "grid_shape": [int(source.latitudes.size), int(source.longitudes.size)],
        "station_profile": "land_only",
        "station_count": len(metadata),
        "station_ids": station_ids,
        "cache_identity": str(Path(args.identity).as_posix()),
        "land_mask_reused_from": str(Path(args.source).as_posix()),
        "distance_method": "nearest physical station; local equirectangular kilometres; normalized by grid maximum",
        "distance_min": float(distance.min()),
        "distance_max": float(distance.max()),
        "distance_mean": float(distance.mean()),
    }
    output.with_suffix(".meta.json").write_text(
        json.dumps(details, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), **details}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
