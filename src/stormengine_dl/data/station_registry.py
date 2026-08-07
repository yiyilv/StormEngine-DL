"""Build and consume the traceable StormEngine station registry."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


PROFILE_FIELDS = {
    "land_only": "profile_land_only",
    "sea_only": "profile_sea_only",
    "dpc_plus_sea": "profile_dpc_plus_sea",
}


@dataclass(frozen=True)
class StationRecord:
    station_id: str
    station_name: str
    latitude: float
    longitude: float
    station_type: str
    network: str
    coordinate_source: str
    pretraining_value_source: str
    operational_value_source: str
    enabled: bool
    profile_land_only: bool
    profile_sea_only: bool
    profile_dpc_plus_sea: bool
    dist_to_coast_km: str
    notes: str


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _inside_domain(lat: float, lon: float, domain: tuple[float, float, float, float]) -> bool:
    lat_min, lat_max, lon_min, lon_max = domain
    return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max


def _unique_station_rows(path: Path) -> Iterable[dict[str, str]]:
    seen: set[tuple[str, str, str]] = set()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            key = (row["station_id"], row["lat"], row["lon"])
            if key not in seen:
                seen.add(key)
                yield row


def build_station_registry(
    dpc_catalog_path: str | Path,
    virtual_catalog_path: str | Path,
    output_path: str | Path,
    *,
    legacy_coastal_paths: Iterable[str | Path] = (),
    domain: tuple[float, float, float, float] = (39.0, 46.5, 12.0, 20.0),
) -> list[StationRecord]:
    """Create a catalog with explicit physical, sea, and disabled legacy classes."""
    records: list[StationRecord] = []
    dpc_path = Path(dpc_catalog_path)
    with dpc_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            lat, lon = float(row["lat"]), float(row["lon"])
            if not _inside_domain(lat, lon, domain):
                continue
            records.append(
                StationRecord(
                    station_id=f"LAND::{row['station_id']}",
                    station_name=row["station_name"],
                    latitude=lat,
                    longitude=lon,
                    station_type="physical_land",
                    network=row.get("gestore", "DPC") or "DPC",
                    coordinate_source=dpc_path.name,
                    pretraining_value_source="ERA5_sampled_at_coordinate",
                    operational_value_source="DPC_regional_observation",
                    enabled=True,
                    profile_land_only=True,
                    profile_sea_only=False,
                    profile_dpc_plus_sea=True,
                    dist_to_coast_km=row.get("dist_to_coast_km", ""),
                    notes="Physical station in the current federated DPC/regional catalog.",
                )
            )

    virtual_path = Path(virtual_catalog_path)
    for row in _unique_station_rows(virtual_path):
        lat, lon = float(row["lat"]), float(row["lon"])
        if row.get("gestore") != "VIRTUAL_ADRIATIC_SEA" or not _inside_domain(
            lat, lon, domain
        ):
            continue
        records.append(
            StationRecord(
                station_id=f"SEA::{row['station_id']}",
                station_name=row.get("sensor_name", row["station_id"]),
                latitude=lat,
                longitude=lon,
                station_type="virtual_sea",
                network="OPEN_METEO_ADRIATIC",
                coordinate_source=virtual_path.name,
                pretraining_value_source="ERA5_or_OpenMeteo_reanalysis",
                operational_value_source="OpenMeteo_forecast_and_marine",
                enabled=True,
                profile_land_only=False,
                profile_sea_only=True,
                profile_dpc_plus_sea=True,
                dist_to_coast_km="",
                notes="Virtual Adriatic coordinate; not a physical observing station.",
            )
        )

    for legacy_value in legacy_coastal_paths:
        legacy_path = Path(legacy_value)
        for row in _unique_station_rows(legacy_path):
            lat, lon = float(row["lat"]), float(row["lon"])
            if not _inside_domain(lat, lon, domain):
                continue
            records.append(
                StationRecord(
                    station_id=f"LEGACY::{legacy_path.stem}::{row['station_id']}",
                    station_name=row.get("sensor_name", row["station_id"]),
                    latitude=lat,
                    longitude=lon,
                    station_type="legacy_virtual_coastal",
                    network=row.get("gestore", "UNKNOWN"),
                    coordinate_source=legacy_path.name,
                    pretraining_value_source="disabled",
                    operational_value_source="unverified",
                    enabled=False,
                    profile_land_only=False,
                    profile_sea_only=False,
                    profile_dpc_plus_sea=False,
                    dist_to_coast_km="",
                    notes="Retained for provenance only; excluded from every training profile.",
                )
            )

    identifiers = [record.station_id for record in records]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("station registry contains duplicate station_id values")
    records.sort(key=lambda item: (item.station_type, item.latitude, item.longitude, item.station_id))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(StationRecord.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)
    return records


def load_station_coordinates(
    registry_path: str | Path,
    profile: str = "dpc_plus_sea",
) -> tuple[np.ndarray, list[dict[str, str]]]:
    """Load coordinates and metadata for one named experimental profile."""
    try:
        profile_field = PROFILE_FIELDS[profile]
    except KeyError as error:
        raise ValueError(f"unknown station profile {profile!r}; choose {sorted(PROFILE_FIELDS)}") from error
    selected: list[dict[str, str]] = []
    with Path(registry_path).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if _truthy(row["enabled"]) and _truthy(row[profile_field]):
                selected.append(row)
    if not selected:
        raise ValueError(f"station profile {profile!r} contains no enabled coordinates")
    coordinates = np.asarray(
        [[float(row["latitude"]), float(row["longitude"])] for row in selected],
        dtype=np.float64,
    )
    return coordinates, selected

