"""Open-Meteo ICON-2I marine support data for the V7-B replay contract."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

OPEN_METEO_SOURCE = "model_derived_open_meteo"
SOURCE_VARIABLES = (
    "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
    "temperature_2m", "precipitation",
)
CONTRACT_VARIABLES = ("u10", "v10", "i10fg", "t2m", "tp")
PRESSURE_CONTRACT_VARIABLES = ("msl",)


@dataclass(frozen=True)
class MarinePoint:
    station_index: int
    station_id: str
    latitude: float
    longitude: float
    coordinate_source: str


@dataclass(frozen=True)
class OpenMeteoBatch:
    times: np.ndarray
    station_ids: tuple[str, ...]
    variable_names: tuple[str, ...]
    values: np.ndarray
    value_mask: np.ndarray
    observation_age: np.ndarray
    coordinates: np.ndarray
    station_static: np.ndarray
    source_type: tuple[str, ...]
    returned_coordinates: np.ndarray
    returned_elevation: np.ndarray


def load_marine_points(registry_path: str | Path) -> tuple[MarinePoint, ...]:
    with Path(registry_path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = tuple(
        MarinePoint(index, row["station_id"], float(row["latitude"]), float(row["longitude"]), row["coordinate_source"])
        for index, row in enumerate(rows)
        if row["station_type"] == "virtual_sea"
        and row["enabled"].strip().lower() == "true"
        and row["profile_sea_only"].strip().lower() == "true"
    )
    if len(selected) != 151:
        raise ValueError(f"Expected 151 enabled profile_sea_only virtual points, got {len(selected)}")
    return selected


def coordinate_manifest_bytes(points: Sequence[MarinePoint]) -> bytes:
    lines = ["station_index,station_id,latitude,longitude,coordinate_source"]
    lines.extend(f"{p.station_index},{p.station_id},{p.latitude:.8f},{p.longitude:.8f},{p.coordinate_source}" for p in points)
    return ("\n".join(lines) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def wind_components(speed: np.ndarray, direction_degrees: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    radians = np.deg2rad(direction_degrees)
    return -speed * np.sin(radians), -speed * np.cos(radians)


def haversine_km(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first_r = np.deg2rad(np.asarray(first, dtype=np.float64)); second_r = np.deg2rad(np.asarray(second, dtype=np.float64))
    dlat = second_r[..., 0] - first_r[..., 0]; dlon = second_r[..., 1] - first_r[..., 1]
    value = np.sin(dlat / 2) ** 2 + np.cos(first_r[..., 0]) * np.cos(second_r[..., 0]) * np.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(value))


def load_download_chunks(raw_dir: str | Path, points: Sequence[MarinePoint]) -> OpenMeteoBatch:
    """Merge validated raw chunks into the unnormalised common V7 contract."""
    directory = Path(raw_dir); by_id: dict[str, dict[str, object]] = {}
    for path in sorted(directory.glob("chunk_*.json")):
        wrapper = json.loads(path.read_text(encoding="utf-8"))
        ids = list(wrapper["station_ids"]); responses = wrapper["response"]
        if isinstance(responses, dict): responses = [responses]
        if len(ids) != len(responses): raise ValueError(f"Response count mismatch in {path}")
        for station_id, response in zip(ids, responses, strict=True):
            if station_id in by_id: raise ValueError(f"Duplicate Open-Meteo station {station_id}")
            by_id[station_id] = response
    expected_ids = tuple(point.station_id for point in points)
    if set(by_id) != set(expected_ids):
        raise ValueError(f"Open-Meteo coverage mismatch: missing={sorted(set(expected_ids)-set(by_id))}")
    expected_times = np.arange(np.datetime64("2026-08-01T00"), np.datetime64("2026-08-08T01"), np.timedelta64(1, "h"))
    values = np.zeros((169, len(points), 5), np.float32); mask = np.zeros_like(values, bool)
    returned = np.zeros((len(points), 2), np.float32); elevation = np.full(len(points), np.nan, np.float32)
    for station, point in enumerate(points):
        response = by_id[point.station_id]; hourly = response["hourly"]
        times = np.asarray(hourly["time"], dtype="datetime64[h]")
        select = (times >= expected_times[0]) & (times <= expected_times[-1])
        if not np.array_equal(times[select], expected_times): raise ValueError(f"Invalid 169-hour axis for {point.station_id}")
        arrays = {name: np.asarray(hourly[name], dtype=np.float64)[select] for name in SOURCE_VARIABLES}
        wind_valid = np.isfinite(arrays["wind_speed_10m"]) & np.isfinite(arrays["wind_direction_10m"])
        u, v = wind_components(arrays["wind_speed_10m"], arrays["wind_direction_10m"])
        mapped = (u, v, arrays["wind_gusts_10m"], arrays["temperature_2m"], arrays["precipitation"])
        for channel, array in enumerate(mapped):
            valid = wind_valid if channel < 2 else np.isfinite(array)
            values[:, station, channel][valid] = array[valid]; mask[:, station, channel] = valid
        returned[station] = [float(response["latitude"]), float(response["longitude"])]
        elevation[station] = float(response.get("elevation", np.nan))
    return OpenMeteoBatch(expected_times.astype("datetime64[ns]"), expected_ids, CONTRACT_VARIABLES, values, mask, np.zeros_like(values), np.asarray([[p.latitude, p.longitude] for p in points], np.float32), np.zeros((len(points), 2), np.float32), tuple(OPEN_METEO_SOURCE for _ in points), returned, elevation)


def load_download_pressure_chunks(
    raw_dir: str | Path, points: Sequence[MarinePoint]
) -> OpenMeteoBatch:
    """Extract aligned mean-sea-level pressure from validated marine chunks.

    ``pressure_msl`` is retained as the deployment-compatible ``msl`` channel.
    ``surface_pressure`` is intentionally not substituted because returned model
    cells can have non-zero terrain elevation even for requested sea coordinates.
    """
    directory = Path(raw_dir)
    by_id: dict[str, dict[str, object]] = {}
    for path in sorted(directory.glob("chunk_*.json")):
        wrapper = json.loads(path.read_text(encoding="utf-8"))
        requested = str(wrapper.get("request_parameters", {}).get("hourly", "")).split(",")
        if "pressure_msl" not in requested:
            raise ValueError(f"Raw chunk was not requested with pressure_msl: {path}")
        ids = list(wrapper["station_ids"])
        responses = wrapper["response"]
        if isinstance(responses, dict):
            responses = [responses]
        if len(ids) != len(responses):
            raise ValueError(f"Response count mismatch in {path}")
        for station_id, response in zip(ids, responses, strict=True):
            if station_id in by_id:
                raise ValueError(f"Duplicate Open-Meteo station {station_id}")
            units = response.get("hourly_units", {})
            if units.get("pressure_msl") != "hPa":
                raise ValueError(f"Unexpected pressure_msl unit for {station_id}: {units.get('pressure_msl')}")
            by_id[station_id] = response

    expected_ids = tuple(point.station_id for point in points)
    if set(by_id) != set(expected_ids):
        raise ValueError(
            f"Open-Meteo pressure coverage mismatch: missing={sorted(set(expected_ids) - set(by_id))}"
        )
    expected_times = np.arange(
        np.datetime64("2026-08-01T00"),
        np.datetime64("2026-08-08T01"),
        np.timedelta64(1, "h"),
    )
    values = np.zeros((169, len(points), 1), np.float32)
    mask = np.zeros_like(values, bool)
    returned = np.zeros((len(points), 2), np.float32)
    elevation = np.full(len(points), np.nan, np.float32)
    for station, point in enumerate(points):
        response = by_id[point.station_id]
        hourly = response["hourly"]
        times = np.asarray(hourly["time"], dtype="datetime64[h]")
        select = (times >= expected_times[0]) & (times <= expected_times[-1])
        if not np.array_equal(times[select], expected_times):
            raise ValueError(f"Invalid 169-hour pressure axis for {point.station_id}")
        pressure = np.asarray(hourly["pressure_msl"], dtype=np.float64)[select]
        valid = np.isfinite(pressure)
        values[:, station, 0][valid] = pressure[valid]
        mask[:, station, 0] = valid
        returned[station] = [float(response["latitude"]), float(response["longitude"])]
        elevation[station] = float(response.get("elevation", np.nan))

    valid_pressure = values[mask]
    if valid_pressure.size and (
        float(valid_pressure.min()) < 850.0 or float(valid_pressure.max()) > 1100.0
    ):
        raise ValueError("pressure_msl falls outside the plausible 850--1100 hPa range")
    return OpenMeteoBatch(
        expected_times.astype("datetime64[ns]"),
        expected_ids,
        PRESSURE_CONTRACT_VARIABLES,
        values,
        mask,
        np.zeros_like(values),
        np.asarray([[p.latitude, p.longitude] for p in points], np.float32),
        np.zeros((len(points), 2), np.float32),
        tuple(OPEN_METEO_SOURCE for _ in points),
        returned,
        elevation,
    )
