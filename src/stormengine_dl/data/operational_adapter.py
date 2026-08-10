"""Convert official point observations into leakage-safe fixed-registry tensors."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .official_observations import Observation


DEFAULT_OPERATIONAL_VARIABLES = (
    "station_pressure_hpa",
    "u10",
    "v10",
    "wind_gust_max",
    "t2m",
    "relative_humidity",
    "tp",
)


@dataclass(frozen=True)
class FixedRegistry:
    station_ids: tuple[str, ...]
    source_station_ids: tuple[str, ...]
    coordinates: np.ndarray
    station_static: np.ndarray
    source_type: tuple[str, ...]


@dataclass(frozen=True)
class OperationalTensorBatch:
    times: np.ndarray
    station_ids: tuple[str, ...]
    variable_names: tuple[str, ...]
    values: np.ndarray
    value_mask: np.ndarray
    source_time: np.ndarray
    observation_age_minutes: np.ndarray
    station_present: np.ndarray
    coordinates: np.ndarray
    station_static: np.ndarray
    source_type: tuple[str, ...]
    diagnostics: dict[str, int]


@dataclass(frozen=True)
class _PrecipitationSegment:
    start_ns: int
    end_ns: int
    value_mm: float
    source_ns: int


def meteorological_wind_to_uv(speed: float, direction_degrees: float) -> tuple[float, float]:
    """Convert wind-from direction and speed to eastward/northward components."""
    radians = math.radians(direction_degrees)
    return -speed * math.sin(radians), -speed * math.cos(radians)


def load_fixed_registry(
    registry_path: str | Path,
    *,
    include_virtual: bool = False,
) -> FixedRegistry:
    """Load enabled coordinates in the exact CSV order used by the project."""
    rows: list[dict[str, str]] = []
    with Path(registry_path).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            enabled = row["enabled"].strip().lower() in {"1", "true", "yes"}
            selected = row["profile_dpc_plus_sea"].strip().lower() in {"1", "true", "yes"}
            if not enabled or not selected:
                continue
            if not include_virtual and row["station_type"] != "physical_land":
                continue
            rows.append(row)
    if not rows:
        raise ValueError(f"No enabled stations found in registry: {registry_path}")
    station_ids = tuple(row["station_id"] for row in rows)
    if len(station_ids) != len(set(station_ids)):
        raise ValueError("Fixed registry contains duplicate station IDs")
    source_ids = tuple(
        station_id.removeprefix("LAND::") if station_id.startswith("LAND::") else station_id
        for station_id in station_ids
    )
    coordinates = np.asarray(
        [[float(row["latitude"]), float(row["longitude"])] for row in rows],
        dtype=np.float32,
    )
    station_static = np.asarray(
        [
            [1.0, 0.0] if row["station_type"] == "physical_land" else [0.0, 1.0]
            for row in rows
        ],
        dtype=np.float32,
    )
    return FixedRegistry(
        station_ids=station_ids,
        source_station_ids=source_ids,
        coordinates=coordinates,
        station_static=station_static,
        source_type=tuple(row["station_type"] for row in rows),
    )


def _timestamp_ns(value: str) -> int:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1_000_000_000)


def _as_hourly_ns(times: Sequence[object] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(times, dtype="datetime64[ns]")
    if values.ndim != 1 or values.size == 0:
        raise ValueError("hourly_times must be a non-empty one-dimensional sequence")
    if np.any(np.diff(values.astype(np.int64)) <= 0):
        raise ValueError("hourly_times must be strictly increasing")
    if np.any(values != values.astype("datetime64[h]").astype("datetime64[ns]")):
        raise ValueError("hourly_times must fall exactly on UTC hour boundaries")
    return values, values.astype(np.int64)


def _is_height(observation: Observation, expected_m: float) -> bool:
    return (
        observation.height_above_ground_m is not None
        and math.isclose(observation.height_above_ground_m, expected_m, abs_tol=0.05)
    )


def _instantaneous(observation: Observation) -> bool:
    return observation.timerange_indicator == 254


def _precipitation_interval(observation: Observation) -> tuple[int, int] | None:
    if (
        observation.bufr_code != "B13011"
        or observation.timerange_indicator not in {1, 2}
        or observation.timerange_start_seconds is None
        or observation.timerange_end_seconds is None
    ):
        return None
    start_offset = observation.timerange_start_seconds
    end_offset = observation.timerange_end_seconds
    if start_offset < 0 or end_offset <= start_offset:
        return None
    source_ns = _timestamp_ns(observation.observation_time)
    start_ns = source_ns - end_offset * 1_000_000_000
    end_ns = source_ns - start_offset * 1_000_000_000
    return start_ns, end_ns


def _resolve_precipitation_hour(
    segments: list[_PrecipitationSegment],
    hour_end_ns: int,
) -> tuple[float, int] | None:
    hour_start_ns = hour_end_ns - 3_600_000_000_000
    exact = [
        segment for segment in segments
        if segment.start_ns == hour_start_ns and segment.end_ns == hour_end_ns
    ]
    if exact:
        chosen = max(exact, key=lambda segment: segment.source_ns)
        return chosen.value_mm, chosen.source_ns
    ordered = sorted(
        (
            segment for segment in segments
            if segment.start_ns >= hour_start_ns and segment.end_ns <= hour_end_ns
        ),
        key=lambda segment: (segment.start_ns, segment.end_ns, segment.source_ns),
    )
    cursor = hour_start_ns
    total = 0.0
    latest_source = hour_start_ns
    for segment in ordered:
        if segment.start_ns != cursor or segment.end_ns <= segment.start_ns:
            return None
        total += segment.value_mm
        cursor = segment.end_ns
        latest_source = max(latest_source, segment.source_ns)
    if cursor != hour_end_ns:
        return None
    return total, latest_source


def build_operational_tensors(
    observations: Iterable[Observation],
    registry_path: str | Path,
    hourly_times: Sequence[object] | np.ndarray,
    *,
    variable_names: Sequence[str] = DEFAULT_OPERATIONAL_VARIABLES,
    max_age_minutes: float = 60.0,
    include_virtual: bool = False,
) -> OperationalTensorBatch:
    """Align official observations without using values later than each target hour."""
    if max_age_minutes < 0:
        raise ValueError("max_age_minutes must be non-negative")
    variables = tuple(variable_names)
    if len(variables) != len(set(variables)):
        raise ValueError("variable_names contains duplicates")
    registry = load_fixed_registry(registry_path, include_virtual=include_virtual)
    times, time_ns = _as_hourly_ns(hourly_times)
    station_index = {
        source_id: index for index, source_id in enumerate(registry.source_station_ids)
    }
    variable_index = {name: index for index, name in enumerate(variables)}
    shape = (times.size, len(registry.station_ids), len(variables))
    values = np.zeros(shape, dtype=np.float32)
    value_mask = np.zeros(shape, dtype=bool)
    ages = np.full(shape, -1.0, dtype=np.float32)
    source_times = np.full(shape, np.iinfo(np.int64).min, dtype=np.int64)
    max_age_ns = int(max_age_minutes * 60 * 1_000_000_000)
    pending_wind: dict[tuple[object, ...], dict[str, Observation]] = {}
    precipitation: dict[tuple[int, int], list[_PrecipitationSegment]] = {}
    diagnostics = {
        "input_observations": 0,
        "unknown_station": 0,
        "unsupported_or_ineligible": 0,
        "unpaired_wind": 0,
        "unpaired_wind_direction_only": 0,
        "unpaired_wind_speed_only": 0,
        "precipitation_candidate_hours": 0,
        "accepted_precipitation_hours": 0,
        "rejected_precipitation_hours": 0,
    }

    def assign(station: int, variable: str, value: float, source_ns: int) -> None:
        channel = variable_index.get(variable)
        if channel is None or not math.isfinite(value):
            return
        first = int(np.searchsorted(time_ns, source_ns, side="left"))
        for time_index in range(first, time_ns.size):
            age_ns = int(time_ns[time_index]) - source_ns
            if age_ns < 0:
                continue
            if age_ns > max_age_ns:
                break
            if source_ns >= source_times[time_index, station, channel]:
                values[time_index, station, channel] = value
                value_mask[time_index, station, channel] = True
                ages[time_index, station, channel] = age_ns / 60_000_000_000
                source_times[time_index, station, channel] = source_ns

    for observation in observations:
        diagnostics["input_observations"] += 1
        station = station_index.get(observation.station_id)
        if station is None:
            diagnostics["unknown_station"] += 1
            continue
        source_ns = _timestamp_ns(observation.observation_time)
        if observation.bufr_code == "B10004" and _instantaneous(observation):
            assign(station, "station_pressure_hpa", observation.canonical_value, source_ns)
        elif observation.bufr_code == "B12101" and _instantaneous(observation) and _is_height(observation, 2.0):
            assign(station, "t2m", observation.canonical_value, source_ns)
        elif observation.bufr_code == "B13003" and _instantaneous(observation) and _is_height(observation, 2.0):
            assign(station, "relative_humidity", observation.canonical_value, source_ns)
        elif (
            observation.bufr_code == "B11041"
            and observation.aggregation_seconds == 3600
            and _is_height(observation, 10.0)
        ):
            assign(station, "wind_gust_max", observation.canonical_value, source_ns)
        elif observation.bufr_code in {"B11001", "B11002"} and _instantaneous(observation) and _is_height(observation, 10.0):
            key = (
                observation.station_id,
                observation.observation_time,
                observation.timerange_indicator,
                observation.timerange_start_seconds,
                observation.timerange_end_seconds,
                observation.level_type,
                observation.level_value_raw,
            )
            pair = pending_wind.setdefault(key, {})
            pair[observation.bufr_code] = observation
            if pair.keys() >= {"B11001", "B11002"}:
                u10, v10 = meteorological_wind_to_uv(
                    pair["B11002"].canonical_value,
                    pair["B11001"].canonical_value,
                )
                assign(station, "u10", u10, source_ns)
                assign(station, "v10", v10, source_ns)
                del pending_wind[key]
        elif observation.bufr_code == "B13011":
            interval = _precipitation_interval(observation)
            if interval is None:
                diagnostics["unsupported_or_ineligible"] += 1
                continue
            start_ns, end_ns = interval
            hour_end_ns = ((end_ns + 3_600_000_000_000 - 1) // 3_600_000_000_000) * 3_600_000_000_000
            time_index = int(np.searchsorted(time_ns, hour_end_ns))
            if time_index < time_ns.size and int(time_ns[time_index]) == hour_end_ns:
                precipitation.setdefault((time_index, station), []).append(
                    _PrecipitationSegment(
                        start_ns=start_ns,
                        end_ns=end_ns,
                        value_mm=observation.canonical_value,
                        source_ns=source_ns,
                    )
                )
        else:
            diagnostics["unsupported_or_ineligible"] += 1

    diagnostics["unpaired_wind"] = len(pending_wind)
    diagnostics["unpaired_wind_direction_only"] = sum(
        set(pair) == {"B11001"} for pair in pending_wind.values()
    )
    diagnostics["unpaired_wind_speed_only"] = sum(
        set(pair) == {"B11002"} for pair in pending_wind.values()
    )
    diagnostics["precipitation_candidate_hours"] = len(precipitation)
    tp_channel = variable_index.get("tp")
    if tp_channel is not None:
        for (time_index, station), segments in precipitation.items():
            resolved = _resolve_precipitation_hour(segments, int(time_ns[time_index]))
            if resolved is None:
                diagnostics["rejected_precipitation_hours"] += 1
                continue
            value, source_ns = resolved
            diagnostics["accepted_precipitation_hours"] += 1
            values[time_index, station, tp_channel] = value
            value_mask[time_index, station, tp_channel] = True
            ages[time_index, station, tp_channel] = (
                int(time_ns[time_index]) - source_ns
            ) / 60_000_000_000
            source_times[time_index, station, tp_channel] = source_ns

    return OperationalTensorBatch(
        times=times,
        station_ids=registry.station_ids,
        variable_names=variables,
        values=values,
        value_mask=value_mask,
        source_time=source_times.astype("datetime64[ns]"),
        observation_age_minutes=ages,
        station_present=value_mask.any(axis=-1),
        coordinates=registry.coordinates,
        station_static=registry.station_static,
        source_type=registry.source_type,
        diagnostics=diagnostics,
    )
