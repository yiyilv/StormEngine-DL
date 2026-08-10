"""Parse and audit MeteoHub JSON Lines observations without losing BUFR semantics."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence


@dataclass(frozen=True)
class VariableSpec:
    canonical_name: str
    raw_unit: str
    canonical_unit: str
    scale: float = 1.0
    offset: float = 0.0


# WMO BUFR Table B quantities that occur in the retained MeteoHub samples.
# Local/uncertain descriptors deliberately remain unmapped and are still emitted.
VARIABLE_SPECS: Mapping[str, VariableSpec] = {
    "B10004": VariableSpec("pressure", "Pa", "hPa", 0.01),
    "B11001": VariableSpec("wind_direction", "degree", "degree"),
    "B11002": VariableSpec("wind_speed", "m s-1", "m s-1"),
    "B11041": VariableSpec("wind_gust_max", "m s-1", "m s-1"),
    "B11043": VariableSpec("wind_gust_direction_max", "degree", "degree"),
    "B12101": VariableSpec("air_temperature", "K", "degree_Celsius", 1.0, -273.15),
    "B13003": VariableSpec("relative_humidity", "%", "%"),
    "B13011": VariableSpec("precipitation_amount", "kg m-2", "mm"),
    "B13013": VariableSpec("total_snow_depth", "m", "m"),
    "B13215": VariableSpec("river_level", "m", "m"),
    "B14198": VariableSpec("global_visible_irradiance_downward", "W m-2", "W m-2"),
}


@dataclass(frozen=True)
class Observation:
    station_id: str
    station_name: str
    network: str
    latitude: float
    longitude: float
    elevation_m: float | None
    observation_time: str
    bufr_code: str
    canonical_variable: str
    raw_value: float
    canonical_value: float
    raw_unit: str
    canonical_unit: str
    timerange_indicator: int | None
    timerange_start_seconds: int | None
    timerange_end_seconds: int | None
    aggregation_seconds: int | None
    level_type: int | None
    level_value_raw: float | None
    height_above_ground_m: float | None
    source_file: str

    @property
    def identity(self) -> tuple[object, ...]:
        """Stable identity used to remove overlap between extraction files."""
        return (
            self.station_id,
            self.observation_time,
            self.bufr_code,
            self.timerange_indicator,
            self.timerange_start_seconds,
            self.timerange_end_seconds,
            self.level_type,
            self.level_value_raw,
        )


def _station_id(network: str, latitude: float, longitude: float) -> str:
    identity = f"meteohub|{network}|{latitude:.6f}|{longitude:.6f}"
    digest = hashlib.sha1(identity.encode()).hexdigest()[:12]
    return f"MH::{network}::{digest}"


def _metadata(record: Mapping[str, object]) -> tuple[str, float, float, float | None]:
    station_name = "unnamed station"
    latitude = float(record["lat"]) / 100000.0
    longitude = float(record["lon"]) / 100000.0
    elevation: float | None = None
    for block in record.get("data", []):  # type: ignore[union-attr]
        if "timerange" in block or "level" in block:
            continue
        values = block.get("vars", {})
        if values.get("B01019", {}).get("v") is not None:
            station_name = str(values["B01019"]["v"])
        if values.get("B05001", {}).get("v") is not None:
            latitude = float(values["B05001"]["v"])
        if values.get("B06001", {}).get("v") is not None:
            longitude = float(values["B06001"]["v"])
        if values.get("B07030", {}).get("v") is not None:
            elevation = float(values["B07030"]["v"])
    return station_name, latitude, longitude, elevation


def parse_meteohub_record(record: Mapping[str, object], source_file: str) -> list[Observation]:
    """Expand one MeteoHub JSONL record into one row per measured variable."""
    network = str(record.get("network", ""))
    station_name, latitude, longitude, elevation = _metadata(record)
    station_id = _station_id(network, latitude, longitude)
    observation_time = str(record["date"])
    observations: list[Observation] = []
    for block in record.get("data", []):  # type: ignore[union-attr]
        if "timerange" not in block and "level" not in block:
            continue
        timerange = block.get("timerange") or [None, None, None]
        level = block.get("level") or [None, None, None, None]
        indicator = int(timerange[0]) if timerange[0] is not None else None
        start = int(timerange[1]) if timerange[1] is not None else None
        end = int(timerange[2]) if timerange[2] is not None else None
        aggregation = end - start if indicator in {1, 2} and start is not None and end is not None else None
        level_type = int(level[0]) if level[0] is not None else None
        level_value = float(level[1]) if level[1] is not None else None
        height_m = level_value / 1000.0 if level_type == 103 and level_value is not None else None
        for code, wrapped in block.get("vars", {}).items():
            if wrapped.get("v") is None or not isinstance(wrapped.get("v"), (int, float)):
                continue
            raw_value = float(wrapped["v"])
            spec = VARIABLE_SPECS.get(str(code))
            canonical_name = spec.canonical_name if spec else f"unmapped_{code}"
            raw_unit = spec.raw_unit if spec else ""
            canonical_unit = spec.canonical_unit if spec else ""
            canonical_value = raw_value * spec.scale + spec.offset if spec else raw_value
            observations.append(
                Observation(
                    station_id=station_id,
                    station_name=station_name,
                    network=network,
                    latitude=latitude,
                    longitude=longitude,
                    elevation_m=elevation,
                    observation_time=observation_time,
                    bufr_code=str(code),
                    canonical_variable=canonical_name,
                    raw_value=raw_value,
                    canonical_value=canonical_value,
                    raw_unit=raw_unit,
                    canonical_unit=canonical_unit,
                    timerange_indicator=indicator,
                    timerange_start_seconds=start,
                    timerange_end_seconds=end,
                    aggregation_seconds=aggregation,
                    level_type=level_type,
                    level_value_raw=level_value,
                    height_above_ground_m=height_m,
                    source_file=source_file,
                )
            )
    return observations


def iter_meteohub_observations(paths: Iterable[str | Path]) -> Iterator[Observation]:
    for value in paths:
        path = Path(value)
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
                yield from parse_meteohub_record(record, path.name)


def _iso_timestamp(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def audit_observations(
    observations: Iterable[Observation],
    selected_station_ids: set[str] | None = None,
    *,
    include_observations: bool = False,
) -> dict[str, list[dict[str, object]] | dict[str, object]]:
    """Deduplicate observations and build audit tables for downstream CSV output."""
    unique: dict[tuple[object, ...], Observation] = {}
    raw_count = 0
    corrected_overlap_count = 0
    source_files: set[str] = set()
    for observation in observations:
        raw_count += 1
        source_files.add(observation.source_file)
        previous = unique.get(observation.identity)
        if previous is not None and previous.raw_value != observation.raw_value:
            corrected_overlap_count += 1
        # Later extraction files win because MeteoHub observations may be revised.
        unique[observation.identity] = observation

    rows = list(unique.values())
    by_variable: dict[str, list[Observation]] = defaultdict(list)
    by_station_variable: dict[tuple[str, str], list[Observation]] = defaultdict(list)
    timeranges: Counter[tuple[object, ...]] = Counter()
    stations: dict[str, Observation] = {}
    station_variables: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_variable[row.bufr_code].append(row)
        by_station_variable[(row.station_id, row.bufr_code)].append(row)
        timeranges[(row.bufr_code, row.timerange_indicator, row.timerange_start_seconds,
                    row.timerange_end_seconds, row.aggregation_seconds)] += 1
        stations.setdefault(row.station_id, row)
        station_variables[row.station_id].add(row.bufr_code)

    variable_summary: list[dict[str, object]] = []
    for code, group in sorted(by_variable.items()):
        sample = group[0]
        variable_summary.append({
            "bufr_code": code,
            "canonical_variable": sample.canonical_variable,
            "canonical_unit": sample.canonical_unit,
            "observation_count": len(group),
            "station_count": len({row.station_id for row in group}),
            "selected_station_count": len({row.station_id for row in group if selected_station_ids is not None and row.station_id in selected_station_ids}),
            "first_time": min(row.observation_time for row in group),
            "last_time": max(row.observation_time for row in group),
        })

    station_variable_summary: list[dict[str, object]] = []
    for (station_id, code), group in sorted(by_station_variable.items()):
        sample = group[0]
        timestamps = sorted({_iso_timestamp(row.observation_time) for row in group})
        intervals = [right - left for left, right in zip(timestamps, timestamps[1:]) if right > left]
        station_variable_summary.append({
            "station_id": station_id,
            "station_name": sample.station_name,
            "network": sample.network,
            "latitude": sample.latitude,
            "longitude": sample.longitude,
            "selected_for_project": selected_station_ids is not None and station_id in selected_station_ids,
            "bufr_code": code,
            "canonical_variable": sample.canonical_variable,
            "observation_count": len(group),
            "unique_timestamp_count": len(timestamps),
            "first_time": min(row.observation_time for row in group),
            "last_time": max(row.observation_time for row in group),
            "median_interval_seconds": statistics.median(intervals) if intervals else "",
        })

    station_summary = [{
        "station_id": station_id,
        "station_name": sample.station_name,
        "network": sample.network,
        "latitude": sample.latitude,
        "longitude": sample.longitude,
        "elevation_m": sample.elevation_m if sample.elevation_m is not None else "",
        "selected_for_project": selected_station_ids is not None and station_id in selected_station_ids,
        "variable_count": len(station_variables[station_id]),
        "variables": "|".join(sorted(station_variables[station_id])),
    } for station_id, sample in sorted(stations.items())]

    timerange_summary = [{
        "bufr_code": key[0],
        "timerange_indicator": key[1],
        "timerange_start_seconds": key[2],
        "timerange_end_seconds": key[3],
        "aggregation_seconds": key[4],
        "observation_count": count,
    } for key, count in sorted(timeranges.items(), key=lambda item: tuple(str(v) for v in item[0]))]

    summary = {
        "source_files": sorted(source_files),
        "raw_measurement_count": raw_count,
        "unique_measurement_count": len(rows),
        "duplicate_measurement_count": raw_count - len(rows),
        "corrected_overlap_count": corrected_overlap_count,
        "station_count": len(stations),
        "selected_station_count": len(set(stations) & selected_station_ids) if selected_station_ids is not None else None,
        "unmapped_bufr_codes": sorted(code for code in by_variable if code not in VARIABLE_SPECS),
    }
    report: dict[str, list[dict[str, object]] | dict[str, object]] = {
        "summary": summary,
        "variable_summary": variable_summary,
        "station_variable_summary": station_variable_summary,
        "station_summary": station_summary,
        "timerange_summary": timerange_summary,
    }
    if include_observations:
        report["observations"] = [asdict(row) for row in rows]
    return report
