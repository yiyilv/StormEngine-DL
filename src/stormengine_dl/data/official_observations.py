"""Parse and audit MeteoHub JSON Lines observations without losing BUFR semantics."""

from __future__ import annotations

import hashlib
import json
import sqlite3
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


_SQLITE_COLUMNS = tuple(Observation.__dataclass_fields__)
_SQLITE_TYPES = {
    "latitude": "REAL",
    "longitude": "REAL",
    "elevation_m": "REAL",
    "raw_value": "REAL",
    "canonical_value": "REAL",
    "timerange_indicator": "INTEGER",
    "timerange_start_seconds": "INTEGER",
    "timerange_end_seconds": "INTEGER",
    "aggregation_seconds": "INTEGER",
    "level_type": "INTEGER",
    "level_value_raw": "REAL",
    "height_above_ground_m": "REAL",
}


def _identity_digest(observation: Observation) -> bytes:
    encoded = json.dumps(observation.identity, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).digest()


def _observation_values(observation: Observation) -> tuple[object, ...]:
    return (_identity_digest(observation),) + tuple(
        getattr(observation, name) for name in _SQLITE_COLUMNS
    )


def _weighted_median(counts: Sequence[tuple[float, int]]) -> float | str:
    total = sum(count for _, count in counts)
    if total == 0:
        return ""
    left_rank = (total - 1) // 2
    right_rank = total // 2
    cumulative = 0
    left_value: float | None = None
    for value, count in counts:
        next_cumulative = cumulative + count
        if left_value is None and left_rank < next_cumulative:
            left_value = value
        if right_rank < next_cumulative:
            assert left_value is not None
            return (left_value + value) / 2.0
        cumulative = next_cumulative
    raise AssertionError("Weighted median ranks exceeded interval counts")


def audit_observations_sqlite(
    observations: Iterable[Observation],
    database_path: str | Path,
    selected_station_ids: set[str] | None = None,
    *,
    batch_size: int = 10_000,
) -> dict[str, list[dict[str, object]] | dict[str, object]]:
    """Build the standard audit report using a disk-backed exact deduplication store."""
    path = Path(database_path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite streaming audit database: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute("PRAGMA cache_size=-131072")
        column_sql = ",\n".join(
            ["identity_key BLOB PRIMARY KEY"]
            + [f"{name} {_SQLITE_TYPES.get(name, 'TEXT')}" for name in _SQLITE_COLUMNS]
        )
        connection.execute(f"CREATE TABLE observations ({column_sql})")
        connection.execute("CREATE TABLE corrected_revisions (marker INTEGER)")
        connection.execute(
            """
            CREATE TRIGGER count_corrected_revision BEFORE UPDATE ON observations
            WHEN OLD.raw_value != NEW.raw_value
            BEGIN
                INSERT INTO corrected_revisions VALUES (1);
            END
            """
        )
        placeholders = ",".join("?" for _ in range(len(_SQLITE_COLUMNS) + 1))
        updates = ",".join(f"{name}=excluded.{name}" for name in _SQLITE_COLUMNS)
        insert_sql = (
            f"INSERT INTO observations VALUES ({placeholders}) "
            f"ON CONFLICT(identity_key) DO UPDATE SET {updates}"
        )
        raw_count = 0
        source_files: set[str] = set()
        batch: list[tuple[object, ...]] = []
        for observation in observations:
            raw_count += 1
            source_files.add(observation.source_file)
            batch.append(_observation_values(observation))
            if len(batch) >= batch_size:
                connection.executemany(insert_sql, batch)
                batch.clear()
        if batch:
            connection.executemany(insert_sql, batch)
        connection.commit()

        connection.execute(
            "CREATE INDEX observation_station_variable_time "
            "ON observations(station_id, bufr_code, observation_time)"
        )
        connection.execute(
            "CREATE INDEX observation_variable ON observations(bufr_code)"
        )
        connection.execute(
            "CREATE TEMP TABLE selected_stations (station_id TEXT PRIMARY KEY)"
        )
        if selected_station_ids:
            connection.executemany(
                "INSERT INTO selected_stations VALUES (?)",
                ((station_id,) for station_id in selected_station_ids),
            )

        interval_counts: dict[tuple[str, str], list[tuple[float, int]]] = defaultdict(list)
        interval_query = """
            WITH distinct_times AS (
                SELECT station_id, bufr_code, observation_time
                FROM observations
                GROUP BY station_id, bufr_code, observation_time
            ), intervals AS (
                SELECT station_id, bufr_code,
                       CAST(strftime('%s', observation_time) AS INTEGER) -
                       LAG(CAST(strftime('%s', observation_time) AS INTEGER)) OVER (
                           PARTITION BY station_id, bufr_code ORDER BY observation_time
                       ) AS delta
                FROM distinct_times
            )
            SELECT station_id, bufr_code, delta, COUNT(*)
            FROM intervals
            WHERE delta > 0
            GROUP BY station_id, bufr_code, delta
            ORDER BY station_id, bufr_code, delta
        """
        for station_id, code, delta, count in connection.execute(interval_query):
            interval_counts[(station_id, code)].append((float(delta), int(count)))

        variable_summary = [
            {
                "bufr_code": row[0],
                "canonical_variable": row[1],
                "canonical_unit": row[2] or "",
                "observation_count": row[3],
                "station_count": row[4],
                "selected_station_count": row[5],
                "first_time": row[6],
                "last_time": row[7],
            }
            for row in connection.execute(
                """
                SELECT o.bufr_code, MIN(o.canonical_variable), MIN(o.canonical_unit),
                       COUNT(*), COUNT(DISTINCT o.station_id),
                       COUNT(DISTINCT CASE WHEN s.station_id IS NOT NULL THEN o.station_id END),
                       MIN(o.observation_time), MAX(o.observation_time)
                FROM observations o
                LEFT JOIN selected_stations s USING (station_id)
                GROUP BY o.bufr_code
                ORDER BY o.bufr_code
                """
            )
        ]

        station_variable_summary = []
        station_variable_query = """
            SELECT o.station_id, MIN(o.station_name), MIN(o.network),
                   MIN(CAST(o.latitude AS REAL)), MIN(CAST(o.longitude AS REAL)),
                   MAX(s.station_id IS NOT NULL), o.bufr_code,
                   MIN(o.canonical_variable), COUNT(*),
                   COUNT(DISTINCT o.observation_time),
                   MIN(o.observation_time), MAX(o.observation_time)
            FROM observations o
            LEFT JOIN selected_stations s USING (station_id)
            GROUP BY o.station_id, o.bufr_code
            ORDER BY o.station_id, o.bufr_code
        """
        for row in connection.execute(station_variable_query):
            station_variable_summary.append({
                "station_id": row[0],
                "station_name": row[1],
                "network": row[2],
                "latitude": row[3],
                "longitude": row[4],
                "selected_for_project": bool(row[5]),
                "bufr_code": row[6],
                "canonical_variable": row[7],
                "observation_count": row[8],
                "unique_timestamp_count": row[9],
                "first_time": row[10],
                "last_time": row[11],
                "median_interval_seconds": _weighted_median(
                    interval_counts.get((row[0], row[6]), [])
                ),
            })

        variables_by_station: dict[str, list[str]] = defaultdict(list)
        for station_id, code in connection.execute(
            "SELECT DISTINCT station_id, bufr_code FROM observations ORDER BY station_id, bufr_code"
        ):
            variables_by_station[station_id].append(code)
        station_summary = [
            {
                "station_id": row[0],
                "station_name": row[1],
                "network": row[2],
                "latitude": row[3],
                "longitude": row[4],
                "elevation_m": row[5] if row[5] is not None else "",
                "selected_for_project": bool(row[6]),
                "variable_count": len(variables_by_station[row[0]]),
                "variables": "|".join(variables_by_station[row[0]]),
            }
            for row in connection.execute(
                """
                SELECT o.station_id, MIN(o.station_name), MIN(o.network),
                       MIN(CAST(o.latitude AS REAL)), MIN(CAST(o.longitude AS REAL)),
                       MIN(CAST(o.elevation_m AS REAL)), MAX(s.station_id IS NOT NULL)
                FROM observations o
                LEFT JOIN selected_stations s USING (station_id)
                GROUP BY o.station_id
                ORDER BY o.station_id
                """
            )
        ]

        timerange_summary = [
            {
                "bufr_code": row[0],
                "timerange_indicator": row[1],
                "timerange_start_seconds": row[2],
                "timerange_end_seconds": row[3],
                "aggregation_seconds": row[4],
                "observation_count": row[5],
            }
            for row in connection.execute(
                """
                SELECT bufr_code, timerange_indicator, timerange_start_seconds,
                       timerange_end_seconds, aggregation_seconds, COUNT(*)
                FROM observations
                GROUP BY bufr_code, timerange_indicator, timerange_start_seconds,
                         timerange_end_seconds, aggregation_seconds
                ORDER BY bufr_code, timerange_indicator, timerange_start_seconds,
                         timerange_end_seconds, aggregation_seconds
                """
            )
        ]

        unique_count = int(connection.execute("SELECT COUNT(*) FROM observations").fetchone()[0])
        corrected_count = int(
            connection.execute("SELECT COUNT(*) FROM corrected_revisions").fetchone()[0]
        )
        station_ids = {row["station_id"] for row in station_summary}
        summary = {
            "source_files": sorted(source_files),
            "raw_measurement_count": raw_count,
            "unique_measurement_count": unique_count,
            "duplicate_measurement_count": raw_count - unique_count,
            "corrected_overlap_count": corrected_count,
            "station_count": len(station_summary),
            "selected_station_count": (
                len(station_ids & selected_station_ids)
                if selected_station_ids is not None else None
            ),
            "unmapped_bufr_codes": [
                row[0] for row in connection.execute(
                    "SELECT DISTINCT bufr_code FROM observations "
                    "WHERE canonical_variable LIKE 'unmapped_%' ORDER BY bufr_code"
                )
            ],
        }
        return {
            "summary": summary,
            "variable_summary": variable_summary,
            "station_variable_summary": station_variable_summary,
            "station_summary": station_summary,
            "timerange_summary": timerange_summary,
        }
    finally:
        connection.close()
