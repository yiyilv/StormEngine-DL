"""Build a validated manifest for monthly ERA5 NetCDF files."""

from __future__ import annotations

import argparse
import calendar
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import xarray as xr


FILE_PATTERN = re.compile(
    r"^era5_std_adriatic_(?P<year>\d{4})_(?P<month>\d{2})_"
    r"(?P<kind>instant|accum)\.nc$"
)
REQUIRED_INSTANT_VARIABLES = ("msl", "u10", "v10", "i10fg", "t2m")
REQUIRED_ACCUM_VARIABLES = ("ssrd", "tp")


@dataclass(frozen=True)
class Era5MonthFiles:
    year: int
    month: int
    instant_path: Path | None = None
    accum_path: Path | None = None


@dataclass
class ManifestRow:
    year: int
    month: int
    instant_path: str
    accum_path: str
    instant_bytes: int
    accum_bytes: int
    time_steps: int
    expected_time_steps: int
    start_time: str
    end_time: str
    latitude_count: int
    longitude_count: int
    latitude_min: float
    latitude_max: float
    longitude_min: float
    longitude_max: float
    latitude_step: float
    longitude_step: float
    instant_variables: str
    accum_variables: str
    nan_count: int | str
    valid: bool
    errors: str


def scan_month_files(root: str | Path) -> list[Era5MonthFiles]:
    """Find all monthly files, retaining incomplete instant/accum pairs."""
    root = Path(root).expanduser().resolve()
    found: dict[tuple[int, int], dict[str, Path]] = {}
    for path in root.glob("era5_std_adriatic_*.nc"):
        match = FILE_PATTERN.match(path.name)
        if not match:
            continue
        year = int(match.group("year"))
        month = int(match.group("month"))
        if not 1 <= month <= 12:
            continue
        found.setdefault((year, month), {})[match.group("kind")] = path.resolve()

    return [
        Era5MonthFiles(
            year=year,
            month=month,
            instant_path=paths.get("instant"),
            accum_path=paths.get("accum"),
        )
        for (year, month), paths in sorted(found.items())
    ]


def _time_name(dataset: xr.Dataset) -> str:
    for candidate in ("valid_time", "time"):
        if candidate in dataset.coords:
            return candidate
    raise ValueError("missing time coordinate (expected valid_time or time)")


def _regular_step(values: np.ndarray, label: str) -> float:
    if values.size < 2:
        return 0.0
    differences = np.diff(values.astype(np.float64))
    if not np.allclose(differences, differences[0], rtol=0.0, atol=1e-8):
        raise ValueError(f"{label} coordinate is not regularly spaced")
    return float(abs(differences[0]))


def _hourly_contiguous(times: np.ndarray) -> bool:
    if times.size < 2:
        return True
    differences = np.diff(times.astype("datetime64[ns]"))
    return bool(np.all(differences == np.timedelta64(1, "h")))


def _count_nan(dataset: xr.Dataset, variables: Iterable[str]) -> int:
    count = 0
    for variable in variables:
        values = np.asarray(dataset[variable].values)
        if np.issubdtype(values.dtype, np.number):
            count += int(np.isnan(values).sum())
    return count


def validate_month(
    files: Era5MonthFiles,
    *,
    deep_check: bool = False,
    required_instant: Iterable[str] = REQUIRED_INSTANT_VARIABLES,
    required_accum: Iterable[str] = REQUIRED_ACCUM_VARIABLES,
) -> ManifestRow:
    """Validate one instant/accum pair and return a serializable row."""
    errors: list[str] = []
    expected_steps = calendar.monthrange(files.year, files.month)[1] * 24
    default = ManifestRow(
        year=files.year,
        month=files.month,
        # Store portable paths. The archive root belongs in configuration and
        # may differ across laptops, servers, and CI runners.
        instant_path=files.instant_path.name if files.instant_path else "",
        accum_path=files.accum_path.name if files.accum_path else "",
        instant_bytes=files.instant_path.stat().st_size if files.instant_path else 0,
        accum_bytes=files.accum_path.stat().st_size if files.accum_path else 0,
        time_steps=0,
        expected_time_steps=expected_steps,
        start_time="",
        end_time="",
        latitude_count=0,
        longitude_count=0,
        latitude_min=float("nan"),
        latitude_max=float("nan"),
        longitude_min=float("nan"),
        longitude_max=float("nan"),
        latitude_step=float("nan"),
        longitude_step=float("nan"),
        instant_variables="[]",
        accum_variables="[]",
        nan_count="not_checked",
        valid=False,
        errors="",
    )

    if files.instant_path is None:
        errors.append("missing instant file")
    if files.accum_path is None:
        errors.append("missing accum file")
    if errors:
        default.errors = "; ".join(errors)
        return default

    try:
        with xr.open_dataset(files.instant_path) as instant, xr.open_dataset(files.accum_path) as accum:
            instant_vars = sorted(instant.data_vars)
            accum_vars = sorted(accum.data_vars)
            default.instant_variables = json.dumps(instant_vars)
            default.accum_variables = json.dumps(accum_vars)

            missing_instant = sorted(set(required_instant) - set(instant_vars))
            missing_accum = sorted(set(required_accum) - set(accum_vars))
            if missing_instant:
                errors.append(f"missing instant variables: {','.join(missing_instant)}")
            if missing_accum:
                errors.append(f"missing accum variables: {','.join(missing_accum)}")

            instant_time_name = _time_name(instant)
            accum_time_name = _time_name(accum)
            instant_times = np.asarray(instant[instant_time_name].values)
            accum_times = np.asarray(accum[accum_time_name].values)
            default.time_steps = int(instant_times.size)
            if instant_times.size:
                default.start_time = str(instant_times[0])
                default.end_time = str(instant_times[-1])

            if instant_times.size != expected_steps:
                errors.append(
                    f"unexpected time count: {instant_times.size} (expected {expected_steps})"
                )
            if not _hourly_contiguous(instant_times):
                errors.append("instant timestamps are not hourly-contiguous")
            if not _hourly_contiguous(accum_times):
                errors.append("accum timestamps are not hourly-contiguous")
            if not np.array_equal(instant_times, accum_times):
                errors.append("instant and accum timestamps differ")

            for coordinate in ("latitude", "longitude"):
                if coordinate not in instant.coords or coordinate not in accum.coords:
                    errors.append(f"missing {coordinate} coordinate")
            if not any(message.startswith("missing latitude") for message in errors) and not any(
                message.startswith("missing longitude") for message in errors
            ):
                latitudes = np.asarray(instant.latitude.values)
                longitudes = np.asarray(instant.longitude.values)
                default.latitude_count = int(latitudes.size)
                default.longitude_count = int(longitudes.size)
                default.latitude_min = float(np.min(latitudes))
                default.latitude_max = float(np.max(latitudes))
                default.longitude_min = float(np.min(longitudes))
                default.longitude_max = float(np.max(longitudes))
                default.latitude_step = _regular_step(latitudes, "latitude")
                default.longitude_step = _regular_step(longitudes, "longitude")
                if not np.array_equal(latitudes, np.asarray(accum.latitude.values)):
                    errors.append("instant and accum latitude coordinates differ")
                if not np.array_equal(longitudes, np.asarray(accum.longitude.values)):
                    errors.append("instant and accum longitude coordinates differ")

            if deep_check and not missing_instant and not missing_accum:
                default.nan_count = _count_nan(instant, required_instant) + _count_nan(
                    accum, required_accum
                )
                if default.nan_count:
                    errors.append(f"data contains {default.nan_count} NaN values")
    except Exception as error:  # preserve every bad month in the manifest
        errors.append(f"read error: {type(error).__name__}: {error}")

    default.valid = not errors
    default.errors = "; ".join(errors) or "none"
    return default


def build_manifest(
    root: str | Path,
    output: str | Path,
    *,
    deep_check: bool = False,
) -> list[ManifestRow]:
    rows = [validate_month(item, deep_check=deep_check) for item in scan_month_files(root)]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(ManifestRow.__dataclass_fields__)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Directory containing monthly NetCDF files")
    parser.add_argument("--output", required=True, help="CSV manifest path")
    parser.add_argument("--deep", action="store_true", help="Read arrays and count NaN values")
    arguments = parser.parse_args()

    rows = build_manifest(arguments.root, arguments.output, deep_check=arguments.deep)
    valid = sum(row.valid for row in rows)
    invalid = len(rows) - valid
    print(f"ERA5 months discovered: {len(rows)}")
    print(f"Valid pairs: {valid}")
    print(f"Invalid pairs: {invalid}")
    print(f"Manifest: {Path(arguments.output).resolve()}")
    for row in rows:
        if not row.valid:
            print(f"  {row.year}-{row.month:02d}: {row.errors}")
    return 1 if invalid else 0


if __name__ == "__main__":
    raise SystemExit(main())
