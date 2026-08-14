"""Load one continuous ERA5/ERA5T grid as an operational evaluation target."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import xarray as xr

from .era5_dataset import ACCUM_VARIABLES, INSTANT_VARIABLES, convert_era5_units


@dataclass(frozen=True)
class Era5TargetGrid:
    """Canonical physical-unit ERA5 fields ordered south-to-north."""

    times: np.ndarray
    variable_names: tuple[str, ...]
    values: np.ndarray
    latitudes: np.ndarray
    longitudes: np.ndarray

    def indices_for(self, requested_times: np.ndarray) -> np.ndarray:
        """Return exact target indices, rejecting gaps and approximate matches."""
        requested = np.asarray(requested_times).astype("datetime64[ns]")
        positions = np.searchsorted(self.times, requested)
        safe = np.minimum(positions, len(self.times) - 1)
        if (positions >= len(self.times)).any() or not np.array_equal(self.times[safe], requested):
            available = set(self.times.tolist())
            missing = [str(value) for value in requested.tolist() if value not in available]
            raise ValueError(f"ERA5 target grid is missing exact timestamps: {missing[:5]}")
        return positions.astype(np.int64)


def _time_name(dataset: xr.Dataset) -> str:
    for name in ("valid_time", "time"):
        if name in dataset.coords:
            return name
    raise ValueError("ERA5 target file has no valid_time or time coordinate")


def _open_fields(
    path: str | Path,
    variables: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    with xr.open_dataset(path) as dataset:
        time_name = _time_name(dataset)
        times = np.asarray(dataset[time_name].values).astype("datetime64[ns]")
        latitudes = np.asarray(dataset.latitude.values, dtype=np.float64)
        longitudes = np.asarray(dataset.longitude.values, dtype=np.float64)
        if len(np.unique(times)) != len(times) or np.any(np.diff(times) != np.timedelta64(1, "h")):
            raise ValueError(f"ERA5 target time axis is not unique and continuous hourly: {path}")
        lat_order = np.argsort(latitudes)
        lon_order = np.argsort(longitudes)
        fields: dict[str, np.ndarray] = {}
        for variable in variables:
            if variable not in dataset:
                raise ValueError(f"ERA5 target file {path} is missing {variable}")
            values = np.asarray(
                dataset[variable].transpose(time_name, "latitude", "longitude").values,
                dtype=np.float32,
            )[:, lat_order][:, :, lon_order]
            values = convert_era5_units(variable, values)
            if not np.isfinite(values).all():
                raise ValueError(f"ERA5 target variable {variable} contains NaN or infinity")
            fields[variable] = values
    return times, latitudes[lat_order], longitudes[lon_order], fields


def load_era5_target_grid(
    instant_path: str | Path,
    accum_path: str | Path,
    variable_names: Sequence[str],
) -> Era5TargetGrid:
    """Load requested target variables from paired instant/accumulated files."""
    names = tuple(variable_names)
    if not names or len(names) != len(set(names)):
        raise ValueError("ERA5 target variables must be non-empty and unique")
    unknown = set(names) - INSTANT_VARIABLES - ACCUM_VARIABLES
    if unknown:
        raise ValueError(f"Unsupported ERA5 target variables: {sorted(unknown)}")
    instant_names = [name for name in names if name in INSTANT_VARIABLES]
    accum_names = [name for name in names if name in ACCUM_VARIABLES]
    loaded: dict[str, np.ndarray] = {}
    axes: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    if instant_names:
        times, latitudes, longitudes, fields = _open_fields(instant_path, instant_names)
        axes.append((times, latitudes, longitudes)); loaded.update(fields)
    if accum_names:
        times, latitudes, longitudes, fields = _open_fields(accum_path, accum_names)
        axes.append((times, latitudes, longitudes)); loaded.update(fields)
    reference = axes[0]
    for axis in axes[1:]:
        if not all(np.array_equal(left, right) for left, right in zip(reference, axis)):
            raise ValueError("ERA5 instant and accumulated files do not share identical axes")
    values = np.stack([loaded[name] for name in names], axis=1).astype(np.float32, copy=False)
    return Era5TargetGrid(
        times=reference[0], variable_names=names, values=values,
        latitudes=reference[1], longitudes=reference[2],
    )
