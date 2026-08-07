"""Lazy, cross-month ERA5 sequence dataset."""

from __future__ import annotations

import csv
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset


INSTANT_VARIABLES = frozenset(("msl", "u10", "v10", "i10fg", "t2m"))
ACCUM_VARIABLES = frozenset(("ssrd", "tp"))


@dataclass(frozen=True)
class _Month:
    year: int
    month: int
    instant_path: Path
    accum_path: Path
    times: np.ndarray


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def _time_name(dataset: xr.Dataset) -> str:
    for name in ("valid_time", "time"):
        if name in dataset.coords:
            return name
    raise ValueError("ERA5 file has no valid_time or time coordinate")


def convert_era5_units(variable: str, values: np.ndarray) -> np.ndarray:
    """Convert ERA5 storage units to the physical units used by the project."""
    values = values.astype(np.float32, copy=False)
    if variable == "msl":  # Pa -> hPa
        return values / 100.0
    if variable == "t2m":  # K -> degC
        return values - 273.15
    if variable == "tp":  # m -> mm
        return values * 1000.0
    if variable == "ssrd":  # J m-2 accumulated over one hour -> W m-2
        return values / 3600.0
    return values


class Era5SequenceDataset(Dataset[dict[str, torch.Tensor]]):
    """Create station-to-grid forecast windows from monthly ERA5 files.

    The monthly files remain lazy and an LRU cache bounds memory use. Latitude is
    canonicalized south-to-north so normalized station coordinates align exactly
    with row zero through row ``H - 1`` in the target tensors.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        data_root: str | Path,
        station_coordinates: Sequence[Sequence[float]] | np.ndarray,
        *,
        input_variables: Sequence[str] = ("msl", "u10", "v10", "i10fg", "t2m"),
        target_variables: Sequence[str] = ("msl", "u10", "v10", "t2m", "tp"),
        history_hours: int = 12,
        forecast_hours: int = 6,
        years: Iterable[int] | None = None,
        station_dropout: float = 0.0,
        cache_months: int = 2,
    ) -> None:
        if history_hours < 1 or forecast_hours < 1:
            raise ValueError("history_hours and forecast_hours must be positive")
        if not 0.0 <= station_dropout < 1.0:
            raise ValueError("station_dropout must be in [0, 1)")
        if cache_months < 1:
            raise ValueError("cache_months must be positive")

        self.data_root = Path(data_root).expanduser().resolve()
        self.input_variables = tuple(input_variables)
        self.target_variables = tuple(target_variables)
        self.history_hours = history_hours
        self.forecast_hours = forecast_hours
        self.station_dropout = station_dropout
        self.cache_months = cache_months
        self._cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()
        self._requested_variables = tuple(dict.fromkeys(self.input_variables + self.target_variables))
        unknown = set(self._requested_variables) - INSTANT_VARIABLES - ACCUM_VARIABLES
        if unknown:
            raise ValueError(f"unsupported ERA5 variables: {sorted(unknown)}")

        requested_years = set(years) if years is not None else None
        self.months = self._read_months(Path(manifest_path), requested_years)
        if not self.months:
            raise ValueError("manifest contains no valid months for the requested years")

        self.times = np.concatenate([month.times for month in self.months])
        self._month_index = np.concatenate(
            [np.full(month.times.size, index, dtype=np.int32) for index, month in enumerate(self.months)]
        )
        self._local_index = np.concatenate(
            [np.arange(month.times.size, dtype=np.int32) for month in self.months]
        )
        window = history_hours + forecast_hours
        self.window_starts = np.asarray(
            [
                start
                for start in range(max(0, self.times.size - window + 1))
                if self.times[start + window - 1] - self.times[start]
                == np.timedelta64(window - 1, "h")
            ],
            dtype=np.int64,
        )

        with xr.open_dataset(self.months[0].instant_path) as dataset:
            latitudes = np.asarray(dataset.latitude.values, dtype=np.float64)
            longitudes = np.asarray(dataset.longitude.values, dtype=np.float64)
        self.latitudes = np.sort(latitudes)
        self.longitudes = np.sort(longitudes)

        stations = np.asarray(station_coordinates, dtype=np.float64)
        if stations.ndim != 2 or stations.shape[1] != 2 or stations.shape[0] == 0:
            raise ValueError("station_coordinates must have shape [N, 2] as (latitude, longitude)")
        if (
            np.any(stations[:, 0] < self.latitudes[0])
            or np.any(stations[:, 0] > self.latitudes[-1])
            or np.any(stations[:, 1] < self.longitudes[0])
            or np.any(stations[:, 1] > self.longitudes[-1])
        ):
            raise ValueError("station coordinates fall outside the ERA5 domain")
        self.station_coordinates = stations
        self._station_rows = np.abs(self.latitudes[:, None] - stations[:, 0]).argmin(axis=0)
        self._station_columns = np.abs(self.longitudes[:, None] - stations[:, 1]).argmin(axis=0)
        lat_norm = (stations[:, 0] - self.latitudes[0]) / (self.latitudes[-1] - self.latitudes[0])
        lon_norm = (stations[:, 1] - self.longitudes[0]) / (
            self.longitudes[-1] - self.longitudes[0]
        )
        self.normalized_station_coordinates = np.stack((lat_norm, lon_norm), axis=-1).astype(
            np.float32
        )

    def _read_months(self, manifest_path: Path, years: set[int] | None) -> list[_Month]:
        months: list[_Month] = []
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                year = int(row["year"])
                if not _as_bool(row["valid"]) or (years is not None and year not in years):
                    continue
                instant_path = self.data_root / row["instant_path"]
                accum_path = self.data_root / row["accum_path"]
                if not instant_path.is_file() or not accum_path.is_file():
                    raise FileNotFoundError(f"missing ERA5 pair for {year}-{int(row['month']):02d}")
                with xr.open_dataset(instant_path) as dataset:
                    times = np.asarray(dataset[_time_name(dataset)].values).astype("datetime64[ns]")
                months.append(_Month(year, int(row["month"]), instant_path, accum_path, times))
        months.sort(key=lambda item: (item.year, item.month))
        return months

    def __len__(self) -> int:
        return int(self.window_starts.size)

    def _load_month(self, month_index: int) -> dict[str, np.ndarray]:
        if month_index in self._cache:
            self._cache.move_to_end(month_index)
            return self._cache[month_index]
        month = self.months[month_index]
        arrays: dict[str, np.ndarray] = {}
        with xr.open_dataset(month.instant_path) as instant, xr.open_dataset(month.accum_path) as accum:
            for variable in self._requested_variables:
                source = instant if variable in INSTANT_VARIABLES else accum
                time_name = _time_name(source)
                values = np.asarray(
                    source[variable].transpose(time_name, "latitude", "longitude").values
                )
                if source.latitude.values[0] > source.latitude.values[-1]:
                    values = values[:, ::-1, :]
                if source.longitude.values[0] > source.longitude.values[-1]:
                    values = values[:, :, ::-1]
                arrays[variable] = convert_era5_units(variable, values)
        self._cache[month_index] = arrays
        while len(self._cache) > self.cache_months:
            self._cache.popitem(last=False)
        return arrays

    def _gather(self, indices: np.ndarray, variables: tuple[str, ...]) -> np.ndarray:
        output: list[np.ndarray] = []
        for global_index in indices:
            month_index = int(self._month_index[global_index])
            local_index = int(self._local_index[global_index])
            month = self._load_month(month_index)
            output.append(np.stack([month[name][local_index] for name in variables], axis=0))
        return np.stack(output, axis=0)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        start = int(self.window_starts[item])
        history_indices = np.arange(start, start + self.history_hours)
        target_indices = np.arange(
            start + self.history_hours,
            start + self.history_hours + self.forecast_hours,
        )
        history_grids = self._gather(history_indices, self.input_variables)
        point_values = history_grids[
            :, :, self._station_rows, self._station_columns
        ].transpose(0, 2, 1)
        target = self._gather(target_indices, self.target_variables)

        mask = torch.ones((self.history_hours, self.station_coordinates.shape[0]))
        if self.station_dropout:
            station_mask = torch.rand(self.station_coordinates.shape[0]) >= self.station_dropout
            mask *= station_mask[None]
            point_values = point_values * mask.numpy()[:, :, None]

        return {
            "point_values": torch.from_numpy(np.ascontiguousarray(point_values)).float(),
            "point_coords": torch.from_numpy(self.normalized_station_coordinates.copy()),
            "point_mask": mask,
            "target": torch.from_numpy(np.ascontiguousarray(target)).float(),
            "start_index": torch.tensor(start, dtype=torch.long),
        }

