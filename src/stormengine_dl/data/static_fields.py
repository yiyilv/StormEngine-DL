"""Static land-sea and station-distance fields used by the decoder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class StaticFields:
    latitudes: np.ndarray
    longitudes: np.ndarray
    land_sea_mask: np.ndarray
    station_distance: np.ndarray

    @classmethod
    def load(cls, path: str | Path) -> "StaticFields":
        with np.load(path) as values:
            return cls(
                values["latitudes"].astype(np.float32),
                values["longitudes"].astype(np.float32),
                values["land_sea_mask"].astype(np.float32),
                values["station_distance"].astype(np.float32),
            )

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            latitudes=self.latitudes,
            longitudes=self.longitudes,
            land_sea_mask=self.land_sea_mask,
            station_distance=self.station_distance,
        )

    def as_tensor(self) -> torch.Tensor:
        return torch.from_numpy(
            np.stack((self.land_sea_mask, self.station_distance), axis=0).astype(np.float32)
        )


def build_station_distance_field(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    station_coordinates: np.ndarray,
) -> np.ndarray:
    """Return metric-aware nearest-station distance normalized to [0, 1]."""
    grid_lat, grid_lon = np.meshgrid(latitudes, longitudes, indexing="ij")
    mean_latitude = float(np.mean(latitudes))
    km_per_lon_degree = 111.32 * np.cos(np.deg2rad(mean_latitude))
    grid = np.stack((grid_lat * 111.32, grid_lon * km_per_lon_degree), axis=-1)
    stations = np.stack(
        (station_coordinates[:, 0] * 111.32, station_coordinates[:, 1] * km_per_lon_degree),
        axis=-1,
    )
    distances = np.sqrt(np.square(grid[:, :, None, :] - stations[None, None, :, :]).sum(axis=-1))
    nearest = distances.min(axis=-1).astype(np.float32)
    # Station coordinates are commonly stored as float32, so an exact grid
    # coincidence can retain a sub-metre numerical residue after projection.
    nearest[np.isclose(nearest, 0.0, atol=1e-3)] = 0.0
    maximum = float(nearest.max())
    return nearest / maximum if maximum > 0 else np.zeros_like(nearest)
