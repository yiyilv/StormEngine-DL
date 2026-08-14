"""Same-time, same-coordinate diagnostics for operational sources and ERA5T."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def bilinear_sample_grid(
    values: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    coordinates: np.ndarray,
) -> np.ndarray:
    """Sample ``[T,C,H,W]`` fields at ``[N,(latitude,longitude)]`` points."""
    fields = np.asarray(values, dtype=np.float32)
    lats = np.asarray(latitudes, dtype=np.float64)
    lons = np.asarray(longitudes, dtype=np.float64)
    points = np.asarray(coordinates, dtype=np.float64)
    if fields.ndim != 4 or fields.shape[-2:] != (len(lats), len(lons)):
        raise ValueError("values must have shape [T,C,H,W] matching the coordinate axes")
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
        raise ValueError("coordinates must have shape [N,2] as latitude,longitude")
    if len(lats) < 2 or len(lons) < 2 or np.any(np.diff(lats) <= 0) or np.any(np.diff(lons) <= 0):
        raise ValueError("grid axes must be strictly ascending and contain at least two values")
    if (
        (points[:, 0] < lats[0]).any() or (points[:, 0] > lats[-1]).any()
        or (points[:, 1] < lons[0]).any() or (points[:, 1] > lons[-1]).any()
    ):
        raise ValueError("sampling coordinates fall outside the ERA5T grid")

    lat_high = np.searchsorted(lats, points[:, 0], side="right").clip(1, len(lats) - 1)
    lon_high = np.searchsorted(lons, points[:, 1], side="right").clip(1, len(lons) - 1)
    lat_low, lon_low = lat_high - 1, lon_high - 1
    lat_weight = (points[:, 0] - lats[lat_low]) / (lats[lat_high] - lats[lat_low])
    lon_weight = (points[:, 1] - lons[lon_low]) / (lons[lon_high] - lons[lon_low])
    south_west = fields[:, :, lat_low, lon_low]
    south_east = fields[:, :, lat_low, lon_high]
    north_west = fields[:, :, lat_high, lon_low]
    north_east = fields[:, :, lat_high, lon_high]
    south = south_west * (1.0 - lon_weight) + south_east * lon_weight
    north = north_west * (1.0 - lon_weight) + north_east * lon_weight
    return (south * (1.0 - lat_weight) + north * lat_weight).transpose(0, 2, 1).astype(np.float32)


def paired_source_statistics(
    source: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
    variable_names: Sequence[str],
) -> dict[str, dict[str, float | int | None]]:
    """Return paired distribution and error statistics per variable."""
    left = np.asarray(source, dtype=np.float64)
    right = np.asarray(reference, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    names = tuple(variable_names)
    if left.shape != right.shape or left.shape != valid.shape or left.ndim != 3:
        raise ValueError("source, reference, and mask must share shape [T,N,C]")
    if left.shape[-1] != len(names):
        raise ValueError("variable_names does not match the channel dimension")
    output: dict[str, dict[str, float | int | None]] = {}
    for channel, name in enumerate(names):
        selected = valid[:, :, channel] & np.isfinite(left[:, :, channel]) & np.isfinite(right[:, :, channel])
        observed = left[:, :, channel][selected]
        expected = right[:, :, channel][selected]
        if observed.size == 0:
            output[name] = {"count": 0, "bias": None, "mae": None, "rmse": None, "correlation": None}
            continue
        error = observed - expected
        correlation = None
        if observed.size > 1 and observed.std() > 0 and expected.std() > 0:
            correlation = float(np.corrcoef(observed, expected)[0, 1])
        source_quantiles = np.quantile(observed, (0.05, 0.5, 0.95))
        reference_quantiles = np.quantile(expected, (0.05, 0.5, 0.95))
        output[name] = {
            "count": int(observed.size),
            "bias": float(error.mean()),
            "mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt(np.square(error).mean())),
            "correlation": correlation,
            "source_mean": float(observed.mean()),
            "source_std": float(observed.std()),
            "source_p05": float(source_quantiles[0]),
            "source_p50": float(source_quantiles[1]),
            "source_p95": float(source_quantiles[2]),
            "era5t_mean": float(expected.mean()),
            "era5t_std": float(expected.std()),
            "era5t_p05": float(reference_quantiles[0]),
            "era5t_p50": float(reference_quantiles[1]),
            "era5t_p95": float(reference_quantiles[2]),
        }
    return output
