"""Canonical V7 input contract shared by ERA5 training and DPC inference."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


V7_INPUT_VARIABLES = ("u10", "v10", "i10fg", "t2m", "tp")
DPC_SOURCE_VARIABLES = ("u10", "v10", "wind_gust_max", "t2m", "tp")


@dataclass(frozen=True)
class V7InputBatch:
    """Model-ready point observations using one stable deployment contract."""

    times: np.ndarray
    station_ids: tuple[str, ...]
    variable_names: tuple[str, ...]
    values: np.ndarray
    value_mask: np.ndarray
    observation_age: np.ndarray
    station_present: np.ndarray
    coordinates: np.ndarray
    station_static: np.ndarray
    source_type: tuple[str, ...]


def _normalize_coordinates(
    coordinates: np.ndarray,
    *,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> np.ndarray:
    if lat_max <= lat_min or lon_max <= lon_min:
        raise ValueError("Invalid V7 coordinate domain")
    result = np.asarray(coordinates, dtype=np.float32).copy()
    result[:, 0] = (result[:, 0] - lat_min) / (lat_max - lat_min)
    result[:, 1] = (result[:, 1] - lon_min) / (lon_max - lon_min)
    if not np.isfinite(result).all() or (result < 0).any() or (result > 1).any():
        raise ValueError("DPC station coordinates fall outside the V7 domain")
    return result


def adapt_dpc_to_v7(
    *,
    times: np.ndarray,
    station_ids: np.ndarray,
    source_variable_names: np.ndarray,
    source_values: np.ndarray,
    source_mask: np.ndarray,
    source_age_minutes: np.ndarray,
    coordinates: np.ndarray,
    station_static: np.ndarray,
    normalization: dict[str, object],
    expected_station_ids: tuple[str, ...] | None = None,
    lat_min: float = 39.0,
    lat_max: float = 46.5,
    lon_min: float = 12.0,
    lon_max: float = 20.0,
) -> V7InputBatch:
    """Select, normalize, and validate DPC values for the V7 model."""
    ids = tuple(str(value) for value in station_ids.tolist())
    if len(ids) != len(set(ids)):
        raise ValueError("DPC tensor contains duplicate station IDs")
    if expected_station_ids is not None and ids != expected_station_ids:
        raise ValueError("DPC station order does not match the fixed V7 registry")
    source_names = tuple(str(value) for value in source_variable_names.tolist())
    missing = set(DPC_SOURCE_VARIABLES) - set(source_names)
    if missing:
        raise ValueError(f"DPC tensor is missing V7 source variables: {sorted(missing)}")
    if source_values.shape != source_mask.shape or source_values.shape != source_age_minutes.shape:
        raise ValueError("DPC values, mask, and observation age must have identical shapes")
    if source_values.shape[:2] != (len(times), len(ids)):
        raise ValueError("DPC tensor time/station dimensions are inconsistent")

    indices = [source_names.index(name) for name in DPC_SOURCE_VARIABLES]
    values = np.stack([source_values[:, :, index] for index in indices], axis=-1).astype(
        np.float32, copy=True
    )
    mask = np.stack([source_mask[:, :, index] for index in indices], axis=-1).astype(
        bool, copy=True
    )
    age_minutes = np.stack(
        [source_age_minutes[:, :, index] for index in indices], axis=-1
    ).astype(np.float32, copy=True)
    if not np.isfinite(values[mask]).all():
        raise ValueError("DPC valid values contain NaN or infinity")
    if ((age_minutes[mask] < 0) | (age_minutes[mask] > 60)).any():
        raise ValueError("DPC valid observation age must be within 0-60 minutes")

    stats = normalization.get("variables")
    if not isinstance(stats, dict):
        raise ValueError("Normalization file has no variable statistics")
    for channel, name in enumerate(V7_INPUT_VARIABLES):
        item = stats.get(name)
        if not isinstance(item, dict):
            raise ValueError(f"Normalization statistics are missing {name}")
        mean, std = float(item["mean"]), float(item["std"])
        if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
            raise ValueError(f"Invalid normalization statistics for {name}")
        values[:, :, channel] = (values[:, :, channel] - mean) / std

    values[~mask] = 0.0
    age = np.zeros(age_minutes.shape, dtype=np.float32)
    age[mask] = age_minutes[mask] / 60.0
    station_present = mask.any(axis=-1)
    normalized_coordinates = _normalize_coordinates(
        coordinates,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
    )
    if not np.isfinite(values).all() or not np.isfinite(age).all():
        raise ValueError("V7 input contains non-finite values")
    return V7InputBatch(
        times=np.asarray(times).copy(),
        station_ids=ids,
        variable_names=V7_INPUT_VARIABLES,
        values=values,
        value_mask=mask,
        observation_age=age,
        station_present=station_present,
        coordinates=normalized_coordinates,
        station_static=np.asarray(station_static, dtype=np.float32).copy(),
        source_type=tuple("physical" for _ in ids),
    )


def load_dpc_v7_input(
    tensor_path: str | Path,
    normalization_path: str | Path,
    *,
    expected_station_ids: tuple[str, ...] | None = None,
) -> V7InputBatch:
    """Load one saved official tensor and convert it to the V7 contract."""
    normalization = json.loads(Path(normalization_path).read_text(encoding="utf-8"))
    with np.load(tensor_path, allow_pickle=False) as source:
        return adapt_dpc_to_v7(
            times=source["times"],
            station_ids=source["station_ids"],
            source_variable_names=source["variable_names"],
            source_values=source["values"],
            source_mask=source["value_mask"],
            source_age_minutes=source["observation_age_minutes"],
            coordinates=source["coordinates"],
            station_static=source["station_static"],
            normalization=normalization,
            expected_station_ids=expected_station_ids,
        )
