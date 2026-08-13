"""Combined physical-DPC and model-derived marine input contract for V7-B."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .v7_input import V7_INPUT_VARIABLES, V7InputBatch, load_dpc_v7_input


@dataclass(frozen=True)
class V7BInputBatch(V7InputBatch):
    physical_station_count: int
    marine_station_count: int


def _normalise_marine(
    values: np.ndarray, mask: np.ndarray, variable_names: tuple[str, ...], normalization: dict[str, object]
) -> np.ndarray:
    if variable_names != V7_INPUT_VARIABLES:
        raise ValueError(f"Open-Meteo variable order must be {V7_INPUT_VARIABLES}")
    result = np.asarray(values, np.float32).copy()
    stats = normalization["variables"]
    for channel, name in enumerate(variable_names):
        item = stats[name]
        result[:, :, channel] = (result[:, :, channel] - float(item["mean"])) / float(item["std"])
    result[~mask] = 0
    return result


def load_v7_b_input(
    dpc_path: str | Path,
    marine_path: str | Path,
    normalization_path: str | Path,
    *,
    expected_station_ids: tuple[str, ...] | None = None,
    lat_min: float = 39.0,
    lat_max: float = 46.5,
    lon_min: float = 12.0,
    lon_max: float = 20.0,
) -> V7BInputBatch:
    """Return the common 169-hour, 390-station V7-B replay contract."""
    physical = load_dpc_v7_input(dpc_path, normalization_path)
    normalization = json.loads(Path(normalization_path).read_text(encoding="utf-8"))
    with np.load(marine_path, allow_pickle=False) as marine:
        marine_times = np.asarray(marine["times"]).astype("datetime64[ns]")
        marine_ids = tuple(str(value) for value in marine["station_ids"].tolist())
        marine_names = tuple(str(value) for value in marine["variable_names"].tolist())
        marine_mask = np.asarray(marine["value_mask"], bool)
        marine_values = _normalise_marine(marine["values"], marine_mask, marine_names, normalization)
        marine_age = np.asarray(marine["observation_age"], np.float32)
        marine_coordinates = np.asarray(marine["coordinates"], np.float32).copy()
    if not np.array_equal(physical.times.astype("datetime64[ns]"), marine_times):
        raise ValueError("DPC and Open-Meteo time axes are not identical")
    marine_coordinates[:, 0] = (marine_coordinates[:, 0] - lat_min) / (lat_max - lat_min)
    marine_coordinates[:, 1] = (marine_coordinates[:, 1] - lon_min) / (lon_max - lon_min)
    if (marine_coordinates < 0).any() or (marine_coordinates > 1).any():
        raise ValueError("Open-Meteo coordinates fall outside the V7 domain")
    station_ids = physical.station_ids + marine_ids
    if expected_station_ids is not None and station_ids != expected_station_ids:
        raise ValueError("Combined V7-B station order does not match the fixed 390-point registry")
    values = np.concatenate((physical.values, marine_values), axis=1)
    mask = np.concatenate((physical.value_mask, marine_mask), axis=1)
    age = np.concatenate((physical.observation_age, marine_age), axis=1)
    coordinates = np.concatenate((physical.coordinates, marine_coordinates), axis=0)
    marine_static = np.tile(np.asarray([[0.0, 1.0]], np.float32), (len(marine_ids), 1))
    station_static = np.concatenate((physical.station_static, marine_static), axis=0)
    return V7BInputBatch(
        times=physical.times,
        station_ids=station_ids,
        variable_names=V7_INPUT_VARIABLES,
        values=values,
        value_mask=mask,
        observation_age=age,
        station_present=mask.any(axis=-1),
        coordinates=coordinates,
        station_static=station_static,
        source_type=physical.source_type + tuple("model_derived_open_meteo" for _ in marine_ids),
        physical_station_count=len(physical.station_ids),
        marine_station_count=len(marine_ids),
    )
