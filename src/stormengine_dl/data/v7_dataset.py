"""V7 zero-copy cache view with validated identity and mask-aware augmentation."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


def sample_normalized_grid_at_points(
    grids: np.ndarray, normalized_coordinates: np.ndarray
) -> np.ndarray:
    """Bilinearly sample ``[T,H,W]`` grids at normalized ``[N,2]`` points."""
    if grids.ndim != 3 or normalized_coordinates.ndim != 2 or normalized_coordinates.shape[1] != 2:
        raise ValueError("Expected grids [T,H,W] and normalized coordinates [N,2]")
    _, height, width = grids.shape
    latitude = np.clip(normalized_coordinates[:, 0], 0.0, 1.0) * (height - 1)
    longitude = np.clip(normalized_coordinates[:, 1], 0.0, 1.0) * (width - 1)
    lat_low, lon_low = np.floor(latitude).astype(np.int64), np.floor(longitude).astype(np.int64)
    lat_high, lon_high = np.ceil(latitude).astype(np.int64), np.ceil(longitude).astype(np.int64)
    lat_weight = (latitude - lat_low).astype(np.float32)[None]
    lon_weight = (longitude - lon_low).astype(np.float32)[None]
    low = grids[:, lat_low, lon_low] * (1.0 - lon_weight) + grids[:, lat_low, lon_high] * lon_weight
    high = grids[:, lat_high, lon_low] * (1.0 - lon_weight) + grids[:, lat_high, lon_high] * lon_weight
    return (low * (1.0 - lat_weight) + high * lat_weight).astype(np.float32)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def _registry_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return [
            row for row in csv.DictReader(handle)
            if _truthy(row["enabled"]) and _truthy(row["profile_dpc_plus_sea"])
        ]


def build_cache_identity(
    cache_dir: str | Path,
    registry_path: str | Path,
) -> dict[str, object]:
    cache = Path(cache_dir)
    registry = Path(registry_path)
    rows = _registry_rows(registry)
    coords = np.load(cache / "point_coords.npy", mmap_mode="r")
    static = np.load(cache / "point_static.npy", mmap_mode="r")
    metadata = json.loads((cache / "metadata.json").read_text(encoding="utf-8"))
    if len(rows) != coords.shape[0] or static.shape[0] != coords.shape[0]:
        raise ValueError("Cache station count does not match the current registry")
    physical_indices = [
        index for index, row in enumerate(rows) if row["station_type"] == "physical_land"
    ]
    return {
        "format_version": 1,
        "cache_format_version": metadata["format_version"],
        "station_profile": metadata["station_profile"],
        "station_ids": [row["station_id"] for row in rows],
        "station_networks": [row["network"] for row in rows],
        "physical_station_indices": physical_indices,
        "input_variables": metadata["input_variables"],
        "target_variables": metadata["target_variables"],
        "registry_sha256": sha256_file(registry),
        "cache_metadata_sha256": sha256_file(cache / "metadata.json"),
        "point_coords_sha256": sha256_file(cache / "point_coords.npy"),
        "point_static_sha256": sha256_file(cache / "point_static.npy"),
    }


def validate_cache_identity(
    cache_dir: str | Path,
    registry_path: str | Path,
    identity_path: str | Path,
) -> dict[str, object]:
    expected = json.loads(Path(identity_path).read_text(encoding="utf-8"))
    actual = build_cache_identity(cache_dir, registry_path)
    if actual != expected:
        differing = sorted(
            key for key in set(actual) | set(expected) if actual.get(key) != expected.get(key)
        )
        raise ValueError(f"V7 cache identity mismatch: {differing}")
    return actual


@dataclass(frozen=True)
class MissingnessStrategy:
    variable_dropout: Mapping[str, float]
    station_dropout: float = 0.0
    network_dropout: float = 0.0
    time_block_probability: float = 0.0
    time_block_hours: int = 3
    age_60_probability: float = 0.0

    def validate(self, variables: Sequence[str]) -> None:
        probabilities = [
            self.station_dropout,
            self.network_dropout,
            self.time_block_probability,
            self.age_60_probability,
            *(self.variable_dropout.get(name, 0.0) for name in variables),
        ]
        if any(not 0.0 <= value < 1.0 for value in probabilities):
            raise ValueError("Missingness probabilities must be in [0, 1)")
        if self.time_block_hours < 1:
            raise ValueError("time_block_hours must be positive")


class V7CachedSequenceDataset(Dataset[dict[str, torch.Tensor]]):
    """View the V6 memmaps as physical-only, mask-aware V7 samples."""

    def __init__(
        self,
        cache_dir: str | Path,
        registry_path: str | Path,
        identity_path: str | Path,
        *,
        years: Sequence[int],
        input_variables: Sequence[str],
        target_variables: Sequence[str],
        strategy: MissingnessStrategy,
        history_hours: int = 12,
        forecast_hours: int = 6,
        window_stride_hours: int = 1,
        seed: int = 42,
    ) -> None:
        if history_hours < 1 or forecast_hours < 1 or window_stride_hours < 1:
            raise ValueError("history, forecast, and stride must be positive")
        self.cache_dir = Path(cache_dir).resolve()
        identity = validate_cache_identity(self.cache_dir, registry_path, identity_path)
        self.metadata = json.loads(
            (self.cache_dir / "metadata.json").read_text(encoding="utf-8")
        )
        cached_inputs = list(self.metadata["input_variables"])
        cached_targets = list(self.metadata["target_variables"])
        available_inputs = set(cached_inputs) | set(cached_targets)
        missing_inputs = set(input_variables) - available_inputs
        if missing_inputs:
            raise ValueError(f"V7 input variables absent from cache: {sorted(missing_inputs)}")
        if list(target_variables) != cached_targets:
            raise ValueError("V7 target order must match the immutable base cache")
        strategy.validate(input_variables)
        self.input_variables = tuple(input_variables)
        self.target_variables = tuple(target_variables)
        self.strategy = strategy
        self.history_hours = history_hours
        self.forecast_hours = forecast_hours
        self.seed = int(seed)
        self.epoch = 0
        self.station_indices = np.asarray(identity["physical_station_indices"], dtype=np.int64)
        self.station_ids = tuple(
            identity["station_ids"][index] for index in self.station_indices  # type: ignore[index]
        )
        self.station_networks = tuple(
            identity["station_networks"][index] for index in self.station_indices  # type: ignore[index]
        )
        self._input_sources = tuple(
            ("point", cached_inputs.index(name))
            if name in cached_inputs else ("grid", cached_targets.index(name))
            for name in self.input_variables
        )
        self.point_values = np.load(self.cache_dir / "point_values.npy", mmap_mode="r")
        self.target_grids = np.load(self.cache_dir / "target_grids.npy", mmap_mode="r")
        self.all_times = np.asarray(
            np.load(self.cache_dir / "times.npy", mmap_mode="r")
        ).astype("datetime64[ns]")
        coords = np.load(self.cache_dir / "point_coords.npy", mmap_mode="r")
        static = np.load(self.cache_dir / "point_static.npy", mmap_mode="r")
        self._coords = torch.from_numpy(
            np.asarray(coords[self.station_indices], dtype=np.float32).copy()
        )
        self._static = torch.from_numpy(
            np.asarray(static[self.station_indices], dtype=np.float32).copy()
        )
        requested_years = np.asarray(sorted(set(int(year) for year in years)), dtype=np.int64)
        all_years = self.all_times.astype("datetime64[Y]").astype(np.int64) + 1970
        self.global_indices = np.flatnonzero(np.isin(all_years, requested_years)).astype(np.int64)
        if self.global_indices.size == 0:
            raise ValueError("Cache contains no timestamps for requested V7 years")
        split_times = self.all_times[self.global_indices]
        window = history_hours + forecast_hours
        self.window_starts = np.asarray(
            [
                start
                for start in range(0, max(0, split_times.size - window + 1), window_stride_hours)
                if split_times[start + window - 1] - split_times[start]
                == np.timedelta64(window - 1, "h")
            ],
            dtype=np.int64,
        )
        self._network_groups = {
            network: np.asarray(
                [index for index, value in enumerate(self.station_networks) if value == network],
                dtype=np.int64,
            )
            for network in sorted(set(self.station_networks))
        }

    def _read_input_values(self, indices: slice | np.ndarray) -> np.ndarray:
        point_block: np.ndarray | None = None
        grid_block: np.ndarray | None = None
        channels: list[np.ndarray] = []
        for source, channel in self._input_sources:
            if source == "point":
                if point_block is None:
                    point_block = np.asarray(self.point_values[indices], dtype=np.float32)[
                        :, self.station_indices
                    ]
                channels.append(point_block[:, :, channel])
            else:
                if grid_block is None:
                    grid_block = np.asarray(self.target_grids[indices], dtype=np.float32)
                channels.append(
                    sample_normalized_grid_at_points(grid_block[:, channel], self._coords.numpy())
                )
        return np.stack(channels, axis=-1).astype(np.float32, copy=False)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return int(self.window_starts.size)

    def close(self) -> None:
        for array in (self.point_values, self.target_grids):
            memory_map = getattr(array, "_mmap", None)
            if memory_map is not None:
                memory_map.close()

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        split_start = int(self.window_starts[item])
        global_start = int(self.global_indices[split_start])
        history_stop = global_start + self.history_hours
        target_stop = history_stop + self.forecast_hours
        current = self._read_input_values(slice(global_start, history_stop)).copy()
        mask = np.ones(current.shape, dtype=bool)
        age = np.zeros(current.shape, dtype=np.float32)
        rng = np.random.default_rng(
            self.seed + self.epoch * 1_000_003 + global_start * 97
        )

        for channel, name in enumerate(self.input_variables):
            probability = float(self.strategy.variable_dropout.get(name, 0.0))
            if probability:
                mask[:, :, channel] &= rng.random(mask[:, :, channel].shape) >= probability
        if self.strategy.station_dropout:
            keep = rng.random(self.station_indices.size) >= self.strategy.station_dropout
            mask &= keep[None, :, None]
        if self.strategy.network_dropout:
            for indices in self._network_groups.values():
                if rng.random() < self.strategy.network_dropout:
                    mask[:, indices, :] = False
        if self.strategy.time_block_probability and rng.random() < self.strategy.time_block_probability:
            network = sorted(self._network_groups)[int(rng.integers(len(self._network_groups)))]
            indices = self._network_groups[network]
            length = min(self.history_hours, self.strategy.time_block_hours)
            start = int(rng.integers(self.history_hours - length + 1))
            mask[start:start + length, indices, :] = False

        shift = rng.random(current.shape) < self.strategy.age_60_probability
        shift &= mask
        if shift.any():
            previous_indices = np.arange(global_start, history_stop, dtype=np.int64) - 1
            valid_previous = previous_indices >= 0
            previous_indices = previous_indices.clip(0)
            previous = self._read_input_values(previous_indices)
            current[shift] = previous[shift]
            age[shift] = 1.0
            if not valid_previous.all():
                mask[~valid_previous, :, :] &= ~shift[~valid_previous, :, :]
        current[~mask] = 0.0
        age[~mask] = 0.0
        target = np.asarray(
            self.target_grids[history_stop:target_stop], dtype=np.float32
        ).copy()
        return {
            "point_values": torch.from_numpy(current),
            "value_mask": torch.from_numpy(mask),
            "observation_age": torch.from_numpy(age),
            "point_mask": torch.from_numpy(mask.any(axis=-1).astype(np.float32)),
            "point_coords": self._coords,
            "point_static": self._static,
            "target": torch.from_numpy(target),
            "start_index": torch.tensor(global_start, dtype=torch.long),
        }
