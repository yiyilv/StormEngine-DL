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


DPC_VARIABLE_NAMES = {
    "u10": "u10",
    "v10": "v10",
    "i10fg": "wind_gust_max",
    "t2m": "t2m",
    "tp": "tp",
}


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
    variable_dropout_range: tuple[float, float] | None = None
    station_dropout_range: tuple[float, float] | None = None
    outage_duration_hours: tuple[int, int] | None = None

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
        for bounds in (self.variable_dropout_range, self.station_dropout_range):
            if bounds is not None and not (0.0 <= bounds[0] <= bounds[1] < 1.0):
                raise ValueError("Missingness probability ranges must lie in [0, 1)")
        if self.outage_duration_hours is not None and not (
            1 <= self.outage_duration_hours[0] <= self.outage_duration_hours[1]
        ):
            raise ValueError("Outage duration range must be positive and ordered")
        if self.time_block_hours < 1:
            raise ValueError("time_block_hours must be positive")


class EmpiricalDPCMaskLibrary:
    """Contiguous DPC mask/age windows retained for realistic diagnostics."""

    def __init__(
        self,
        tensor_path: str | Path,
        *,
        station_ids: Sequence[str],
        variables: Sequence[str],
        history_hours: int,
        manifest_path: str | Path | None = None,
    ) -> None:
        tensor = Path(tensor_path)
        manifest = None
        if manifest_path is not None:
            manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
            if tensor.stat().st_size != int(manifest["tensor_bytes"]):
                raise ValueError("Empirical DPC tensor size does not match its manifest")
            if sha256_file(tensor) != manifest["tensor_sha256"]:
                raise ValueError("Empirical DPC tensor SHA-256 does not match its manifest")
        with np.load(tensor, allow_pickle=False) as source:
            source_ids = tuple(str(value) for value in source["station_ids"].tolist())
            if source_ids != tuple(station_ids):
                raise ValueError("Empirical DPC station order does not match the V7 cache view")
            source_names = tuple(str(value) for value in source["variable_names"].tolist())
            mapped_names = tuple(DPC_VARIABLE_NAMES.get(name, name) for name in variables)
            missing = set(mapped_names) - set(source_names)
            if missing:
                raise ValueError(f"Empirical DPC tensor is missing variables: {sorted(missing)}")
            indices = [source_names.index(name) for name in mapped_names]
            self.times = np.asarray(source["times"]).astype("datetime64[ns]")
            self.mask = np.stack(
                [source["value_mask"][:, :, channel] for channel in indices], axis=-1
            ).astype(bool)
            age_minutes = np.stack(
                [source["observation_age_minutes"][:, :, channel] for channel in indices],
                axis=-1,
            ).astype(np.float32)
        if self.mask.shape[:2] != (len(self.times), len(station_ids)):
            raise ValueError("Empirical DPC tensor dimensions are inconsistent")
        if not np.isfinite(age_minutes[self.mask]).all():
            raise ValueError("Empirical DPC valid ages contain NaN or infinity")
        if ((age_minutes[self.mask] < 0) | (age_minutes[self.mask] > 60)).any():
            raise ValueError("Empirical DPC valid ages must be within 0-60 minutes")
        self.age = np.zeros(age_minutes.shape, dtype=np.float32)
        self.age[self.mask] = age_minutes[self.mask] / 60.0
        if "tp" in variables:
            tp_channel = tuple(variables).index("tp")
            if (age_minutes[:, :, tp_channel][self.mask[:, :, tp_channel]] != 0).any():
                raise ValueError("Empirical DPC TP must end at the current valid time")
        self.starts = np.asarray(
            [
                start
                for start in range(max(0, len(self.times) - history_hours + 1))
                if self.times[start + history_hours - 1] - self.times[start]
                == np.timedelta64(history_hours - 1, "h")
            ],
            dtype=np.int64,
        )
        self.history_hours = history_hours
        if self.starts.size == 0:
            raise ValueError("Empirical DPC tensor has no contiguous history window")
        if manifest is not None:
            checks = {
                "hour_count": len(self.times),
                "station_count": len(station_ids),
                "history_hours": history_hours,
                "contiguous_template_count": int(self.starts.size),
            }
            differing = [key for key, value in checks.items() if int(manifest[key]) != value]
            if differing:
                raise ValueError(f"Empirical DPC manifest metadata mismatch: {differing}")

    def sample(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, int]:
        template = int(rng.integers(self.starts.size))
        start = int(self.starts[template])
        stop = start + self.history_hours
        return self.mask[start:stop].copy(), self.age[start:stop].copy(), start


class V7CachedSequenceDataset(Dataset[dict[str, torch.Tensor]]):
    """View the V6 memmaps through a configurable mask-aware station profile."""

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
        empirical_mask_path: str | Path | None = None,
        empirical_mask_manifest_path: str | Path | None = None,
        station_profile: str = "physical_only",
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
        if station_profile == "physical_only":
            selected_indices = identity["physical_station_indices"]
        elif station_profile == "dpc_plus_sea":
            selected_indices = list(range(len(identity["station_ids"])))
        else:
            raise ValueError(f"Unknown V7 station profile: {station_profile}")
        self.station_profile = station_profile
        self.station_indices = np.asarray(selected_indices, dtype=np.int64)
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
        self.empirical_masks = (
            EmpiricalDPCMaskLibrary(
                empirical_mask_path,
                station_ids=self.station_ids,
                variables=self.input_variables,
                history_hours=self.history_hours,
                manifest_path=empirical_mask_manifest_path,
            )
            if empirical_mask_path is not None
            else None
        )

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

        if self.empirical_masks is not None:
            empirical_mask, empirical_age, _ = self.empirical_masks.sample(rng)
            stale = empirical_mask & (empirical_age > 0)
            if stale.any():
                previous_indices = np.arange(global_start, history_stop, dtype=np.int64) - 1
                valid_previous = previous_indices >= 0
                previous = self._read_input_values(previous_indices.clip(0))
                interpolated = current * (1.0 - empirical_age) + previous * empirical_age
                for channel, name in enumerate(self.input_variables):
                    selected = stale[:, :, channel]
                    if name == "i10fg":
                        current[:, :, channel][selected] = previous[:, :, channel][selected]
                    elif name != "tp":
                        current[:, :, channel][selected] = interpolated[:, :, channel][selected]
                if not valid_previous.all():
                    empirical_mask[~valid_previous, :, :] &= ~stale[~valid_previous, :, :]
            mask &= empirical_mask
            age[mask] = empirical_age[mask]

        for channel, name in enumerate(self.input_variables):
            probability = float(self.strategy.variable_dropout.get(name, 0.0))
            if self.strategy.variable_dropout_range is not None:
                probability = float(rng.uniform(*self.strategy.variable_dropout_range))
            if probability:
                mask[:, :, channel] &= rng.random(mask[:, :, channel].shape) >= probability
        station_dropout = self.strategy.station_dropout
        if self.strategy.station_dropout_range is not None:
            station_dropout = float(rng.uniform(*self.strategy.station_dropout_range))
        if station_dropout:
            keep = rng.random(self.station_indices.size) >= station_dropout
            mask &= keep[None, :, None]
        if self.strategy.network_dropout:
            for indices in self._network_groups.values():
                if rng.random() < self.strategy.network_dropout:
                    minimum, maximum = self.strategy.outage_duration_hours or (
                        self.history_hours, self.history_hours
                    )
                    length = min(self.history_hours, int(rng.integers(minimum, maximum + 1)))
                    start = int(rng.integers(self.history_hours - length + 1))
                    mask[start:start + length, indices, :] = False
        if self.strategy.time_block_probability and rng.random() < self.strategy.time_block_probability:
            network = sorted(self._network_groups)[int(rng.integers(len(self._network_groups)))]
            indices = self._network_groups[network]
            length = min(self.history_hours, self.strategy.time_block_hours)
            start = int(rng.integers(self.history_hours - length + 1))
            mask[start:start + length, indices, :] = False

        shift = rng.random(current.shape) < self.strategy.age_60_probability
        shift &= mask
        shift &= age == 0
        if "tp" in self.input_variables:
            shift[:, :, self.input_variables.index("tp")] = False
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
            "station_present": torch.from_numpy(mask.any(axis=-1)),
            "source_type": torch.tensor(
                [0 if self.station_ids[index].startswith("LAND::") else 1 for index in range(len(self.station_ids))],
                dtype=torch.long,
            ),
            "point_coords": self._coords,
            "point_static": self._static,
            "target": torch.from_numpy(target),
            "start_index": torch.tensor(global_start, dtype=torch.long),
        }
