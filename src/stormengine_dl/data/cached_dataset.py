"""Memory-mapped hourly cache for fast StormEngine sequence training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


class CachedEra5SequenceDataset(Dataset[dict[str, torch.Tensor]]):
    """Build sequence windows from normalized, memory-mapped hourly arrays."""

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        years: Iterable[int],
        history_hours: int = 12,
        forecast_hours: int = 6,
        window_stride_hours: int = 1,
        station_dropout: float = 0.0,
        input_variables: Iterable[str] | None = None,
        target_variables: Iterable[str] | None = None,
    ) -> None:
        if history_hours < 1 or forecast_hours < 1 or window_stride_hours < 1:
            raise ValueError("history, forecast, and stride must be positive")
        if not 0.0 <= station_dropout < 1.0:
            raise ValueError("station_dropout must be in [0, 1)")

        self.cache_dir = Path(cache_dir).expanduser().resolve()
        metadata_path = self.cache_dir / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"training cache not found at {metadata_path}; run scripts/build_training_cache.py"
            )
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if int(self.metadata.get("format_version", 0)) != 1:
            raise ValueError("unsupported training cache format")
        if input_variables is not None and list(input_variables) != self.metadata["input_variables"]:
            raise ValueError("training cache input variable order does not match configuration")
        if target_variables is not None and list(target_variables) != self.metadata["target_variables"]:
            raise ValueError("training cache target variable order does not match configuration")

        self.point_values = np.load(self.cache_dir / "point_values.npy", mmap_mode="r")
        self.target_grids = np.load(self.cache_dir / "target_grids.npy", mmap_mode="r")
        raw_times = np.load(self.cache_dir / "times.npy", mmap_mode="r")
        self.all_times = np.asarray(raw_times).astype("datetime64[ns]")
        self.normalized_station_coordinates = np.load(
            self.cache_dir / "point_coords.npy", mmap_mode="r"
        )
        self.station_features = np.load(self.cache_dir / "point_static.npy", mmap_mode="r")

        if self.point_values.shape[0] != self.all_times.size:
            raise ValueError("point cache and time cache lengths differ")
        if self.target_grids.shape[0] != self.all_times.size:
            raise ValueError("target cache and time cache lengths differ")
        if self.point_values.shape[1] != self.normalized_station_coordinates.shape[0]:
            raise ValueError("point cache and coordinate cache station counts differ")

        requested_years = np.asarray(sorted(set(int(year) for year in years)), dtype=np.int64)
        all_years = self.all_times.astype("datetime64[Y]").astype(np.int64) + 1970
        self.global_indices = np.flatnonzero(np.isin(all_years, requested_years)).astype(np.int64)
        if self.global_indices.size == 0:
            raise ValueError("cache contains no timestamps for the requested years")
        self.times = self.all_times[self.global_indices]
        month_ids = self.times.astype("datetime64[M]").astype(np.int64)
        self.months = np.unique(month_ids).tolist()
        self.history_hours = history_hours
        self.forecast_hours = forecast_hours
        self.window_stride_hours = window_stride_hours
        self.station_dropout = station_dropout
        # These arrays are tiny and invariant across every sample. Keep one safe,
        # writable tensor copy instead of allocating them again for every window.
        self._point_coords_tensor = torch.from_numpy(
            np.asarray(self.normalized_station_coordinates, dtype=np.float32).copy()
        )
        self._point_static_tensor = torch.from_numpy(
            np.asarray(self.station_features, dtype=np.float32).copy()
        )
        window = history_hours + forecast_hours
        self.window_starts = np.asarray(
            [
                start
                for start in range(0, max(0, self.times.size - window + 1), window_stride_hours)
                if self.times[start + window - 1] - self.times[start]
                == np.timedelta64(window - 1, "h")
            ],
            dtype=np.int64,
        )

    @property
    def input_variables(self) -> tuple[str, ...]:
        return tuple(self.metadata["input_variables"])

    @property
    def target_variables(self) -> tuple[str, ...]:
        return tuple(self.metadata["target_variables"])

    def __len__(self) -> int:
        return int(self.window_starts.size)

    def close(self) -> None:
        """Release memory-map handles explicitly, which is required on Windows."""
        for name in (
            "point_values",
            "target_grids",
            "normalized_station_coordinates",
            "station_features",
        ):
            values = getattr(self, name, None)
            memory_map = getattr(values, "_mmap", None)
            if memory_map is not None:
                memory_map.close()

    def __enter__(self) -> "CachedEra5SequenceDataset":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        start = int(self.window_starts[item])
        global_start = int(self.global_indices[start])
        history_stop = global_start + self.history_hours
        target_stop = history_stop + self.forecast_hours
        # Valid windows are hourly-contiguous, so ordinary slices avoid NumPy's
        # slower advanced-indexing path on the memory-mapped files.
        point_values = np.asarray(
            self.point_values[global_start:history_stop], dtype=np.float32
        ).copy()
        target = np.asarray(self.target_grids[history_stop:target_stop], dtype=np.float32).copy()
        mask = torch.ones((self.history_hours, point_values.shape[1]))
        if self.station_dropout:
            station_mask = torch.rand(point_values.shape[1]) >= self.station_dropout
            mask *= station_mask[None]
            point_values *= mask.numpy()[:, :, None]
        return {
            "point_values": torch.from_numpy(point_values),
            "point_coords": self._point_coords_tensor,
            "point_mask": mask,
            "point_static": self._point_static_tensor,
            "target": torch.from_numpy(target),
            "start_index": torch.tensor(global_start, dtype=torch.long),
        }
