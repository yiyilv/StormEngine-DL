"""Current-time sparse-to-grid views of the hourly training cache."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .cached_dataset import CachedEra5SequenceDataset
from .v7_dataset import V7CachedSequenceDataset


class CachedReconstructionDataset(Dataset[dict[str, torch.Tensor]]):
    """Use the final history snapshot to reconstruct its simultaneous dense grid."""

    def __init__(
        self,
        source: CachedEra5SequenceDataset,
        target_variables: Sequence[str],
    ) -> None:
        self.source = source
        self.target_variables = tuple(target_variables)
        missing = set(self.target_variables) - set(source.target_variables)
        if missing:
            raise ValueError(f"reconstruction targets missing from cache: {sorted(missing)}")
        self.target_indices = [source.target_variables.index(name) for name in self.target_variables]

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        sample = self.source[item]
        current_index = int(sample["start_index"]) + self.source.history_hours - 1
        target = np.asarray(
            self.source.target_grids[current_index, self.target_indices], dtype=np.float32
        ).copy()
        return {
            "point_values": sample["point_values"][-1:],
            "point_coords": sample["point_coords"],
            "point_mask": sample["point_mask"][-1:],
            "point_static": sample["point_static"],
            "target": torch.from_numpy(target[None]),
            "start_index": sample["start_index"],
        }

    def close(self) -> None:
        self.source.close()


class V7CachedReconstructionDataset(Dataset[dict[str, torch.Tensor]]):
    """Mask-aware current-time reconstruction view of a V7 cache dataset.

    The final history snapshot is paired with the dense grid at that exact
    valid time. Future target hours are never used by this diagnostic.
    """

    def __init__(
        self,
        source: V7CachedSequenceDataset,
        target_variables: Sequence[str],
    ) -> None:
        self.source = source
        self.target_variables = tuple(target_variables)
        missing = set(self.target_variables) - set(source.target_variables)
        if missing:
            raise ValueError(f"reconstruction targets missing from cache: {sorted(missing)}")
        self.target_indices = [source.target_variables.index(name) for name in self.target_variables]

    def __len__(self) -> int:
        return len(self.source)

    def set_epoch(self, epoch: int) -> None:
        self.source.set_epoch(epoch)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        sample = self.source[item]
        current_index = int(sample["start_index"]) + self.source.history_hours - 1
        target = np.asarray(
            self.source.target_grids[current_index, self.target_indices], dtype=np.float32
        ).copy()
        return {
            "point_values": sample["point_values"][-1:],
            "value_mask": sample["value_mask"][-1:],
            "observation_age": sample["observation_age"][-1:],
            "point_coords": sample["point_coords"],
            "point_static": sample["point_static"],
            "source_type": sample["source_type"],
            "target": torch.from_numpy(target[None]),
            "start_index": sample["start_index"],
        }

    def close(self) -> None:
        self.source.close()
