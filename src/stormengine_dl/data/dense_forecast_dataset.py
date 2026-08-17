"""Dense ERA5 history-to-future windows for Processor-only development."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .cached_dataset import CachedEra5SequenceDataset


class DenseGridForecastDataset(Dataset[dict[str, torch.Tensor]]):
    """Expose dense history and future grids without reading sparse point arrays.

    The wrapped cache already contains normalized target grids in chronological
    order.  Reusing those grids on both sides of the forecast contract isolates
    temporal Processor skill from the sparse Encoder/Decoder interface.
    """

    def __init__(self, source: CachedEra5SequenceDataset) -> None:
        self.source = source

    @property
    def variables(self) -> tuple[str, ...]:
        return self.source.target_variables

    def __len__(self) -> int:
        return len(self.source)

    def close(self) -> None:
        self.source.close()

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        local_start = int(self.source.window_starts[item])
        global_start = int(self.source.global_indices[local_start])
        history_stop = global_start + self.source.history_hours
        target_stop = history_stop + self.source.forecast_hours
        history = np.asarray(
            self.source.target_grids[global_start:history_stop], dtype=np.float32
        ).copy()
        target = np.asarray(
            self.source.target_grids[history_stop:target_stop], dtype=np.float32
        ).copy()
        return {
            "history": torch.from_numpy(history),
            "target": torch.from_numpy(target),
            "start_index": torch.tensor(global_start, dtype=torch.long),
        }
