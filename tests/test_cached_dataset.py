import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from stormengine_dl.data import CachedEra5SequenceDataset, CachedReconstructionDataset


class CachedDatasetTest(unittest.TestCase):
    def test_year_split_window_and_station_dropout_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            times = np.arange(
                np.datetime64("2010-12-31T22"),
                np.datetime64("2011-01-01T06"),
                np.timedelta64(1, "h"),
            ).astype("datetime64[ns]")
            point_values = np.arange(8 * 2, dtype=np.float32).reshape(8, 2, 1)
            targets = np.arange(8 * 1 * 2 * 2, dtype=np.float32).reshape(8, 1, 2, 2)
            np.save(root / "times.npy", times)
            np.save(root / "point_values.npy", point_values)
            np.save(root / "target_grids.npy", targets)
            np.save(root / "point_coords.npy", np.asarray([[0.0, 0.0], [1.0, 1.0]], np.float32))
            np.save(root / "point_static.npy", np.asarray([[1.0, 0.0], [0.0, 1.0]], np.float32))
            (root / "metadata.json").write_text(
                json.dumps(
                    {
                        "format_version": 1,
                        "input_variables": ["u10"],
                        "target_variables": ["u10"],
                    }
                ),
                encoding="utf-8",
            )
            dataset = CachedEra5SequenceDataset(
                root,
                years=[2011],
                history_hours=2,
                forecast_hours=1,
                window_stride_hours=2,
                input_variables=["u10"],
                target_variables=["u10"],
            )
            self.assertEqual(len(dataset), 2)
            sample = dataset[1]
            self.assertEqual(tuple(sample["point_values"].shape), (2, 2, 1))
            self.assertEqual(tuple(sample["target"].shape), (1, 1, 2, 2))
            self.assertEqual(tuple(sample["point_static"].shape), (2, 2))
            self.assertEqual(sample["start_index"].item(), 4)
            dataset.close()

    def test_reconstruction_view_uses_last_history_hour_as_simultaneous_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            times = np.arange(
                np.datetime64("2011-01-01T00"),
                np.datetime64("2011-01-01T06"),
                np.timedelta64(1, "h"),
            ).astype("datetime64[ns]")
            np.save(root / "times.npy", times)
            np.save(root / "point_values.npy", np.arange(12, dtype=np.float32).reshape(6, 2, 1))
            targets = np.arange(24, dtype=np.float32).reshape(6, 1, 2, 2)
            np.save(root / "target_grids.npy", targets)
            np.save(root / "point_coords.npy", np.zeros((2, 2), np.float32))
            np.save(root / "point_static.npy", np.zeros((2, 2), np.float32))
            (root / "metadata.json").write_text(
                json.dumps(
                    {"format_version": 1, "input_variables": ["u10"], "target_variables": ["u10"]}
                ),
                encoding="utf-8",
            )
            source = CachedEra5SequenceDataset(
                root,
                years=[2011],
                history_hours=2,
                forecast_hours=1,
                input_variables=["u10"],
                target_variables=["u10"],
            )
            dataset = CachedReconstructionDataset(source, ["u10"])
            sample = dataset[0]
            self.assertTrue(torch.equal(sample["point_values"], torch.tensor([[[2.0], [3.0]]])))
            self.assertTrue(torch.equal(sample["target"], torch.from_numpy(targets[1:2])))
            dataset.close()

    def test_variable_order_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "times.npy", np.asarray([np.datetime64("2010-01-01")]))
            np.save(root / "point_values.npy", np.zeros((1, 1, 1), np.float32))
            np.save(root / "target_grids.npy", np.zeros((1, 1, 1, 1), np.float32))
            np.save(root / "point_coords.npy", np.zeros((1, 2), np.float32))
            np.save(root / "point_static.npy", np.zeros((1, 2), np.float32))
            (root / "metadata.json").write_text(
                json.dumps(
                    {
                        "format_version": 1,
                        "input_variables": ["u10"],
                        "target_variables": ["u10"],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "input variable order"):
                CachedEra5SequenceDataset(
                    root,
                    years=[2010],
                    input_variables=["v10"],
                    target_variables=["u10"],
                )


if __name__ == "__main__":
    unittest.main()
