import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from stormengine_dl.data.v7_dataset import (
    MissingnessStrategy,
    V7CachedSequenceDataset,
    build_cache_identity,
)


class V7DatasetTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, Path]:
        cache = root / "cache"
        cache.mkdir()
        times = np.arange(
            np.datetime64("2010-01-01T00"),
            np.datetime64("2010-01-01T10"),
            np.timedelta64(1, "h"),
        ).astype("datetime64[ns]")
        points = np.arange(10 * 3 * 2, dtype=np.float32).reshape(10, 3, 2)
        np.save(cache / "times.npy", times)
        np.save(cache / "point_values.npy", points)
        np.save(cache / "target_grids.npy", np.zeros((10, 1, 2, 2), np.float32))
        np.save(cache / "point_coords.npy", np.asarray([[0, 0], [.5, .5], [1, 1]], np.float32))
        np.save(cache / "point_static.npy", np.asarray([[1, 0], [1, 0], [0, 1]], np.float32))
        (cache / "metadata.json").write_text(json.dumps({
            "format_version": 1,
            "station_profile": "dpc_plus_sea",
            "input_variables": ["u10", "t2m"],
            "target_variables": ["u10"],
        }), encoding="utf-8")
        registry = root / "registry.csv"
        fields = ["station_id", "station_name", "latitude", "longitude", "station_type",
                  "network", "enabled", "profile_dpc_plus_sea"]
        rows = [
            ["LAND::a", "A", 40, 13, "physical_land", "one", True, True],
            ["LAND::b", "B", 41, 14, "physical_land", "two", True, True],
            ["SEA::c", "C", 42, 15, "virtual_sea", "sea", True, True],
        ]
        with registry.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(fields)
            writer.writerows(rows)
        identity_path = root / "identity.json"
        identity_path.write_text(
            json.dumps(build_cache_identity(cache, registry)), encoding="utf-8"
        )
        return cache, registry, identity_path

    def test_physical_variable_view_reuses_cache_without_copying_base(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache, registry, identity = self._fixture(Path(directory))
            dataset = V7CachedSequenceDataset(
                cache, registry, identity,
                years=[2010], input_variables=["t2m"], target_variables=["u10"],
                strategy=MissingnessStrategy({}), history_hours=2, forecast_hours=1,
            )
            sample = dataset[0]
            self.assertEqual(tuple(sample["point_values"].shape), (2, 2, 1))
            self.assertTrue(torch.equal(sample["point_values"][:, :, 0], torch.tensor([[1., 3.], [7., 9.]])))
            self.assertEqual(tuple(sample["value_mask"].shape), (2, 2, 1))
            self.assertTrue(sample["value_mask"].all())
            self.assertEqual(dataset.station_ids, ("LAND::a", "LAND::b"))
            dataset.close()

    def test_identity_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache, registry, identity = self._fixture(Path(directory))
            registry.write_text(registry.read_text() + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                V7CachedSequenceDataset(
                    cache, registry, identity,
                    years=[2010], input_variables=["u10"], target_variables=["u10"],
                    strategy=MissingnessStrategy({}), history_hours=2, forecast_hours=1,
                )

    def test_masks_zero_missing_values_and_are_epoch_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache, registry, identity = self._fixture(Path(directory))
            dataset = V7CachedSequenceDataset(
                cache, registry, identity,
                years=[2010], input_variables=["u10", "t2m"], target_variables=["u10"],
                strategy=MissingnessStrategy({"u10": 0.5}, station_dropout=0.2,
                                             age_60_probability=0.4),
                history_hours=3, forecast_hours=1, seed=7,
            )
            first = dataset[1]
            repeated = dataset[1]
            self.assertTrue(torch.equal(first["value_mask"], repeated["value_mask"]))
            self.assertTrue((first["point_values"][~first["value_mask"]] == 0).all())
            self.assertTrue((first["observation_age"] <= 1).all())
            dataset.set_epoch(1)
            changed = dataset[1]
            self.assertFalse(torch.equal(first["value_mask"], changed["value_mask"]))
            dataset.close()


if __name__ == "__main__":
    unittest.main()
