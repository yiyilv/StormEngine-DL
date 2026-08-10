import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from stormengine_dl.data.v7_dataset import (
    EmpiricalDPCMaskLibrary,
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

    def test_target_only_tp_is_bilinearly_sampled_as_an_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache, registry, identity = self._fixture(Path(directory))
            metadata_path = cache / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["target_variables"] = ["u10", "tp"]
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            targets = np.zeros((10, 2, 2, 2), dtype=np.float32)
            for time in range(10):
                targets[time, 1] = np.asarray(
                    [[time, time + 2], [time + 4, time + 6]], dtype=np.float32
                )
            np.save(cache / "target_grids.npy", targets)
            identity.write_text(
                json.dumps(build_cache_identity(cache, registry)), encoding="utf-8"
            )
            dataset = V7CachedSequenceDataset(
                cache, registry, identity,
                years=[2010], input_variables=["tp"], target_variables=["u10", "tp"],
                strategy=MissingnessStrategy({}), history_hours=2, forecast_hours=1,
            )
            sample = dataset[0]
            self.assertEqual(tuple(sample["point_values"].shape), (2, 2, 1))
            self.assertTrue(torch.allclose(
                sample["point_values"][:, :, 0], torch.tensor([[0., 3.], [1., 4.]])
            ))
            dataset.close()

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

    def test_empirical_dpc_mask_and_fractional_age_are_replayed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache, registry, identity = self._fixture(Path(directory))
            empirical = Path(directory) / "official.npz"
            mask = np.ones((2, 2, 2), dtype=bool)
            mask[1, 1, 1] = False
            ages = np.zeros((2, 2, 2), dtype=np.float32)
            ages[0, 0, 0] = 30.0
            ages[~mask] = -1.0
            np.savez_compressed(
                empirical,
                times=np.arange(
                    np.datetime64("2026-08-01T00"),
                    np.datetime64("2026-08-01T02"),
                    np.timedelta64(1, "h"),
                ).astype("datetime64[ns]"),
                station_ids=np.asarray(["LAND::a", "LAND::b"]),
                variable_names=np.asarray(["u10", "t2m"]),
                value_mask=mask,
                observation_age_minutes=ages,
            )
            dataset = V7CachedSequenceDataset(
                cache, registry, identity,
                years=[2010], input_variables=["u10", "t2m"], target_variables=["u10"],
                strategy=MissingnessStrategy({}), history_hours=2, forecast_hours=1,
                empirical_mask_path=empirical,
            )
            sample = dataset[1]
            self.assertAlmostEqual(float(sample["point_values"][0, 0, 0]), 3.0)
            self.assertAlmostEqual(float(sample["observation_age"][0, 0, 0]), 0.5)
            self.assertFalse(bool(sample["value_mask"][1, 1, 1]))
            self.assertEqual(float(sample["point_values"][1, 1, 1]), 0.0)
            dataset.close()

    def test_empirical_stale_tp_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tensor = Path(directory) / "stale_tp.npz"
            np.savez_compressed(
                tensor,
                times=np.arange(
                    np.datetime64("2026-08-01T00"),
                    np.datetime64("2026-08-01T02"),
                    np.timedelta64(1, "h"),
                ).astype("datetime64[ns]"),
                station_ids=np.asarray(["LAND::a"]),
                variable_names=np.asarray(["tp"]),
                value_mask=np.ones((2, 1, 1), dtype=bool),
                observation_age_minutes=np.asarray([[[0.0]], [[10.0]]], dtype=np.float32),
            )
            with self.assertRaisesRegex(ValueError, "TP must end"):
                EmpiricalDPCMaskLibrary(
                    tensor, station_ids=["LAND::a"], variables=["tp"], history_hours=2
                )


if __name__ == "__main__":
    unittest.main()
