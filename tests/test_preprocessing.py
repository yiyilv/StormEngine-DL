import tempfile
import unittest
from pathlib import Path

import numpy as np

from stormengine_dl.data import NormalizationStats, StaticFields, VariableStat, build_station_distance_field


class PreprocessingTest(unittest.TestCase):
    def test_normalization_round_trip(self) -> None:
        stats = NormalizationStats({"msl": VariableStat(1000.0, 10.0, 2)}, {"fit_years": [2010]})
        values = np.array([990.0, 1010.0], dtype=np.float32)
        normalized = stats.normalize("msl", values)
        self.assertTrue(np.allclose(normalized, [-1.0, 1.0]))
        self.assertTrue(np.allclose(stats.denormalize("msl", normalized), values))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stats.json"
            stats.save(path)
            loaded = NormalizationStats.load(path)
            self.assertEqual(loaded.variables["msl"], stats.variables["msl"])

    def test_distance_field_and_static_round_trip(self) -> None:
        latitudes = np.array([40.0, 41.0], dtype=np.float32)
        longitudes = np.array([15.0, 16.0], dtype=np.float32)
        distance = build_station_distance_field(latitudes, longitudes, np.array([[40.0, 15.0]]))
        self.assertEqual(distance.shape, (2, 2))
        self.assertEqual(float(distance[0, 0]), 0.0)
        self.assertAlmostEqual(float(distance.max()), 1.0)
        fields = StaticFields(latitudes, longitudes, np.zeros((2, 2), np.float32), distance)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "static.npz"
            fields.save(path)
            loaded = StaticFields.load(path)
            self.assertEqual(tuple(loaded.as_tensor().shape), (2, 2, 2))
            self.assertTrue(np.allclose(loaded.station_distance, distance))


if __name__ == "__main__":
    unittest.main()
