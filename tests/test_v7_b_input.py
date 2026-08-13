import unittest
from pathlib import Path

import numpy as np

from stormengine_dl.data import load_v7_b_input


class V7BInputTests(unittest.TestCase):
    def test_real_local_contract_when_external_inputs_are_available(self) -> None:
        root = Path(__file__).resolve().parents[1]
        dpc = root / "data_external/meteohub/processed/20260801_20260808/official_hourly_physical.npz"
        marine = root / "data_external/open_meteo/processed/20260801_20260808/icon2i_hourly_marine.npz"
        if not dpc.exists() or not marine.exists():
            self.skipTest("external replay tensors are intentionally not stored in Git")
        batch = load_v7_b_input(dpc, marine, root / "data/normalization/era5_2010_2015.json")
        self.assertEqual(batch.values.shape, (169, 390, 5))
        self.assertEqual(batch.physical_station_count, 239)
        self.assertEqual(batch.marine_station_count, 151)
        self.assertEqual(batch.source_type.count("model_derived_open_meteo"), 151)
        self.assertTrue(np.isfinite(batch.values).all())
        self.assertTrue(batch.value_mask[:, 239:].all())
        np.testing.assert_array_equal(batch.station_present, batch.value_mask.any(axis=-1))


if __name__ == "__main__":
    unittest.main()
