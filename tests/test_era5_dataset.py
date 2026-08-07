import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import xarray as xr

from stormengine_dl.data import Era5SequenceDataset, convert_era5_units


class Era5SequenceDatasetTest(unittest.TestCase):
    def _write_pair(self, root: Path, year: int, month: int, times: np.ndarray, offset: float) -> tuple[str, str]:
        latitude = np.array([2.0, 1.0, 0.0])
        longitude = np.array([10.0, 11.0])
        shape = (times.size, latitude.size, longitude.size)
        base = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + offset
        coords = {"valid_time": times, "latitude": latitude, "longitude": longitude}
        instant = xr.Dataset(
            {name: (("valid_time", "latitude", "longitude"), base + index)
             for index, name in enumerate(("msl", "u10", "v10", "i10fg", "t2m"))},
            coords=coords,
        )
        accum = xr.Dataset(
            {name: (("valid_time", "latitude", "longitude"), base + index)
             for index, name in enumerate(("ssrd", "tp"))},
            coords=coords,
        )
        stem = f"era5_std_adriatic_{year}_{month:02d}"
        instant_name = f"{stem}_instant.nc"
        accum_name = f"{stem}_accum.nc"
        instant.to_netcdf(root / instant_name)
        accum.to_netcdf(root / accum_name)
        return instant_name, accum_name

    def test_window_crosses_month_and_flips_descending_latitude(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jan_times = np.arange(
                np.datetime64("2010-01-31T21"), np.datetime64("2010-02-01T00"), np.timedelta64(1, "h")
            )
            feb_times = np.arange(
                np.datetime64("2010-02-01T00"), np.datetime64("2010-02-01T03"), np.timedelta64(1, "h")
            )
            pairs = [
                (2010, 1, *self._write_pair(root, 2010, 1, jan_times, 0.0)),
                (2010, 2, *self._write_pair(root, 2010, 2, feb_times, 100.0)),
            ]
            manifest = root / "manifest.csv"
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=("year", "month", "instant_path", "accum_path", "valid")
                )
                writer.writeheader()
                for year, month, instant, accum in pairs:
                    writer.writerow(
                        {"year": year, "month": month, "instant_path": instant,
                         "accum_path": accum, "valid": True}
                    )

            dataset = Era5SequenceDataset(
                manifest,
                root,
                station_coordinates=[[0.0, 10.0], [2.0, 11.0]],
                input_variables=("u10",),
                target_variables=("u10", "tp"),
                history_hours=2,
                forecast_hours=2,
            )
            self.assertEqual(len(dataset), 3)
            sample = dataset[1]  # 22:00, 23:00 -> 00:00, 01:00
            self.assertEqual(tuple(sample["point_values"].shape), (2, 2, 1))
            self.assertEqual(tuple(sample["target"].shape), (2, 2, 3, 2))
            self.assertTrue(np.allclose(dataset.latitudes, [0.0, 1.0, 2.0]))
            self.assertTrue(np.allclose(sample["point_coords"].numpy(), [[0.0, 0.0], [1.0, 1.0]]))
            self.assertAlmostEqual(float(sample["target"][0, 0, 0, 0]), 105.0)
            self.assertAlmostEqual(float(sample["target"][0, 1, 0, 0]), 105000.0)

    def test_unit_conversions(self) -> None:
        values = np.array([100.0], dtype=np.float32)
        self.assertAlmostEqual(float(convert_era5_units("msl", values)[0]), 1.0)
        self.assertAlmostEqual(float(convert_era5_units("t2m", values)[0]), -173.15, places=4)
        self.assertAlmostEqual(float(convert_era5_units("tp", values)[0]), 100000.0)
        self.assertAlmostEqual(float(convert_era5_units("ssrd", values)[0]), 100.0 / 3600.0)

    def test_bilinear_sampling_at_subgrid_coordinate(self) -> None:
        dataset = object.__new__(Era5SequenceDataset)
        dataset._lat_low = np.array([0])
        dataset._lat_high = np.array([1])
        dataset._lat_weight = np.array([0.5], dtype=np.float32)
        dataset._lon_low = np.array([0])
        dataset._lon_high = np.array([1])
        dataset._lon_weight = np.array([0.5], dtype=np.float32)
        grids = np.array([[[[0.0, 2.0], [4.0, 6.0]]]], dtype=np.float32)
        sampled = dataset._sample_stations(grids)
        self.assertEqual(sampled.shape, (1, 1, 1))
        self.assertAlmostEqual(float(sampled[0, 0, 0]), 3.0)


if __name__ == "__main__":
    unittest.main()
