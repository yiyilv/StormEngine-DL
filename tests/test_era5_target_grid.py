import tempfile
import unittest
from pathlib import Path

import numpy as np
import xarray as xr

from stormengine_dl.data import load_era5_target_grid


class Era5TargetGridTests(unittest.TestCase):
    def _write_files(self, root: Path, *, shifted_accum: bool = False) -> tuple[Path, Path]:
        times = np.arange(
            np.datetime64("2026-08-01T00"), np.datetime64("2026-08-01T04"),
            np.timedelta64(1, "h"),
        )
        latitude = np.asarray([2.0, 1.0, 0.0])
        longitude = np.asarray([10.0, 11.0])
        shape = (len(times), len(latitude), len(longitude))
        base = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        coords = {"valid_time": times, "latitude": latitude, "longitude": longitude}
        instant = xr.Dataset(
            {
                "msl": (("valid_time", "latitude", "longitude"), base + 100000.0),
                "u10": (("valid_time", "latitude", "longitude"), base),
                "v10": (("valid_time", "latitude", "longitude"), base + 1),
                "i10fg": (("valid_time", "latitude", "longitude"), base + 2),
                "t2m": (("valid_time", "latitude", "longitude"), base + 273.15),
            }, coords=coords,
        )
        accum_times = times + np.timedelta64(1, "h") if shifted_accum else times
        accum = xr.Dataset(
            {"tp": (("valid_time", "latitude", "longitude"), base / 1000.0)},
            coords={"valid_time": accum_times, "latitude": latitude, "longitude": longitude},
        )
        instant_path, accum_path = root / "instant.nc", root / "accum.nc"
        instant.to_netcdf(instant_path); accum.to_netcdf(accum_path)
        return instant_path, accum_path

    def test_loads_physical_units_and_flips_latitude(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            instant, accum = self._write_files(Path(directory))
            target = load_era5_target_grid(instant, accum, ("msl", "u10", "t2m", "tp"))
            self.assertEqual(target.values.shape, (4, 4, 3, 2))
            np.testing.assert_array_equal(target.latitudes, [0.0, 1.0, 2.0])
            self.assertAlmostEqual(float(target.values[0, 0, 0, 0]), 1000.04, places=3)
            self.assertAlmostEqual(float(target.values[0, 1, 0, 0]), 4.0)
            self.assertAlmostEqual(float(target.values[0, 2, 0, 0]), 4.0, places=4)
            self.assertAlmostEqual(float(target.values[0, 3, 0, 0]), 4.0)
            indices = target.indices_for(np.asarray(["2026-08-01T01", "2026-08-01T03"], dtype="datetime64[h]"))
            np.testing.assert_array_equal(indices, [1, 3])
            with self.assertRaisesRegex(ValueError, "missing exact timestamps"):
                target.indices_for(np.asarray(["2026-08-01T04"], dtype="datetime64[h]"))

    def test_rejects_mismatched_instant_and_accum_axes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            instant, accum = self._write_files(Path(directory), shifted_accum=True)
            with self.assertRaisesRegex(ValueError, "identical axes"):
                load_era5_target_grid(instant, accum, ("u10", "tp"))


if __name__ == "__main__":
    unittest.main()
