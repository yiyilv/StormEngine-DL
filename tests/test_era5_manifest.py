import tempfile
import unittest
from pathlib import Path

import numpy as np
import xarray as xr

from stormengine_dl.data.manifest import build_manifest, scan_month_files, validate_month


def write_month(root: Path, year: int, month: int, *, accum_offset_hours: int = 0) -> None:
    steps = 31 * 24
    times = np.arange(
        np.datetime64(f"{year:04d}-{month:02d}-01T00:00"),
        np.datetime64(f"{year:04d}-{month:02d}-01T00:00") + np.timedelta64(steps, "h"),
        np.timedelta64(1, "h"),
    )
    latitudes = np.array([46.5, 46.25], dtype=np.float64)
    longitudes = np.array([12.0, 12.25, 12.5], dtype=np.float64)
    shape = (steps, len(latitudes), len(longitudes))
    coordinates = {"valid_time": times, "latitude": latitudes, "longitude": longitudes}
    instant = xr.Dataset(
        {name: (("valid_time", "latitude", "longitude"), np.zeros(shape, dtype=np.float32))
         for name in ("msl", "u10", "v10", "i10fg", "t2m")},
        coords=coordinates,
    )
    accum_coordinates = dict(coordinates)
    accum_coordinates["valid_time"] = times + np.timedelta64(accum_offset_hours, "h")
    accum = xr.Dataset(
        {name: (("valid_time", "latitude", "longitude"), np.zeros(shape, dtype=np.float32))
         for name in ("ssrd", "tp")},
        coords=accum_coordinates,
    )
    instant.to_netcdf(root / f"era5_std_adriatic_{year:04d}_{month:02d}_instant.nc")
    accum.to_netcdf(root / f"era5_std_adriatic_{year:04d}_{month:02d}_accum.nc")


class ManifestTest(unittest.TestCase):
    def test_valid_pair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_month(root, 2010, 1)
            files = scan_month_files(root)
            self.assertEqual(len(files), 1)
            row = validate_month(files[0], deep_check=True)
            self.assertTrue(row.valid, row.errors)
            self.assertEqual(row.time_steps, 744)
            self.assertEqual(row.nan_count, 0)
            self.assertEqual((row.latitude_count, row.longitude_count), (2, 3))

    def test_missing_accum_file_is_retained_as_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_month(root, 2010, 1)
            (root / "era5_std_adriatic_2010_01_accum.nc").unlink()
            row = validate_month(scan_month_files(root)[0])
            self.assertFalse(row.valid)
            self.assertIn("missing accum file", row.errors)

    def test_time_mismatch_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_month(root, 2010, 1, accum_offset_hours=1)
            output = root / "manifest.csv"
            rows = build_manifest(root, output)
            self.assertEqual(len(rows), 1)
            self.assertFalse(rows[0].valid)
            self.assertIn("timestamps differ", rows[0].errors)
            self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()

