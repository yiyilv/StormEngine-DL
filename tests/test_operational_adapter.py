import csv
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from stormengine_dl.data.official_observations import Observation
from stormengine_dl.data.operational_adapter import (
    build_operational_tensors,
    meteorological_wind_to_uv,
)


def observation(
    code: str,
    value: float,
    time: str,
    *,
    station_id: str = "MH::test::a",
    indicator: int = 254,
    start: int = 0,
    end: int = 0,
    height_m: float | None = None,
) -> Observation:
    names = {
        "B10004": ("pressure", "hPa"),
        "B11001": ("wind_direction", "degree"),
        "B11002": ("wind_speed", "m s-1"),
        "B11041": ("wind_gust_max", "m s-1"),
        "B12101": ("air_temperature", "degree_Celsius"),
        "B13003": ("relative_humidity", "%"),
        "B13011": ("precipitation_amount", "mm"),
    }
    canonical, unit = names[code]
    aggregation = end - start if indicator in {1, 2} and end > start else None
    return Observation(
        station_id=station_id,
        station_name="Test",
        network="test",
        latitude=43.0,
        longitude=13.0,
        elevation_m=10.0,
        observation_time=time,
        bufr_code=code,
        canonical_variable=canonical,
        raw_value=value,
        canonical_value=value,
        raw_unit=unit,
        canonical_unit=unit,
        timerange_indicator=indicator,
        timerange_start_seconds=start,
        timerange_end_seconds=end,
        aggregation_seconds=aggregation,
        level_type=103 if height_m is not None else 1,
        level_value_raw=height_m * 1000 if height_m is not None else None,
        height_above_ground_m=height_m,
        source_file="sample.json",
    )


class OperationalAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.registry = Path(self.temp.name) / "registry.csv"
        fieldnames = [
            "station_id", "station_name", "latitude", "longitude", "station_type",
            "network", "coordinate_source", "pretraining_value_source",
            "operational_value_source", "enabled", "profile_land_only",
            "profile_sea_only", "profile_dpc_plus_sea", "dist_to_coast_km", "notes",
        ]
        rows = [
            {
                "station_id": "LAND::MH::test::a", "station_name": "A", "latitude": 43,
                "longitude": 13, "station_type": "physical_land", "network": "test",
                "coordinate_source": "test", "pretraining_value_source": "ERA5",
                "operational_value_source": "MeteoHub", "enabled": True,
                "profile_land_only": True, "profile_sea_only": False,
                "profile_dpc_plus_sea": True, "dist_to_coast_km": 1, "notes": "",
            },
            {
                "station_id": "LAND::MH::test::offline", "station_name": "Offline",
                "latitude": 44, "longitude": 14, "station_type": "physical_land",
                "network": "test", "coordinate_source": "test",
                "pretraining_value_source": "ERA5", "operational_value_source": "MeteoHub",
                "enabled": True, "profile_land_only": True, "profile_sea_only": False,
                "profile_dpc_plus_sea": True, "dist_to_coast_km": 1, "notes": "",
            },
        ]
        with self.registry.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_wind_uses_meteorological_convention(self) -> None:
        u, v = meteorological_wind_to_uv(10.0, 90.0)
        self.assertAlmostEqual(u, -10.0)
        self.assertAlmostEqual(v, 0.0, places=6)
        u, v = meteorological_wind_to_uv(10.0, 0.0)
        self.assertAlmostEqual(u, 0.0)
        self.assertAlmostEqual(v, -10.0)

    def test_no_future_leak_and_partial_variable_mask(self) -> None:
        rows = [
            observation("B12101", 20.0, "2026-08-01T09:30:00Z", height_m=2.0),
            observation("B10004", 1000.0, "2026-08-01T10:01:00Z"),
        ]
        batch = build_operational_tensors(
            rows,
            self.registry,
            np.asarray(["2026-08-01T10:00", "2026-08-01T11:00"], dtype="datetime64[m]"),
        )
        pressure = batch.variable_names.index("station_pressure_hpa")
        temperature = batch.variable_names.index("t2m")
        self.assertFalse(batch.value_mask[0, 0, pressure])
        self.assertTrue(batch.value_mask[1, 0, pressure])
        self.assertEqual(batch.observation_age_minutes[1, 0, pressure], 59.0)
        self.assertEqual(
            batch.source_time[1, 0, pressure],
            np.datetime64("2026-08-01T10:01:00", "ns"),
        )
        self.assertTrue(batch.value_mask[0, 0, temperature])
        self.assertEqual(batch.observation_age_minutes[0, 0, temperature], 30.0)
        self.assertFalse(batch.value_mask[:, 1].any())
        self.assertFalse(batch.station_present[:, 1].any())
        self.assertNotIn("msl", batch.variable_names)

    def test_wind_pair_requires_same_context(self) -> None:
        direction = observation("B11001", 90.0, "2026-08-01T10:00:00Z", height_m=10.0)
        speed = observation("B11002", 5.0, "2026-08-01T10:00:00Z", height_m=10.0)
        mismatched = replace(speed, level_value_raw=2000.0, height_above_ground_m=2.0)
        times = np.asarray(["2026-08-01T10:00"], dtype="datetime64[m]")
        rejected = build_operational_tensors([direction, mismatched], self.registry, times)
        self.assertFalse(rejected.value_mask.any())
        accepted = build_operational_tensors([direction, speed], self.registry, times)
        u10 = accepted.variable_names.index("u10")
        v10 = accepted.variable_names.index("v10")
        self.assertAlmostEqual(accepted.values[0, 0, u10], -5.0)
        self.assertAlmostEqual(accepted.values[0, 0, v10], 0.0, places=6)

    def test_precipitation_requires_complete_non_overlapping_hour(self) -> None:
        complete = [
            observation("B13011", 0.4, "2026-08-01T09:30:00Z", indicator=1, end=1800),
            observation("B13011", 0.6, "2026-08-01T10:00:00Z", indicator=1, end=1800),
        ]
        times = np.asarray(["2026-08-01T10:00"], dtype="datetime64[m]")
        batch = build_operational_tensors(complete, self.registry, times)
        tp = batch.variable_names.index("tp")
        self.assertTrue(batch.value_mask[0, 0, tp])
        self.assertAlmostEqual(batch.values[0, 0, tp], 1.0)
        overlapping = complete + [
            observation("B13011", 0.2, "2026-08-01T09:45:00Z", indicator=1, end=900)
        ]
        rejected = build_operational_tensors(overlapping, self.registry, times)
        self.assertFalse(rejected.value_mask[0, 0, tp])
        self.assertEqual(rejected.diagnostics["rejected_precipitation_hours"], 1)

        exact = build_operational_tensors(
            [observation("B13011", 1.5, "2026-08-01T10:00:00Z", indicator=1, end=3600)],
            self.registry,
            times,
        )
        self.assertTrue(exact.value_mask[0, 0, tp])
        self.assertAlmostEqual(exact.values[0, 0, tp], 1.5)

    def test_registry_order_is_preserved(self) -> None:
        batch = build_operational_tensors(
            [],
            self.registry,
            np.asarray(["2026-08-01T10:00"], dtype="datetime64[m]"),
        )
        self.assertEqual(
            batch.station_ids,
            ("LAND::MH::test::a", "LAND::MH::test::offline"),
        )


if __name__ == "__main__":
    unittest.main()
