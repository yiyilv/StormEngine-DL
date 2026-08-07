import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from stormengine_dl.data.coastal_filter import (
    adriatic_coastal_polygon,
    distance_to_adriatic_coast_km,
    is_in_adriatic_coastal_area,
)


class CoastalFilterTest(unittest.TestCase):
    def test_polygon_accepts_adriatic_coast_and_rejects_western_inland(self) -> None:
        for lat, lon in [(45.65, 13.75), (45.44, 12.33), (43.62, 13.51),
                         (42.47, 14.22), (41.13, 16.87), (40.14, 18.49)]:
            self.assertTrue(is_in_adriatic_coastal_area(lat, lon), (lat, lon))
        for lat, lon in [(41.90, 12.50), (40.85, 14.27), (43.11, 12.39)]:
            self.assertFalse(is_in_adriatic_coastal_area(lat, lon), (lat, lon))

    def test_polygon_is_valid(self) -> None:
        polygon = adriatic_coastal_polygon()
        self.assertTrue(polygon.is_valid)
        self.assertEqual(polygon.geom_type, "Polygon")

    def test_distance_is_recorded_in_kilometres(self) -> None:
        self.assertLess(distance_to_adriatic_coast_km(43.62, 13.51), 0.01)
        self.assertGreater(distance_to_adriatic_coast_km(41.90, 12.50), 100.0)

    def test_measurement_script_reapplies_same_filter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "measurements.csv", root / "coastal.csv"
            with source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["station_id", "lat", "lon", "value"])
                writer.writeheader()
                writer.writerows([
                    {"station_id": "coast", "lat": 43.62, "lon": 13.51, "value": 1},
                    {"station_id": "rome", "lat": 41.90, "lon": 12.50, "value": 2},
                ])
            subprocess.run(
                [sys.executable, "scripts/filter_coastal_observations.py", "--input", str(source),
                 "--output", str(output)], check=True, capture_output=True, text=True,
            )
            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["station_id"] for row in rows], ["coast"])


if __name__ == "__main__":
    unittest.main()
