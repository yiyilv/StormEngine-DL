import json
import tempfile
import unittest
from pathlib import Path

from stormengine_dl.data.official_observations import (
    audit_observations,
    audit_observations_sqlite,
    iter_meteohub_observations,
    parse_meteohub_record,
)


SAMPLE = {
    "network": "dpcn-marche",
    "lon": 1350830,
    "lat": 4361030,
    "date": "2026-06-18T11:30:00Z",
    "data": [
        {"vars": {"B01019": {"v": "Ancona Regione"}, "B05001": {"v": 43.6103},
                  "B06001": {"v": 13.5083}, "B07030": {"v": 47.0}}},
        {"timerange": [1, 0, 900], "level": [1, None, None, None],
         "vars": {"B13011": {"v": 1.2}}},
        {"timerange": [254, 0, 0], "level": [103, 2000, None, None],
         "vars": {"B12101": {"v": 303.05}, "B13003": {"v": 37}}},
        {"timerange": [254, 0, 0], "level": [103, 10000, None, None],
         "vars": {"B11001": {"v": 67}, "B11002": {"v": 3.3}}},
        {"timerange": [2, 0, 3600], "level": [103, 10000, None, None],
         "vars": {"B11041": {"v": 8.1}, "B13215": {"v": 99}}},
    ],
}


class OfficialObservationTests(unittest.TestCase):
    def test_preserves_time_range_level_and_unmapped_variables(self) -> None:
        rows = parse_meteohub_record(SAMPLE, "sample.json")
        by_code = {row.bufr_code: row for row in rows}
        self.assertAlmostEqual(by_code["B12101"].canonical_value, 29.9)
        self.assertEqual(by_code["B12101"].height_above_ground_m, 2.0)
        self.assertEqual(by_code["B13011"].aggregation_seconds, 900)
        self.assertEqual(by_code["B11041"].aggregation_seconds, 3600)
        self.assertEqual(by_code["B13215"].canonical_variable, "river_level")

    def test_jsonl_iterator_and_audit_remove_overlapping_extracts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.json"
            second = Path(directory) / "second.json"
            first.write_text(json.dumps(SAMPLE) + "\n", encoding="utf-8")
            second.write_text(json.dumps(SAMPLE) + "\n", encoding="utf-8")
            rows = list(iter_meteohub_observations([first, second]))
            report = audit_observations(rows, {rows[0].station_id})
            self.assertEqual(report["summary"]["raw_measurement_count"], 14)
            self.assertEqual(report["summary"]["unique_measurement_count"], 7)
            self.assertEqual(report["summary"]["selected_station_count"], 1)
            self.assertEqual(report["summary"]["unmapped_bufr_codes"], [])

    def test_streaming_audit_matches_in_memory_summary(self) -> None:
        revised = json.loads(json.dumps(SAMPLE))
        revised["data"][1]["vars"]["B13011"]["v"] = 2.4
        rows = parse_meteohub_record(SAMPLE, "first.json")
        rows.extend(parse_meteohub_record(revised, "second.json"))
        selected = {rows[0].station_id}
        expected = audit_observations(rows, selected)
        with tempfile.TemporaryDirectory() as directory:
            actual = audit_observations_sqlite(
                rows,
                Path(directory) / "audit.sqlite3",
                selected,
                batch_size=3,
            )
        self.assertEqual(actual["summary"], expected["summary"])
        self.assertEqual(actual["variable_summary"], expected["variable_summary"])
        self.assertEqual(actual["timerange_summary"], expected["timerange_summary"])
        self.assertEqual(actual["station_variable_summary"], expected["station_variable_summary"])


if __name__ == "__main__":
    unittest.main()
