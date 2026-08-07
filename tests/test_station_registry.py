import csv
import json
import tempfile
import unittest
from pathlib import Path

from stormengine_dl.data import build_station_registry, load_station_coordinates


class StationRegistryTest(unittest.TestCase):
    def _write(self, path: Path, rows: list[dict[str, object]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_profiles_separate_physical_sea_and_legacy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dpc = root / "dpc.csv"
            virtual = root / "virtual.csv"
            legacy = root / "abr.csv"
            output = root / "registry.csv"
            self._write(
                dpc,
                [
                    {"station_id": "1", "station_name": "DPC A", "lat": 44, "lon": 13,
                     "gestore": "DPC", "dist_to_coast_km": 5},
                    {"station_id": "2", "station_name": "Outside", "lat": 48, "lon": 13,
                     "gestore": "DPC", "dist_to_coast_km": 5},
                ],
            )
            self._write(
                virtual,
                [
                    {"station_id": "S1", "sensor_name": "Sea", "lat": 43, "lon": 15,
                     "gestore": "VIRTUAL_ADRIATIC_SEA"},
                    {"station_id": "I1", "sensor_name": "Ionian", "lat": 40, "lon": 18,
                     "gestore": "VIRTUAL_IONIAN_SEA"},
                ],
            )
            self._write(
                legacy,
                [{"station_id": "A1", "sensor_name": "Legacy", "lat": 42, "lon": 14,
                  "gestore": "ARTA_VIRTUAL"}],
            )
            records = build_station_registry(dpc, virtual, output, legacy_coastal_paths=[legacy])
            self.assertEqual(len(records), 3)
            combined, combined_meta = load_station_coordinates(output, "dpc_plus_sea")
            land, _ = load_station_coordinates(output, "land_only")
            sea, _ = load_station_coordinates(output, "sea_only")
            self.assertEqual(combined.shape, (2, 2))
            self.assertEqual(land.shape, (1, 2))
            self.assertEqual(sea.shape, (1, 2))
            self.assertEqual({row["station_type"] for row in combined_meta}, {"physical_land", "virtual_sea"})

    def test_unknown_profile_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            load_station_coordinates("unused.csv", "not_a_profile")

    def test_recovers_selected_meteohub_network_from_json_lines(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dpc, virtual, raw, output = (
                root / "dpc.csv", root / "virtual.csv", root / "raw.json", root / "registry.csv"
            )
            self._write(
                dpc,
                [{"station_id": "1", "station_name": "North", "lat": 44, "lon": 13,
                  "gestore": "DPC", "dist_to_coast_km": 5}],
            )
            self._write(
                virtual,
                [{"station_id": "S1", "sensor_name": "Sea", "lat": 43, "lon": 15,
                  "gestore": "VIRTUAL_ADRIATIC_SEA"}],
            )
            observations = [
                {"network": "dpcn-puglia", "data": [{"vars": {
                    "B01019": {"v": "Bari"}, "B05001": {"v": 41.1}, "B06001": {"v": 16.9}
                }}]},
                {"network": "dpcn-other", "data": [{"vars": {
                    "B01019": {"v": "Other"}, "B05001": {"v": 41.2}, "B06001": {"v": 16.8}
                }}]},
            ]
            raw.write_text("\n".join(json.dumps(item) for item in observations), encoding="utf-8")
            records = build_station_registry(
                dpc, virtual, output, meteohub_json_paths=[raw],
                meteohub_networks=["dpcn-puglia"]
            )
            physical = [record for record in records if record.station_type == "physical_land"]
            self.assertEqual(len(physical), 2)
            self.assertEqual({record.station_name for record in physical}, {"North", "Bari"})


if __name__ == "__main__":
    unittest.main()
