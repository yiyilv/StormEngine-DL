import tempfile
import unittest
from pathlib import Path

from stormengine_dl.data.official_station_catalog import (
    collect_abruzzo_stations,
    collect_meteohub_stations,
    write_official_station_catalog,
)


class OfficialStationCatalogTest(unittest.TestCase):
    def test_meteohub_union_filters_network_and_domain(self) -> None:
        first = {"data": [{"stat": {"lat": 44, "lon": 13, "net": "dpcn-marche",
                                     "details": [{"var": "B01019", "val": "A"}]},
                           "prod": [{"var": "B12101"}]}]}
        second = {"data": [
            {"stat": {"lat": 44, "lon": 13, "net": "dpcn-marche",
                      "details": [{"var": "B01019", "val": "A"}]},
             "prod": [{"var": "B13011"}]},
            {"stat": {"lat": 44, "lon": 13.5, "net": "unrelated", "details": []}},
            {"stat": {"lat": 48, "lon": 13, "net": "dpcn-veneto", "details": []}},
        ]}
        stations = collect_meteohub_stations([("t0", first), ("t1", second)])
        self.assertEqual(len(stations), 1)
        self.assertEqual(stations[0].observed_snapshots, 2)
        self.assertEqual(stations[0].variables, "B12101|B13011")

    def test_abruzzo_keeps_only_official_polaris_rows(self) -> None:
        payload = {"data": [
            {"polaris_id": 7, "name": "Official", "lat": "42", "lon": "14",
             "source": "Regione Abruzzo", "last_week": 6},
            {"polaris_id": None, "name": "External", "lat": "42", "lon": "14.2",
             "source": "External", "last_week": 0},
        ]}
        stations = collect_abruzzo_stations(payload)
        self.assertEqual([station.station_name for station in stations], ["Official"])

    def test_writer_deduplicates_same_coordinate(self) -> None:
        payload = {"data": [{"polaris_id": 7, "name": "Official", "lat": "42", "lon": "14",
                             "source": "Regione Abruzzo", "last_week": 6}]}
        station = collect_abruzzo_stations(payload)[0]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "catalog.csv"
            written = write_official_station_catalog([station, station], output)
            self.assertEqual(len(written), 1)
            self.assertEqual(len(output.read_text().splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
