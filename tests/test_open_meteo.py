import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from stormengine_dl.data.open_meteo import load_marine_points, wind_components


class OpenMeteoTests(unittest.TestCase):
    def test_wind_cardinal_directions(self) -> None:
        speed=np.ones(4); direction=np.asarray([0,90,180,270])
        u,v=wind_components(speed,direction)
        np.testing.assert_allclose(u,[0,-1,0,1],atol=1e-7)
        np.testing.assert_allclose(v,[-1,0,1,0],atol=1e-7)

    def test_registry_selection_preserves_order_and_requires_151(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"stations.csv"
            fields=["station_id","latitude","longitude","station_type","enabled","profile_sea_only","coordinate_source"]
            with path.open("w",newline="",encoding="utf-8") as handle:
                writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader()
                for index in range(151): writer.writerow({"station_id":f"S{index:03}","latitude":40,"longitude":15,"station_type":"virtual_sea","enabled":"True","profile_sea_only":"True","coordinate_source":"registry"})
                writer.writerow({"station_id":"OLD","latitude":40,"longitude":15,"station_type":"legacy_virtual_coastal","enabled":"False","profile_sea_only":"False","coordinate_source":"legacy"})
            selected=load_marine_points(path)
            self.assertEqual(len(selected),151); self.assertEqual(selected[0].station_id,"S000"); self.assertEqual(selected[-1].station_id,"S150")


if __name__=="__main__": unittest.main()
