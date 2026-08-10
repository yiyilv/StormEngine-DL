import unittest

import numpy as np

from stormengine_dl.data.v7_input import adapt_dpc_to_v7


class V7InputTests(unittest.TestCase):
    def test_dpc_mapping_normalization_mask_age_and_coordinates(self) -> None:
        names = np.asarray([
            "station_pressure_hpa", "u10", "v10", "wind_gust_max", "t2m",
            "relative_humidity", "tp",
        ])
        values = np.zeros((1, 2, 7), dtype=np.float32)
        mask = np.zeros_like(values, dtype=bool)
        ages = np.full_like(values, -1.0)
        source = {name: index for index, name in enumerate(names)}
        selected = [source[name] for name in ("u10", "v10", "wind_gust_max", "t2m", "tp")]
        values[0, 0, selected] = [3, -2, 8, 20, 1.5]
        mask[0, 0, selected] = True
        ages[0, 0, selected] = [0, 30, 60, 15, 45]
        values[0, 1, source["t2m"]] = 10
        mask[0, 1, source["t2m"]] = True
        ages[0, 1, source["t2m"]] = 0
        normalization = {"variables": {
            "u10": {"mean": 1, "std": 2}, "v10": {"mean": 0, "std": 2},
            "i10fg": {"mean": 4, "std": 2}, "t2m": {"mean": 10, "std": 5},
            "tp": {"mean": .5, "std": .5},
        }}
        batch = adapt_dpc_to_v7(
            times=np.asarray([np.datetime64("2026-08-01T00")]),
            station_ids=np.asarray(["A", "B"]), source_variable_names=names,
            source_values=values, source_mask=mask, source_age_minutes=ages,
            coordinates=np.asarray([[39, 12], [46.5, 20]], np.float32),
            station_static=np.ones((2, 2), np.float32), normalization=normalization,
            expected_station_ids=("A", "B"),
        )
        np.testing.assert_allclose(batch.values[0, 0], [1, -1, 2, 2, 2])
        np.testing.assert_allclose(batch.observation_age[0, 0], [0, .5, 1, .25, .75])
        np.testing.assert_allclose(batch.values[0, 1], [0, 0, 0, 0, 0])
        np.testing.assert_array_equal(
            batch.value_mask[0, 1], [False, False, False, True, False]
        )
        np.testing.assert_allclose(batch.coordinates, [[0, 0], [1, 1]])
        self.assertEqual(batch.variable_names, ("u10", "v10", "i10fg", "t2m", "tp"))
        self.assertEqual(batch.source_type, ("physical", "physical"))
        np.testing.assert_array_equal(batch.station_present, batch.value_mask.any(axis=-1))

    def test_wrong_station_order_is_rejected(self) -> None:
        shape = (1, 2, 5)
        with self.assertRaisesRegex(ValueError, "station order"):
            adapt_dpc_to_v7(
                times=np.asarray([np.datetime64("2026-08-01T00")]),
                station_ids=np.asarray(["B", "A"]),
                source_variable_names=np.asarray(
                    ["u10", "v10", "wind_gust_max", "t2m", "tp"]
                ),
                source_values=np.zeros(shape, np.float32), source_mask=np.ones(shape, bool),
                source_age_minutes=np.zeros(shape, np.float32),
                coordinates=np.asarray([[40, 13], [41, 14]], np.float32),
                station_static=np.ones((2, 2), np.float32),
                normalization={"variables": {
                    name: {"mean": 0, "std": 1}
                    for name in ("u10", "v10", "i10fg", "t2m", "tp")
                }},
                expected_station_ids=("A", "B"),
            )


if __name__ == "__main__":
    unittest.main()
