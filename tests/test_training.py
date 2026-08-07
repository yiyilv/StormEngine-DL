import unittest

import torch

from stormengine_dl.training import RegionMetricAccumulator, sea_weight_map, weighted_mse


class TrainingHelpersTest(unittest.TestCase):
    def test_v6_sea_weights_are_mean_normalized(self) -> None:
        lsm = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        weights = sea_weight_map(lsm, sea_weight=2.0)
        self.assertAlmostEqual(float(weights.mean()), 1.0)
        self.assertAlmostEqual(float(weights[0, 1] / weights[0, 0]), 2.0)

    def test_weighted_mse_uses_spatial_map(self) -> None:
        prediction = torch.tensor([[[[[1.0, 2.0]]]]])
        target = torch.zeros_like(prediction)
        weights = torch.tensor([[0.5, 1.5]])
        self.assertAlmostEqual(float(weighted_mse(prediction, target, weights)), 3.25)

    def test_region_metrics_are_per_variable(self) -> None:
        prediction = torch.tensor([[[[[2.0, 3.0]], [[4.0, 6.0]]]]])
        target = torch.tensor([[[[[1.0, 1.0]], [[2.0, 2.0]]]]])
        metrics = RegionMetricAccumulator(("a", "b"))
        metrics.update(prediction, target, torch.tensor([[1.0, 0.0]]))
        result = metrics.compute()
        self.assertAlmostEqual(result["land"]["a"]["mae"], 1.0)
        self.assertAlmostEqual(result["sea"]["b"]["rmse"], 4.0)


if __name__ == "__main__":
    unittest.main()
