import unittest

import numpy as np
import torch

from stormengine_dl.baselines import (
    build_idw_neighbors,
    dense_grid_persistence,
    idw_interpolate,
    rmse_skill_scores,
    sparse_idw_persistence,
)


class BaselinesTest(unittest.TestCase):
    def test_idw_preserves_values_at_exact_grid_stations(self) -> None:
        stations = np.asarray([[0.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        indices, weights = build_idw_neighbors(
            stations, np.asarray([0.0]), np.asarray([0.0, 1.0]), neighbors=2
        )
        values = torch.tensor([[[2.0], [7.0]]])
        grid = idw_interpolate(values, indices, weights, 1, 2)
        self.assertTrue(torch.allclose(grid, torch.tensor([[[[2.0, 7.0]]]])))

    def test_dense_persistence_repeats_current_grid(self) -> None:
        current = torch.tensor([[[[1.0, 2.0]]]])
        prediction = dense_grid_persistence(current, 3)
        self.assertEqual(tuple(prediction.shape), (1, 3, 1, 1, 2))
        self.assertTrue(torch.equal(prediction[:, 0], prediction[:, 2]))

    def test_sparse_persistence_interpolates_common_channels_and_zeros_tp(self) -> None:
        indices = torch.tensor([[0], [1]])
        weights = torch.ones((2, 1))
        points = torch.tensor([[[10.0, 20.0], [30.0, 40.0]]])
        prediction = sparse_idw_persistence(
            points,
            ["t2m", "u10"],
            ["u10", "t2m", "tp"],
            indices,
            weights,
            1,
            2,
            2,
        )
        self.assertTrue(torch.equal(prediction[0, 0, 0], torch.tensor([[20.0, 40.0]])))
        self.assertTrue(torch.equal(prediction[0, 1, 1], torch.tensor([[10.0, 30.0]])))
        self.assertTrue(torch.equal(prediction[:, :, 2], torch.zeros_like(prediction[:, :, 2])))

    def test_rmse_skill_is_positive_when_model_is_better(self) -> None:
        def metrics(rmse: float) -> dict[str, object]:
            region = {"full": {"x": {"mae": rmse, "rmse": rmse}}}
            return {"aggregate": region, "by_lead_hour": {"1": region}}

        result = rmse_skill_scores(metrics(2.0), metrics(4.0))
        self.assertAlmostEqual(result["aggregate"]["full"]["x"]["skill"], 0.5)


if __name__ == "__main__":
    unittest.main()
