import unittest

import torch

from stormengine_dl import StormEngineV7ForecastModel
from stormengine_dl.models.mask_aware import require_v7_checkpoint_contract


class V7ModelTests(unittest.TestCase):
    def test_mask_aware_shape_backward_and_all_missing_are_finite(self) -> None:
        model = StormEngineV7ForecastModel(
            input_channels=3, output_channels=5, include_age=True,
            point_hidden=8, latent_channels=8, height=5, width=7,
            processor_layers=1, static_channels=2, point_static_channels=2,
        )
        values = torch.randn(2, 4, 6, 3)
        mask = torch.rand(2, 4, 6, 3) > 0.4
        mask[:, :, -1] = False
        values[~mask] = 0
        prediction = model(
            values,
            torch.rand(2, 6, 2),
            mask,
            2,
            observation_age=torch.rand(2, 4, 6, 3),
            static_fields=torch.rand(2, 2, 5, 7),
            point_static=torch.rand(2, 6, 2),
        )
        self.assertEqual(tuple(prediction.shape), (2, 2, 5, 5, 7))
        self.assertTrue(torch.isfinite(prediction).all())
        prediction.mean().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_age_contract_and_v6_checkpoint_are_rejected(self) -> None:
        model = StormEngineV7ForecastModel(
            input_channels=3, output_channels=1, include_age=True,
            point_hidden=4, latent_channels=4, height=3, width=3,
        )
        with self.assertRaisesRegex(ValueError, "observation_age"):
            model(torch.zeros(1, 2, 1, 3), torch.zeros(1, 1, 2),
                  torch.ones(1, 2, 1, 3, dtype=torch.bool), 1)
        with self.assertRaisesRegex(ValueError, "incompatible"):
            require_v7_checkpoint_contract({"model_state_dict": {}}, {
                "version": model.contract_version,
                "input_variables": ["u10", "v10", "t2m"],
                "include_age": True,
            })


if __name__ == "__main__":
    unittest.main()
