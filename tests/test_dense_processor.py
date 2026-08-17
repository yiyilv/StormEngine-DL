import unittest

import numpy as np
import torch

from stormengine_dl.data import DenseGridForecastDataset
from stormengine_dl.models import make_dense_processor_model


class _FakeCache:
    history_hours = 3
    forecast_hours = 2
    target_variables = ("msl", "u10")

    def __init__(self) -> None:
        self.target_grids = np.arange(10 * 2 * 3 * 5, dtype=np.float32).reshape(10, 2, 3, 5)
        self.window_starts = np.asarray([0, 2], dtype=np.int64)
        self.global_indices = np.arange(10, dtype=np.int64)
        self.closed = False

    def __len__(self) -> int:
        return 2

    def close(self) -> None:
        self.closed = True


class DenseProcessorTests(unittest.TestCase):
    def test_dense_dataset_uses_history_and_future_without_point_data(self) -> None:
        source = _FakeCache()
        dataset = DenseGridForecastDataset(source)  # type: ignore[arg-type]
        sample = dataset[1]
        self.assertTrue(torch.equal(sample["history"], torch.from_numpy(source.target_grids[2:5])))
        self.assertTrue(torch.equal(sample["target"], torch.from_numpy(source.target_grids[5:7])))
        dataset.close()
        self.assertTrue(source.closed)

    def test_both_processor_families_share_the_forecast_contract(self) -> None:
        history = torch.randn(2, 3, 2, 5, 7)
        for family in ("convgru", "factorized_vit"):
            model = make_dense_processor_model(
                family,
                input_channels=2,
                output_channels=2,
                latent_channels=8,
                height=5,
                width=7,
                history_steps=3,
                forecast_steps=2,
                processor_layers=2,
                kernel_size=3,
                patch_size=4,
                transformer_dimension=16,
                transformer_heads=4,
            )
            prediction = model(history, 2)
            self.assertEqual(tuple(prediction.shape), (2, 2, 2, 5, 7))
            self.assertTrue(torch.isfinite(prediction).all())
            prediction.mean().backward()
            self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))

    def test_vit_rejects_a_different_horizon(self) -> None:
        model = make_dense_processor_model(
            "factorized_vit", input_channels=2, output_channels=2,
            latent_channels=8, height=5, width=7, history_steps=3,
            forecast_steps=2, processor_layers=1, patch_size=4,
            transformer_dimension=16, transformer_heads=4,
        )
        with self.assertRaisesRegex(ValueError, "forecast_steps"):
            model(torch.randn(1, 3, 2, 5, 7), 1)


if __name__ == "__main__":
    unittest.main()
