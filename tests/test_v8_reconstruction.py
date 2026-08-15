import unittest

import numpy as np
import torch

from stormengine_dl import MaskAwareReconstructionModel, StormEngineV7ForecastModel
from stormengine_dl.data import V7CachedReconstructionDataset
from stormengine_dl.models.mask_aware_reconstruction import (
    V8_RECONSTRUCTION_CONTRACT,
    load_spatial_pretraining,
)


class _FakeV7Source:
    history_hours = 3
    target_variables = ("msl", "u10")

    def __init__(self) -> None:
        self.target_grids = np.arange(8 * 2 * 2 * 2, dtype=np.float32).reshape(8, 2, 2, 2)
        self.epoch = -1
        self.closed = False

    def __len__(self) -> int:
        return 1

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def close(self) -> None:
        self.closed = True

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        del item
        mask = torch.ones(3, 4, 2, dtype=torch.bool)
        mask[-1, -1, -1] = False
        values = torch.arange(24, dtype=torch.float32).reshape(3, 4, 2)
        values[~mask] = 0
        return {
            "point_values": values,
            "value_mask": mask,
            "observation_age": torch.zeros_like(values),
            "point_coords": torch.rand(4, 2),
            "point_static": torch.rand(4, 2),
            "source_type": torch.tensor([0, 0, 1, 1]),
            "target": torch.zeros(2, 2, 2, 2),
            "start_index": torch.tensor(1),
        }


class V8ReconstructionTests(unittest.TestCase):
    def test_dataset_uses_last_history_hour_and_simultaneous_grid(self) -> None:
        source = _FakeV7Source()
        dataset = V7CachedReconstructionDataset(source, ["msl", "u10"])  # type: ignore[arg-type]
        sample = dataset[0]
        self.assertEqual(tuple(sample["point_values"].shape), (1, 4, 2))
        self.assertTrue(torch.equal(sample["point_values"][0], source[0]["point_values"][-1]))
        expected = torch.from_numpy(source.target_grids[3].copy())[None]
        self.assertTrue(torch.equal(sample["target"], expected))
        dataset.set_epoch(7)
        self.assertEqual(source.epoch, 7)
        dataset.close()
        self.assertTrue(source.closed)

    def test_model_shape_backward_and_all_missing_are_finite(self) -> None:
        model = MaskAwareReconstructionModel(
            3, 5, include_age=True, point_hidden=8, latent_channels=8,
            height=5, width=7, static_channels=2, point_static_channels=2,
        )
        values = torch.zeros(2, 1, 6, 3)
        mask = torch.zeros_like(values, dtype=torch.bool)
        prediction = model(
            values, torch.rand(2, 6, 2), mask,
            observation_age=torch.zeros_like(values),
            static_fields=torch.rand(2, 2, 5, 7),
            point_static=torch.rand(2, 6, 2),
        )
        self.assertEqual(tuple(prediction.shape), (2, 1, 5, 5, 7))
        self.assertTrue(torch.isfinite(prediction).all())
        prediction.mean().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_spatial_weights_transfer_without_changing_processor(self) -> None:
        spatial = MaskAwareReconstructionModel(
            3, 5, include_age=True, point_hidden=8, latent_channels=8,
            height=5, width=7, static_channels=2, point_static_channels=2,
        )
        forecast = StormEngineV7ForecastModel(
            3, 5, include_age=True, point_hidden=8, latent_channels=8,
            height=5, width=7, processor_layers=1,
            static_channels=2, point_static_channels=2,
        )
        before = {name: value.detach().clone() for name, value in forecast.processor.state_dict().items()}
        checkpoint = {
            "model_contract": {"version": V8_RECONSTRUCTION_CONTRACT},
            "model_state_dict": spatial.state_dict(),
        }
        load_spatial_pretraining(forecast, checkpoint)
        for name, value in spatial.encoder.state_dict().items():
            self.assertTrue(torch.equal(value, forecast.encoder.state_dict()[name]))
        for name, value in spatial.decoder.state_dict().items():
            self.assertTrue(torch.equal(value, forecast.decoder.state_dict()[name]))
        for name, value in before.items():
            self.assertTrue(torch.equal(value, forecast.processor.state_dict()[name]))

    def test_non_reconstruction_checkpoint_is_rejected(self) -> None:
        forecast = StormEngineV7ForecastModel(1, 1, include_age=False, height=3, width=3)
        with self.assertRaisesRegex(ValueError, "not a V8"):
            load_spatial_pretraining(forecast, {"model_contract": {"version": "wrong"}})


if __name__ == "__main__":
    unittest.main()

