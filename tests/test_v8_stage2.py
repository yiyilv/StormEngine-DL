import unittest

import torch

from stormengine_dl import MaskAwareReconstructionModel, StormEngineV7ForecastModel
from stormengine_dl.models.mask_aware_reconstruction import (
    freeze_spatial_modules,
    load_spatial_pretraining,
    set_processor_only_training_mode,
)


class V8Stage2Test(unittest.TestCase):
    def test_only_processor_updates_after_spatial_transfer(self) -> None:
        spatial = MaskAwareReconstructionModel(
            5, 5, include_age=True, point_hidden=8, latent_channels=8,
            height=5, width=7, static_channels=2, point_static_channels=2,
        )
        checkpoint = {
            "model_contract": {
                "version": spatial.contract_version,
                "task": "simultaneous_sparse_to_grid_reconstruction",
            },
            "model_state_dict": spatial.state_dict(),
        }
        forecast = StormEngineV7ForecastModel(
            5, 5, include_age=True, point_hidden=8, latent_channels=8,
            height=5, width=7, processor_layers=3, kernel_size=3,
            static_channels=2, point_static_channels=2,
        )
        load_spatial_pretraining(forecast, checkpoint)
        trainable = freeze_spatial_modules(forecast)
        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith("processor.") for name in trainable))

        encoder_before = {
            name: value.detach().clone() for name, value in forecast.encoder.state_dict().items()
        }
        decoder_before = {
            name: value.detach().clone() for name, value in forecast.decoder.state_dict().items()
        }
        optimizer = torch.optim.AdamW(forecast.processor.parameters(), lr=1e-3)
        set_processor_only_training_mode(forecast, True)
        self.assertFalse(forecast.encoder.training)
        self.assertTrue(forecast.processor.training)
        self.assertFalse(forecast.decoder.training)

        prediction = forecast(
            torch.randn(2, 4, 6, 5),
            torch.rand(2, 6, 2),
            torch.ones(2, 4, 6, 5, dtype=torch.bool),
            forecast_steps=3,
            observation_age=torch.zeros(2, 4, 6, 5),
            static_fields=torch.rand(2, 2, 5, 7),
            point_static=torch.rand(2, 6, 2),
        )
        optimizer.zero_grad(set_to_none=True)
        prediction.square().mean().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in forecast.processor.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in forecast.encoder.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in forecast.decoder.parameters()))
        optimizer.step()

        for name, value in forecast.encoder.state_dict().items():
            self.assertTrue(torch.equal(value, encoder_before[name]))
        for name, value in forecast.decoder.state_dict().items():
            self.assertTrue(torch.equal(value, decoder_before[name]))


if __name__ == "__main__":
    unittest.main()
