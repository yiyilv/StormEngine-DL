import unittest

import torch

from stormengine_dl import StormEngineForecastModel, StormEngineReconstructionModel


class ModelShapeTest(unittest.TestCase):
    def test_encoder_decoder_reconstruction_bypasses_processor(self) -> None:
        model = StormEngineReconstructionModel(
            input_channels=5,
            output_channels=4,
            point_hidden=8,
            latent_channels=8,
            height=5,
            width=7,
            static_channels=2,
            point_static_channels=2,
        )
        prediction = model(
            torch.randn(2, 1, 6, 5),
            torch.rand(2, 6, 2),
            torch.ones(2, 1, 6),
            torch.rand(2, 2, 5, 7),
            torch.rand(2, 6, 2),
        )
        self.assertEqual(tuple(prediction.shape), (2, 1, 4, 5, 7))
        self.assertFalse(hasattr(model, "processor"))
        prediction.mean().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.encoder.parameters()))

    def test_end_to_end_forecast_shape_and_backward(self) -> None:
        batch, history, stations = 2, 4, 10
        input_channels, output_channels = 5, 5
        height, width, forecast_steps = 9, 11, 3

        model = StormEngineForecastModel(
            input_channels=input_channels,
            output_channels=output_channels,
            point_hidden=16,
            latent_channels=16,
            height=height,
            width=width,
            processor_layers=2,
            static_channels=2,
            point_static_channels=2,
        )
        values = torch.randn(batch, history, stations, input_channels)
        coords = torch.rand(batch, stations, 2)
        mask = torch.ones(batch, history, stations)
        mask[:, :, -2:] = 0
        static = torch.rand(batch, 2, height, width)
        point_static = torch.zeros(batch, stations, 2)
        point_static[:, : stations // 2, 0] = 1
        point_static[:, stations // 2 :, 1] = 1

        prediction = model(
            values,
            coords,
            forecast_steps=forecast_steps,
            point_mask=mask,
            static_fields=static,
            point_static=point_static,
        )
        self.assertEqual(
            tuple(prediction.shape),
            (batch, forecast_steps, output_channels, height, width),
        )
        prediction.square().mean().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
