import unittest

import torch

from stormengine_dl import StormEngineForecastModel


class ModelShapeTest(unittest.TestCase):
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
        )
        values = torch.randn(batch, history, stations, input_channels)
        coords = torch.rand(batch, stations, 2)
        mask = torch.ones(batch, history, stations)
        mask[:, :, -2:] = 0
        static = torch.rand(batch, 2, height, width)

        prediction = model(
            values,
            coords,
            forecast_steps=forecast_steps,
            point_mask=mask,
            static_fields=static,
        )
        self.assertEqual(
            tuple(prediction.shape),
            (batch, forecast_steps, output_channels, height, width),
        )
        prediction.square().mean().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()

