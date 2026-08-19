from __future__ import annotations

import torch

from stormengine_dl import StormEngineV7ForecastModel
from stormengine_dl.models.mask_aware_reconstruction import (
    configure_gradual_unfreezing,
    set_gradual_unfreezing_mode,
)

def _model() -> StormEngineV7ForecastModel:
    return StormEngineV7ForecastModel(
        5, 5, include_age=True, point_hidden=8, latent_channels=8,
        height=5, width=7, processor_layers=3, kernel_size=3,
        static_channels=2, point_static_channels=2,
    )


def _step(model: StormEngineV7ForecastModel, phase: str) -> dict[str, dict[str, torch.Tensor]]:
    before = {
        module: {
            name: value.detach().clone()
            for name, value in getattr(model, module).state_dict().items()
        }
        for module in ("encoder", "processor", "decoder")
    }
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-3
    )
    set_gradual_unfreezing_mode(model, phase, True)
    prediction = model(
        torch.randn(2, 4, 6, 5), torch.rand(2, 6, 2),
        torch.ones(2, 4, 6, 5, dtype=torch.bool), forecast_steps=3,
        observation_age=torch.zeros(2, 4, 6, 5),
        static_fields=torch.rand(2, 2, 5, 7),
        point_static=torch.rand(2, 6, 2),
    )
    optimizer.zero_grad(set_to_none=True)
    prediction.square().mean().backward()
    optimizer.step()
    return before


def _changed(
    model: StormEngineV7ForecastModel,
    before: dict[str, dict[str, torch.Tensor]],
    module: str,
) -> bool:
    return any(
        not torch.equal(value, before[module][name])
        for name, value in getattr(model, module).state_dict().items()
    )


def test_stage3a_freezes_encoder_and_updates_processor_decoder() -> None:
    model = _model()
    names = configure_gradual_unfreezing(model, "stage3a")
    assert names and all(name.startswith(("processor.", "decoder.")) for name in names)
    set_gradual_unfreezing_mode(model, "stage3a", True)
    assert not model.encoder.training
    assert model.processor.training and model.decoder.training
    before = _step(model, "stage3a")
    assert not _changed(model, before, "encoder")
    assert _changed(model, before, "processor")
    assert _changed(model, before, "decoder")


def test_stage3b_updates_all_modules() -> None:
    model = _model()
    names = configure_gradual_unfreezing(model, "stage3b")
    assert names and any(name.startswith("encoder.") for name in names)
    before = _step(model, "stage3b")
    assert _changed(model, before, "encoder")
    assert _changed(model, before, "processor")
    assert _changed(model, before, "decoder")
