from __future__ import annotations

import torch

from stormengine_dl.models.v9 import StormEngineV9ForecastModel, warm_start_from_v7b


def make_model(temporal_mode: str, output_mode: str) -> StormEngineV9ForecastModel:
    return StormEngineV9ForecastModel(
        5,
        5,
        6,
        temporal_mode=temporal_mode,
        output_mode=output_mode,
        include_age=True,
        point_hidden=8,
        latent_channels=8,
        height=5,
        width=7,
        processor_layers=2,
        kernel_size=3,
        static_channels=2,
        point_static_channels=2,
    )


def inputs() -> tuple[torch.Tensor, ...]:
    values = torch.randn(2, 4, 9, 5)
    coords = torch.rand(2, 9, 2)
    mask = torch.rand(2, 4, 9, 5) > 0.2
    age = torch.rand(2, 4, 9, 5)
    static = torch.rand(2, 2, 5, 7)
    point_static = torch.rand(2, 9, 2)
    return values, coords, mask, age, static, point_static


def test_all_four_v9_forms_have_identical_output_contract() -> None:
    values, coords, mask, age, static, point_static = inputs()
    for temporal_mode in ("autoregressive", "direct"):
        for output_mode in ("field", "residual"):
            model = make_model(temporal_mode, output_mode)
            prediction, current = model.forward_with_reconstruction(
                values,
                coords,
                mask,
                6,
                observation_age=age,
                static_fields=static,
                point_static=point_static,
            )
            assert prediction.shape == (2, 6, 5, 5, 7)
            assert torch.isfinite(prediction).all()
            assert current is not None and current.shape == (2, 5, 5, 7)
            assert torch.isfinite(current).all()


def test_residual_prediction_is_current_plus_forecast_increment() -> None:
    model = make_model("autoregressive", "residual")
    assert model.reconstruction_decoder is not None
    for parameter in model.decoder.parameters():
        parameter.data.zero_()
    values, coords, mask, age, static, point_static = inputs()
    prediction, current = model.forward_with_reconstruction(
        values,
        coords,
        mask,
        6,
        observation_age=age,
        static_fields=static,
        point_static=point_static,
    )
    assert current is not None
    assert torch.allclose(prediction, current[:, None].expand_as(prediction))


def test_direct_horizon_rejects_changed_forecast_length() -> None:
    model = make_model("direct", "field")
    values, coords, mask, age, static, point_static = inputs()
    try:
        model(
            values,
            coords,
            mask,
            5,
            observation_age=age,
            static_fields=static,
            point_static=point_static,
        )
    except ValueError as exc:
        assert "built for 6" in str(exc)
    else:
        raise AssertionError("Expected a fixed-horizon contract error")


def test_v7b_warm_start_copies_compatible_and_residual_decoder_tensors(tmp_path) -> None:
    source = make_model("autoregressive", "field")
    for parameter in source.parameters():
        parameter.data.fill_(0.25)
    checkpoint = tmp_path / "best.pt"
    torch.save({"model_state_dict": source.state_dict()}, checkpoint)
    target = make_model("direct", "residual")
    transfer = warm_start_from_v7b(target, checkpoint)
    assert int(transfer["loaded_tensor_count"]) > 0
    assert torch.allclose(target.encoder.point_mlp[0].weight, source.encoder.point_mlp[0].weight)
    assert torch.allclose(target.decoder.network[0].weight, source.decoder.network[0].weight)
    assert target.reconstruction_decoder is not None
    assert torch.allclose(
        target.reconstruction_decoder.network[0].weight,
        source.decoder.network[0].weight,
    )
    assert torch.count_nonzero(target.decoder.network[-1].weight) == 0
    assert torch.count_nonzero(target.decoder.network[-1].bias) == 0
    assert transfer["zero_initialized_tensors"] == [
        "decoder.network.4.weight",
        "decoder.network.4.bias",
    ]
