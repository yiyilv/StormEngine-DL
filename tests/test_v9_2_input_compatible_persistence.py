from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_v9_2_final16_input_compatible_persistence import (  # noqa: E402
    reconstruct_current_without_processor,
    rmse_skill,
    validate_timeline,
)
from stormengine_dl.baselines import dense_grid_persistence  # noqa: E402
from stormengine_dl.models.v9 import StormEngineV9ForecastModel  # noqa: E402


def make_model() -> StormEngineV9ForecastModel:
    return StormEngineV9ForecastModel(
        6,
        5,
        6,
        temporal_mode="autoregressive",
        output_mode="field",
        include_age=True,
        point_hidden=8,
        latent_channels=8,
        height=5,
        width=7,
        processor_layers=2,
        kernel_size=3,
        static_channels=2,
        point_static_channels=2,
    ).eval()


def make_batch() -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    generator = torch.Generator().manual_seed(42)
    batch = {
        "point_values": torch.randn(2, 12, 9, 6, generator=generator),
        "point_coords": torch.rand(2, 9, 2, generator=generator),
        "value_mask": torch.rand(2, 12, 9, 6, generator=generator) > 0.2,
        "observation_age": torch.rand(2, 12, 9, 6, generator=generator),
        "point_static": torch.rand(2, 9, 2, generator=generator),
        "target": torch.empty(2, 6, 5, 5, 7),
    }
    static = torch.rand(1, 2, 5, 7, generator=generator)
    return batch, static


def test_processor_free_reconstruction_matches_validated_model_interface() -> None:
    model = make_model()
    batch, static = make_batch()
    calls = [0]

    def count_calls(_module: object, _inputs: object, _output: object) -> None:
        calls[0] += 1

    hook = model.processor.register_forward_hook(count_calls)
    with torch.no_grad():
        isolated = reconstruct_current_without_processor(model, batch, static)
    hook.remove()
    assert calls == [0]
    with torch.no_grad():
        _, validated = model.forward_with_reconstruction(
            batch["point_values"],
            batch["point_coords"],
            batch["value_mask"],
            6,
            observation_age=batch["observation_age"],
            static_fields=static.expand(2, -1, -1, -1),
            point_static=batch["point_static"],
        )
    assert isolated.shape == (2, 5, 5, 7)
    assert torch.equal(isolated, validated)


def test_reconstruction_persistence_repeats_current_field_exactly() -> None:
    current = torch.randn(2, 5, 5, 7)
    prediction = dense_grid_persistence(current, 6)
    assert prediction.shape == (2, 6, 5, 5, 7)
    for lead in range(6):
        assert torch.equal(prediction[:, lead], current)


def test_skill_uses_one_minus_rmse_ratio_for_all_regions_and_leads() -> None:
    targets = ("msl", "u10", "v10", "t2m", "tp")

    def regions(value: float) -> dict[str, object]:
        return {
            region: {name: {"mae": value, "rmse": value} for name in targets}
            for region in ("full", "land", "sea")
        }

    candidate = {
        "aggregate": regions(2.0),
        "by_lead_hour": {str(lead): regions(2.0) for lead in range(1, 7)},
    }
    baseline = {
        "aggregate": regions(4.0),
        "by_lead_hour": {str(lead): regions(4.0) for lead in range(1, 7)},
    }
    skill = rmse_skill(candidate, baseline)
    assert skill["aggregate"]["full"]["msl"]["skill"] == 0.5
    assert skill["aggregate"]["land"]["tp"]["skill"] == 0.5
    assert skill["by_lead_hour"]["6"]["sea"]["v10"]["skill"] == 0.5


def test_timeline_enforces_hourly_plus_one_to_plus_six_alignment() -> None:
    times = np.arange(
        np.datetime64("2026-08-01T00"),
        np.datetime64("2026-08-02T00"),
        np.timedelta64(1, "h"),
    ).astype("datetime64[ns]")
    result = validate_timeline(times, np.arange(3), 12, 6)
    assert result["first_forecast_origin"].startswith("2026-08-01T11:00")
    assert result["first_target_time"].startswith("2026-08-01T12:00")
    broken = times.copy()
    broken[8] = broken[7]
    try:
        validate_timeline(broken, np.arange(3), 12, 6)
    except ValueError as exc:
        assert "strictly hourly" in str(exc)
    else:
        raise AssertionError("Expected non-hourly input timeline to be rejected")
