from __future__ import annotations

import numpy as np
import pytest

from stormengine_dl.data.pressure_reduction import station_pressure_to_msl_hpa


def test_zero_elevation_preserves_station_pressure() -> None:
    assert station_pressure_to_msl_hpa(1008.4, 0.0, 25.0, 70.0) == pytest.approx(
        1008.4
    )


def test_pressure_reduction_increases_pressure_at_positive_elevation() -> None:
    corrected = station_pressure_to_msl_hpa(980.0, 268.0, 25.0, 60.0)
    assert 1009.0 < corrected < 1013.0


def test_pressure_reduction_rejects_implausible_pressure() -> None:
    with pytest.raises(ValueError, match="station pressure"):
        station_pressure_to_msl_hpa(200.0, 10.0, 20.0)


def test_build_corrected_msl_never_uses_future_temperature() -> None:
    from scripts.build_dpc_msl import build_corrected_msl

    variables = ("station_pressure_hpa", "t2m", "relative_humidity")
    values = np.zeros((2, 1, 3), np.float32)
    masks = np.zeros_like(values, bool)
    values[:, 0, 0] = 1000.0
    masks[:, 0, 0] = True
    values[1, 0, 1] = 40.0
    masks[1, 0, 1] = True
    result = build_corrected_msl(values, masks, variables, np.asarray([100.0]))
    assert not result["value_mask"][0, 0]
    assert result["value_mask"][1, 0]
