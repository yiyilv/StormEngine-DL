"""Transparent low-elevation reduction of station pressure to mean sea level."""

from __future__ import annotations

import math


GRAVITY_M_S2 = 9.80665
DRY_AIR_GAS_CONSTANT_J_KG_K = 287.05
LAPSE_RATE_K_M = 0.0065
EPSILON = 0.622


def virtual_temperature_k(
    temperature_c: float,
    station_pressure_hpa: float,
    relative_humidity_percent: float | None = None,
) -> float:
    """Return virtual temperature; fall back to dry temperature without RH."""
    temperature_k = float(temperature_c) + 273.15
    if relative_humidity_percent is None:
        return temperature_k
    humidity = min(100.0, max(0.0, float(relative_humidity_percent)))
    saturation_hpa = 6.112 * math.exp(
        17.67 * float(temperature_c) / (float(temperature_c) + 243.5)
    )
    vapor_hpa = humidity / 100.0 * saturation_hpa
    mixing_ratio = EPSILON * vapor_hpa / max(float(station_pressure_hpa) - vapor_hpa, 1.0)
    specific_humidity = mixing_ratio / (1.0 + mixing_ratio)
    return temperature_k * (1.0 + 0.61 * specific_humidity)


def station_pressure_to_msl_hpa(
    station_pressure_hpa: float,
    elevation_m: float,
    temperature_c: float,
    relative_humidity_percent: float | None = None,
) -> float:
    """Reduce station pressure to MSL with a low-elevation hypsometric model.

    The mean layer virtual temperature is approximated from observed station
    temperature plus half the standard 6.5 K/km lapse-rate correction down to
    sea level. This is intended for the project's coastal stations (<=268 m),
    not high-elevation synoptic pressure reduction.
    """
    pressure = float(station_pressure_hpa)
    elevation = float(elevation_m)
    if not math.isfinite(pressure) or not 800.0 <= pressure <= 1100.0:
        raise ValueError("station pressure must be finite and within 800--1100 hPa")
    if not math.isfinite(elevation) or not -20.0 <= elevation <= 500.0:
        raise ValueError("elevation must be finite and within -20--500 m")
    station_virtual = virtual_temperature_k(
        temperature_c, pressure, relative_humidity_percent
    )
    mean_layer_virtual = station_virtual + LAPSE_RATE_K_M * elevation / 2.0
    exponent = GRAVITY_M_S2 * elevation / (
        DRY_AIR_GAS_CONSTANT_J_KG_K * mean_layer_virtual
    )
    return pressure * math.exp(exponent)
