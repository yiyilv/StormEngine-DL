# V9.1 pressure input contract

## Purpose

Frozen V9-A predicts ERA5 `msl` without any pressure input. V9.1 restores a
deployment-compatible pressure channel without relabelling raw DPC station pressure.
It combines sparse corrected physical observations with model-derived marine MSL.

## Source-specific adapters

### Open-Meteo marine support

The Historical Forecast API field `pressure_msl` is an instantaneous sea-level
pressure in hPa. The existing ICON-2I raw responses already contain it. The aligned
output is:

```text
data_external/open_meteo/processed/20260801_20260808/
└── icon2i_hourly_marine_pressure.npz
```

It contains 169 hours, 151 fixed marine coordinates and one `msl` channel. All
25,519 cells are valid. `surface_pressure` is deliberately excluded because a
returned model cell can have non-zero terrain elevation and surface pressure is not
interchangeable with MSL.

### DPC physical observations

MeteoHub `B10004` remains `station_pressure_hpa`. It is reduced to sea level only
when station elevation is known. The low-elevation hypsometric reduction is:

```text
p_msl = p_station * exp(g * z / (R_d * T_v_mean_layer))
```

`T_v_mean_layer` uses observed temperature from the current and preceding hours,
up to a trailing 12-hour window. Relative humidity is used to estimate virtual
temperature when available. The mean layer temperature adds half of a standard
6.5 K/km lapse-rate correction between station elevation and sea level.

No observation after the target hour is used. If temperature is entirely absent,
a 15 C standard-temperature fallback is permitted only for elevations at or below
10 m. In this snapshot that fallback applies to one station at 2 m elevation. Every
output retains the raw station pressure, elevation, correction, method code,
temperature/RH support counts, source time, mask and age.

The aligned output is:

```text
data_external/meteohub/processed/20260801_20260808/
└── official_hourly_physical_msl.npz
```

Only 13 of 239 physical stations provide pressure in this week. Non-pressure stations
remain in their fixed positions with `value_mask=0`.

## Same-time ERA5T diagnostics

These comparisons use ERA5T sampled at the same valid time and requested coordinate.
ERA5T is a gridded diagnostic reference, not error-free station truth.

| source | bias | MAE | RMSE | correlation |
|---|---:|---:|---:|---:|
| raw DPC station pressure | -3.703 hPa | 4.086 hPa | 9.191 hPa | -0.028 |
| corrected DPC MSL | -0.509 hPa | 1.145 hPa | 1.677 hPa | 0.606 |
| Open-Meteo marine `pressure_msl` | +0.263 hPa | 0.409 hPa | 0.510 hPa | 0.957 |

The DPC correction removes the dominant elevation effect. Per-station temporal
correlations are generally much higher than the pooled correlation because the
pooled statistic also contains station-specific representativeness and instrument
offsets. This single week must not be used to fit permanent per-station bias values.

## Unified V9.1 observation batch

The six input variables should be ordered as:

```text
msl, u10, v10, i10fg, t2m, tp
```

The combined pressure channel is constructed as:

```text
first 239 coordinates  = corrected DPC MSL, sparse variable-level mask
last 151 coordinates   = Open-Meteo pressure_msl, model-derived source type
```

Every channel continues to provide `value`, `value_mask` and `observation_age`.
Pressure additionally retains source and correction-quality metadata. A missing DPC
pressure is zero-filled only after normalization and always has `value_mask=0`.

For ERA5 training, `msl` is sampled at all 390 coordinates, but availability must be
simulated according to source role: sparse structural pressure capability on physical
stations and high availability on marine model points. Training must not pretend that
all 239 DPC stations carry barometers.

Adding the sixth variable changes the encoder input width. Frozen V9-A remains the
reference checkpoint; compatible ConvGRU/decoder tensors may be warm-started, while
the expanded encoder input weights require controlled initialization. V9-A itself is
never overwritten.

## Acceptance conditions before training

- exact 169-hour time equality between physical and marine tensors;
- exact 239 + 151 fixed registry order;
- pressure in hPa and plausible 850--1100 hPa range;
- raw station pressure is preserved separately;
- no future observation contributes to a corrected value;
- non-pressure physical stations use a variable-level mask;
- source type distinguishes corrected DPC from model-derived Open-Meteo;
- the one-week diagnostics determine plausible simulation ranges, not test-set bias
  corrections.
