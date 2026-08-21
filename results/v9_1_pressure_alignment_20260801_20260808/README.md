# V9.1 MSL input alignment, 2026-08-01 to 2026-08-08

## Purpose

This package records the pressure-input preparation that follows the frozen V9-A
real-input diagnosis. It is data-interface validation for a future V9.1/V10 model,
not a new training result and not a change to the frozen V9-A checkpoint.

The deployment contract combines two sources:

- 239 physical DPC/MeteoHub coordinates, with sparse station pressure reduced to
  mean sea level and represented by a variable-level mask;
- 151 marine Open-Meteo ICON-2I support coordinates using the API field
  `pressure_msl` directly.

Together they form one pressure channel ordered as 239 physical plus 151 marine
points. Over the 169-hour snapshot, 13 physical stations provide pressure and all
151 marine points provide model-derived MSL, giving 42.05% valid cells across the
combined 390-point pressure tensor.

## DPC station-pressure reduction

Raw MeteoHub `B10004` is station pressure and must not be labelled as ERA5 `msl`.
Where station elevation is known, it is reduced with the low-elevation hypsometric
relationship:

```text
p_msl = p_station * exp(g * z / (R_d * T_v_mean_layer))
```

Temperature and humidity support uses observations at or before the valid hour only.
A trailing window of at most 12 hours is allowed; no future observation is used. A
15 C fallback is allowed only when temperature is absent and elevation is at most
10 m. Raw pressure, correction method, mask, age, elevation and provenance remain
available in the local processed product.

## Same-time ERA5T alignment

Metrics compare each source with ERA5T sampled at the same coordinate and valid
time. They diagnose scale and interface alignment; ERA5T is a gridded reference,
not error-free station truth.

| source | valid cells | bias | MAE | RMSE | correlation |
|---|---:|---:|---:|---:|---:|
| raw DPC station pressure | 2,193 | -3.703 hPa | 4.086 hPa | 9.191 hPa | -0.028 |
| corrected DPC MSL | 2,193 | -0.509 hPa | 1.145 hPa | 1.677 hPa | 0.606 |
| Open-Meteo marine `pressure_msl` | 25,519 | +0.263 hPa | 0.409 hPa | 0.510 hPa | 0.957 |

The DPC correction removes the dominant elevation mismatch. Individual DPC station
time-series correlations are generally 0.88--0.98; the lower pooled correlation also
contains station offsets and point-versus-grid representativeness. Ancona Regione
retains an approximately -4.83 hPa bias and requires station metadata or sensor
review. A permanent correction must not be fitted from this single week.

## Training implication

Future ERA5 training should expose `msl` at the same 390 coordinates while simulating
the actual source roles: sparse structural barometer availability on physical
stations and high availability on marine model points. Every value still requires a
per-variable mask and observation age. Adding MSL expands the encoder input width,
so the frozen V9-A checkpoint cannot be loaded as if its encoder were unchanged.

## Reproducibility and storage

The local processed NPZ products are intentionally excluded from Git because they
are derived data. Their relative paths, byte sizes and SHA-256 checksums are recorded
in `source_alignment.json` and the source manifests under `data/manifests/`. The
conversion code and tests are versioned so another machine with the raw inputs can
rebuild and verify them.
