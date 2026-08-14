# Frozen V7-B baseline

V7-B extends the deployment-compatible V7-A baseline with 151 model-derived
Open-Meteo marine support points. Its input contains 239 physical DPC/MeteoHub
stations plus 151 marine points, a 12-hour history, and variable-level
values/masks/ages for `u10,v10,i10fg,t2m,tp`. It predicts
`msl,u10,v10,t2m,tp` on the Adriatic 31x33 grid for leads +1 through +6 hours.

## Training and selection

- ERA5 2010-2015: training.
- ERA5 2016: validation and model selection.
- ERA5 2017: frozen independent test.
- Early stopping completed after epoch 51.
- Best checkpoint: epoch 39, validation loss `0.3284957390`.
- V7-A best validation loss was `0.3712803712`; V7-B was 11.52% lower.
- Frozen implementation commit: `1edc03720d8a17691df0af8aeaf396e48943d553`.
- Local `best.pt`: 6,314,452 bytes, SHA-256
  `2149960f7c71ebb88324aee033ed11fe903f2c34b4064eafdaa5ee95c677f991`.

The checkpoint, ERA5 cache, NetCDF data, and raw external inputs remain local
and are intentionally excluded from ordinary Git. Their required locations and
hashes are recorded in `frozen_manifest.json`.

## 2017 independent test

All 8,743 test windows completed. Clean-input aggregate full-domain RMSE:

| variable | V7-B | V7-A | dense persistence | sparse IDW / climatology |
|---|---:|---:|---:|---:|
| msl | 4.1848 | 4.1725 | 1.2319 | 6.5455 |
| u10 | 1.5615 | 1.7207 | 1.5158 | 2.3259 |
| v10 | 1.4310 | 1.5797 | 1.4563 | 2.0932 |
| t2m | 1.9892 | 2.3250 | 2.6389 | 4.6828 |
| tp | 0.3333 | 0.3486 | 0.3935 | 0.3744 |

Relative to V7-A, V7-B improves full-domain RMSE by about 9.3% for `u10`,
9.4% for `v10`, 14.4% for `t2m`, and 4.4% for `tp`; `msl` is essentially flat
and 0.3% worse. V7-B outperforms the deployment-comparable sparse baseline on
all five targets. Dense persistence uses the last complete ERA5 grid and is a
stronger-information reference, not a deployment-equivalent input.

Three fixed missing-input seeds (42, 123, and 2026) produced very similar
results to clean input, demonstrating stable behavior under the configured
missingness, outage, and observation-delay perturbations.

## Real combined-input replay

The 2026-08-01 through 2026-08-08 DPC plus Open-Meteo input produced 152/152
forecast windows, covering analysis times 2026-08-01 11:00 through
2026-08-07 18:00. All outputs were finite and no window failed.

- Stations: 239 physical plus 151 marine, 390 total.
- Mean physical variable-cell availability: 31.52%.
- Physical availability range by 12-hour window: 30.29% to 31.90%.
- Marine variable-cell availability: 100%.

This replay proves runtime compatibility and numerical stability of the real
combined input path. It is not a 2026 full-grid accuracy claim because matching
ERA5 2026 target grids were not available for this run.

## Reproducible record

The directory includes smoke, benchmark, pilot, full training, four 2017
evaluation JSON files, the combined replay summary, and the executed V7-B
notebook. See `frozen_manifest.json` for source paths and SHA-256 checksums.
