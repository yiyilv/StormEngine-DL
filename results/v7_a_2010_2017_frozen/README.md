# Frozen V7-A baseline

V7-A is the first deployment-compatible physical-station baseline. It uses 239
DPC/MeteoHub stations, a 12-hour history, variable-level values/masks/ages for
`u10,v10,i10fg,t2m,tp`, and predicts `msl,u10,v10,t2m,tp` on the Adriatic
31x33 grid for leads +1 through +6 hours. It contains no Open-Meteo input and no
virtual sea points.

## Training and selection

- ERA5 2010-2015: training.
- ERA5 2016: validation and model selection.
- ERA5 2017: frozen independent test.
- Early stopping completed after epoch 31.
- Best checkpoint: epoch 19, validation loss `0.3712803712`.
- Frozen code commit: `c14cf004a4bf20765c3922919e5c88c942509980`.
- Local `best.pt`: 6,313,492 bytes, SHA-256
  `07a7bd4c78fa271a83b4315b2640ee833c375dc2a57d093e4fa87a8a68551e95`.

The checkpoint is intentionally excluded from ordinary Git. See
`frozen_manifest.json` for hashes of the local checkpoint, configuration, and
all full evaluation files.

## 2017 full-grid result

Clean-input aggregate full-domain RMSE:

| variable | V7-A | frozen V6 | dense persistence | sparse IDW / climatology |
|---|---:|---:|---:|---:|
| msl | 4.1725 | 1.1960 | 1.2319 | 6.5455 |
| u10 | 1.7207 | 1.4798 | 1.5158 | 2.5147 |
| v10 | 1.5797 | 1.3767 | 1.4563 | 2.1856 |
| t2m | 2.3250 | 1.8258 | 2.6389 | 4.7160 |
| tp | 0.3486 | 0.3469 | 0.3935 | 0.4002 |

V7-A is weaker than V6 on clean full-grid inputs, which is consistent with
removing 151 virtual marine support coordinates. It remains better than the
input-fair sparse baseline on all targets and is stable across three randomized
missing-input seeds. Dense persistence uses the last complete ERA5 grid and is
therefore a stronger-information reference rather than a deployment-equivalent
input.

## Real DPC operation and observation evaluation

- 152/152 real DPC replay windows completed.
- All forecast outputs were finite.
- Mean input variable-cell availability was 31.52%.
- Stations with any variable ranged from 119 to 164 per input hour.

Future DPC observations were then used as station-space targets. Forecast grids
were sampled at the physical station coordinates and scored only where the
future variable-level DPC mask was valid.

| variable | observed cells | MAE | RMSE | skill vs station persistence |
|---|---:|---:|---:|---:|
| u10 | 36,825 | 1.1824 m/s | 1.6585 m/s | +15.02% |
| v10 | 36,825 | 1.1375 m/s | 1.4869 m/s | +15.99% |
| t2m | 119,598 | 2.5903 degC | 3.2457 degC | +16.63% |
| tp | 131,105 | 0.0222 mm | 0.2197 mm | +26.06% |

This demonstrates deployment compatibility and positive short-range skill over
station persistence during the available week. It does not establish full-grid
2026 accuracy or annual performance. DPC pressure is not directly comparable to
ERA5 MSL, and V7-A does not forecast gust, so those variables are excluded from
the external observation evaluation.

