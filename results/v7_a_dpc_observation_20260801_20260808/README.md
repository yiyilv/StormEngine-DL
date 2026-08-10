# V7-A external DPC observation evaluation

The frozen V7-A `best.pt` was evaluated over 152 forecast origins. Each origin
uses only the preceding 12 DPC hours. Forecast grids were bilinearly sampled at
the 239 physical station coordinates and compared with observed DPC values at
leads +1 through +6 hours using the future variable-level masks.

This is an external station-space evaluation, not a full-grid 2026 evaluation.
`msl` is excluded because DPC station pressure is not directly comparable to
ERA5 mean-sea-level pressure. Gust is excluded because V7-A does not forecast
gust. Hourly TP is aligned by valid time and is never forward-filled.

| variable | observed cells | MAE | RMSE | skill vs station persistence |
|---|---:|---:|---:|---:|
| u10 | 36,825 | 1.1824 m/s | 1.6585 m/s | 0.1502 |
| v10 | 36,825 | 1.1375 m/s | 1.4869 m/s | 0.1599 |
| t2m | 119,598 | 2.5903 °C | 3.2457 °C | 0.1663 |
| tp | 131,105 | 0.0222 mm | 0.2197 mm | 0.2606 |

Skill is `1 - V7-A_RMSE / persistence_RMSE` on the identical comparable cells;
positive values mean V7-A is better. Detailed lead-hour metrics and per-window
diagnostics remain with the local evaluation artifacts.

