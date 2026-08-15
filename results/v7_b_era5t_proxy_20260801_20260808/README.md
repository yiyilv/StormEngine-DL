# V7-B ERA5T proxy-input counterfactual, 2026-08-01 to 2026-08-08

This controlled counterfactual estimates the combined operational input-system
penalty in the frozen V7-B model.

```text
Published operation: DPC + Open-Meteo -> frozen V7-B -> future ERA5T grid
Counterfactual:      ERA5T at 390 points -> same V7-B -> same future grid
```

Both runs use the same checkpoint, 390 coordinates, 12-hour histories, +1
through +6-hour targets, 152 windows, target grids, land-sea mask, and metrics.
The counterfactual uses complete ERA5T point values with valid masks and zero
observation age, approximating idealized training-style input.

## Full-domain aggregate result

| variable | operational RMSE | ERA5T-proxy RMSE | excess share of operational MSE |
|---|---:|---:|---:|
| msl | 1.9541 | 1.7274 | 21.86% |
| u10 | 1.3500 | 1.1904 | 22.24% |
| v10 | 1.3988 | 1.1979 | 26.66% |
| t2m | 2.1166 | 1.8559 | 23.11% |
| tp | 0.1194 | 0.1181 | 2.26% |

The excess quantity is:

```text
operational real-input MSE - ERA5T-proxy input MSE
```

Positive excess MSE indicates a combined deployment-input penalty. It includes
source mismatch, missingness, observation age, and point-versus-grid
representativeness. It is not an additive causal decomposition of DPC and
Open-Meteo separately.

## Regional interpretation

Over sea, the operational-input excess share of MSE is 16.89% for `msl`,
26.25% for `u10`, and 30.78% for `v10`. This identifies input-system alignment
as a substantial part of the remaining marine wind error, while most error
still remains in the model even under idealized proxy input.

Sea `t2m` and `tp` are exceptions: operational inputs perform slightly better
than the ERA5T proxy (`t2m` operational RMSE 1.0368 versus proxy 1.1776; `tp`
0.02399 versus 0.02420). The real marine pathway therefore contains useful
signal and should not be characterized as uniformly inferior to ERA5 input.

Precipitation changes little overall, consistent with the separate alignment
audit showing that sparsity and reporting-window semantics require dedicated
analysis.

## Artifacts

- `metrics.json`: complete operational/proxy metrics and MSE penalties for
  full/land/sea and every lead hour.
- `manifest.json`: exact code, checkpoint, operational metrics, and output
  checksums.

The model checkpoint and raw input files remain local and are not committed to
ordinary Git.
