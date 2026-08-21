# V9 frozen real-input diagnosis, 2026-08-01 to 2026-08-08

This is a post-freeze diagnosis, not training or model selection. The frozen V9-A and
V7-B checkpoints were evaluated on the same 152 windows with 12-hour histories and
six-hour ERA5T targets.

## Input coverage

- DPC physical stations: 239, overall variable-cell validity 31.49%.
- Open-Meteo marine support points: 151, validity 100% in the stitched product.
- DPC validity by variable: u10 16.91%, v10 16.91%, gust 8.77%, t2m 54.90%, tp 59.95%.
- All 152 V9 and V7-B operational forecasts were finite.

## Sea RMSE skill of frozen V9-A over frozen V7-B

Positive values favour V9-A.

| input | msl | u10 | v10 | t2m | tp | five-variable mean |
|---|---:|---:|---:|---:|---:|---:|
| real DPC + Open-Meteo | -4.87% | +1.23% | -1.29% | +6.93% | -0.44% | +0.31% |
| complete ERA5T point proxy | +8.12% | +1.59% | -1.85% | +15.99% | +0.62% | +4.89% |

With real input, V9-A wins 6 of 12 sea u10/v10 component-by-lead comparisons. This
does not reproduce the 12/12 wind result from the locked 2025 ERA5-proxy final test.

## Operational-input penalty for V9-A

The fraction below is the excess MSE caused by replacing complete ERA5T point inputs
with the real operational input system while holding the model, week, coordinates,
targets, and windows fixed.

| sea variable | excess fraction of operational MSE |
|---|---:|
| msl | 36.2% |
| u10 | 26.8% |
| v10 | 30.0% |
| t2m | -5.1% |
| tp | 0.4% |

The controlled counterfactual therefore identifies source mismatch, structural
missingness, and point/grid representativeness as the main loss of the V9 advantage,
especially for msl and wind. Real t2m happens to outperform the complete ERA5T point
proxy in this week and should not be interpreted as a general causal benefit.

## Interpretation

V9-A is operationally compatible: it accepts the real 390-point input contract and
runs every window without NaN. It remains slightly better than V7-B on the unweighted
mean sea skill, but its frozen replacement advantage does not transfer uniformly to
the real input domain. The next development target should be source/domain adaptation
and realistic variable-specific missingness, not additional epochs on the frozen V9-A.

This seven-day period is too short to estimate seasonal or extreme-event performance.
ERA5T is a gridded reference rather than error-free in-situ truth. The result must be
used as a diagnostic for a future V9.1/V10 design, not to retune or overwrite V9-A.
