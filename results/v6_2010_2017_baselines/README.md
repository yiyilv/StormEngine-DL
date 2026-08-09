# StormEngine V6 persistence baselines

This directory compares the frozen V6 checkpoint against two deterministic
references on the same 8,743 windows from the held-out 2017 test year. All RMSE
values use the same physical-unit conversion and full/land/sea masks as the V6
evaluation.

## Baseline definitions

- **Dense-grid persistence** repeats the last complete ERA5 target grid for all
  six lead hours. It is a deliberately strong reference and is not input-fair:
  V6 receives 390 sparse coordinates, not the complete current grid.
- **Sparse IDW persistence** uses only the final-hour values at the same 390
  coordinates available to V6. The four shared variables are interpolated with
  metric-aware eight-neighbor inverse-distance weighting (power 2) and held
  constant. Because `tp` is not a V6 input variable, its fair baseline is zero
  hourly precipitation.

Skill is `1 - V6_RMSE / baseline_RMSE`; positive values mean V6 is better.

## Aggregate full-domain RMSE

| Method | msl | u10 | v10 | t2m | tp |
|---|---:|---:|---:|---:|---:|
| V6 | 1.1960 | 1.4798 | 1.3767 | 1.8258 | 0.3469 |
| Dense persistence | 1.2319 | 1.5158 | 1.4563 | 2.6389 | 0.3935 |
| Sparse IDW persistence | 2.3445 | 2.5940 | 2.2140 | 4.4081 | 0.4360 |

## Aggregate V6 skill

| Reference / region | msl | u10 | v10 | t2m | tp |
|---|---:|---:|---:|---:|---:|
| Dense / full | +0.029 | +0.024 | +0.055 | +0.308 | +0.118 |
| Dense / land | -0.003 | -0.024 | -0.003 | +0.351 | +0.130 |
| Dense / sea | +0.088 | +0.042 | +0.082 | -0.316 | +0.096 |
| Sparse IDW / full | +0.490 | +0.430 | +0.378 | +0.586 | +0.204 |
| Sparse IDW / land | +0.530 | +0.551 | +0.476 | +0.582 | +0.219 |
| Sparse IDW / sea | +0.352 | +0.333 | +0.296 | +0.606 | +0.176 |

V6 beats the input-fair sparse baseline for every variable and region. Against
the stronger dense-grid reference, it is better in aggregate for all
full-domain variables, but the lead-hour detail matters: over sea, V6 is worse
at +1 h and +2 h for pressure and wind, then becomes better at longer leads.
Sea temperature remains worse than dense persistence through +5 h and is only
approximately equal at +6 h. This identifies short-lead marine reconstruction,
not the existence of forecast skill, as the next diagnostic target.

The complete aggregate, lead-hour, regional, and skill values are stored in
`baseline_metrics.json`.
