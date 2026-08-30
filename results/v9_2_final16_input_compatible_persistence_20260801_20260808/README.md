# V9.2 Final16 input-compatible persistence evaluation

This is a supplemental evaluation of frozen checkpoints. No model was retrained, no parameter or threshold was changed, and no 2026 result was used for tuning.

## Fair persistence baseline

For each of the same 152 operational windows, the baseline receives the same 12 hours of DPC/MeteoHub physical observations and Open-Meteo marine support, including the same coordinate registry, variable order, normalization, per-variable masks, observation ages, station static features, and source type. The frozen Final16 encoder produces the historical latent sequence. Its last latent state is decoded by the checkpoint's existing, validated `reconstruction_decoder`; the Processor is not called. The reconstructed current 31 x 33 field is copied unchanged to leads +1 through +6.

Dense ERA5T persistence remains a useful stronger-information diagnostic because it receives the complete ERA5T grid at the forecast origin. It is not input-compatible with the operational model. The reconstruction-persistence baseline is therefore the fair test of whether the Processor adds value beyond holding the model-reconstructed current state fixed.

## Scope

The period is 2026-08-01 through 2026-08-08, with 152 windows, 12 historical hours, and +1...+6 h forecasts verified against the same ERA5T 31 x 33 targets as the published operational evaluation. No observed case reached the frozen 30 mm/6 h heavy-rain or 15 m/s strong-wind threshold, so this experiment evaluates ordinary continuous fields only; it cannot support an extreme-event skill claim.

## Aggregate RMSE skill against the fair baseline

Positive skill means V9.2 Final16 has lower RMSE than input-compatible reconstruction persistence.

| region | MSL | u10 | v10 | T2m | TP |
|---|---:|---:|---:|---:|---:|
| full | 64.31% | 37.13% | 21.66% | 86.74% | 44.83% |
| land | 60.18% | 40.59% | 19.18% | 85.36% | 24.69% |
| sea | 70.73% | 34.14% | 23.09% | 92.00% | 87.74% |

## Integrity

- Smoke test passed: `True`
- Full window count: `152`
- All outputs finite: `True`
- +1...+6 h alignment passed: `+1 through +6 hours from the last historical input`
- Original V9.2, V9.1, and dense-persistence metrics reproduced exactly: `True`
- TP semantics are unchanged: hourly ERA5T TP is verified at each lead; six-hour event   diagnostics, when referenced, use `sum(max(tp_hourly_mm, 0))` over +1...+6.

The JSON and CSV files contain aggregate and lead-specific MAE, RMSE, and RMSE skill for full, land, and sea domains.
