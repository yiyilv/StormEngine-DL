# Metric review — V8 Stage 2 Processor-only

## Primary two-seed comparison

| Candidate | Seed 42 | Seed 43 | Mean | Sample std |
|---|---:|---:|---:|---:|
| PH64-LAT96 | 0.3466828829 | 0.3470679740 | 0.3468754285 | 0.0002723006 |
| PH96-LAT96 | 0.3480467535 | 0.3466895386 | 0.3473681461 | 0.0009596958 |

The relative mean gap is only 0.142%. PH64-LAT96 has the lower mean primary loss and substantially lower seed-to-seed variability. Under the declared rule (difference below 1% => prefer the smaller spatial MLP), **PH64-LAT96 remains the provisional Stage 2 spatial recommendation**.

## Per-variable and region review

The two candidates are close on most aggregate metrics. In the sea region, PH64-LAT96 has lower `u10` RMSE (0.16%), `v10` RMSE (0.03%), and `tp` RMSE (0.31%). PH96-LAT96 has lower sea `t2m` RMSE (1.38%) and `tp` MAE (1.84%), as well as small advantages in several MAE metrics. The mixed result does not justify overriding the predeclared primary metric and complexity rule.

This is a validation-stage selection only. Encoder and Decoder were frozen, only the random-initialized L3K3 Processor was trained, and ERA5 2017 remained locked and unread. Stage 3 joint fine-tuning is required before any end-to-end model claim.

## Validation

- Generated result manifest entries match file sizes and SHA-256 hashes.
- All result JSON files parse successfully.
- Stage 2 Notebook has zero saved Python error outputs.
- Stage 2 and runner tests pass.
