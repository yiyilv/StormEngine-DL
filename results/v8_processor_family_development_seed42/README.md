# V8 dense Processor family development — seed 42

This directory records the first V8 dense temporal Processor family comparison. Both models receive 12 hours of dense normalized ERA5 grids and predict the next 6 hours directly. The run used commit `a82127135cd92ad60de517c4c0d1cb5ce573a0d2` on branch `v8-model-development`.

## Scientific scope

- Training: ERA5 2013-2015
- Validation and early stopping: ERA5 2016
- ERA5 2017: not read
- August 2026 operational data: not read
- Sparse stations, DPC, Open-Meteo, Encoder, and Decoder: not read
- History: 12 hours
- Forecast horizon: 6 hours
- Development window stride: 3 hours
- Common seed: 42
- Maximum: 120 epochs
- Early-stopping patience: 10

This experiment isolates temporal forecasting skill. It is not an end-to-end operational model evaluation.

## Ranking

| Rank | Family | Parameters | Completed epochs | Best epoch | Best validation loss |
|---:|---|---:|---:|---:|---:|
| 1 | ConvGRU | 996,965 | 56 | 46 | 0.1350536781459884 |
| 2 | Factorized ViT | 1,995,173 | 45 | 35 | 0.13762686965997264 |

ConvGRU wins the normalized sea-weighted 2016 validation objective by `1.8697%` while using approximately half as many parameters. It also has lower aggregate full-domain RMSE for u10, v10, t2m, and tp. The Factorized ViT has lower MSL error, so the variable-level result is not completely uniform.

This is a single-seed family screen. Both families must be repeated with a second seed before choosing the Processor family. If the ranking is unstable or nearly tied, the documented rule is to prefer the smaller ConvGRU and test both at the end-to-end compatibility gate.

## Storage and reproducibility

The development cache and all `best.pt`/`last.pt` files remain local and are intentionally excluded from Git. Their sizes and SHA-256 hashes are recorded in `manifest.json`.

## Contents

- `convgru/`: complete history and converged summary.
- `factorized_vit/`: complete history and converged summary.
- `comparison.json`: aggregate and +1h through +6h full/land/sea metrics.
- `manifest.json`: provenance and local checkpoint hashes.
- `StormEngine_V8_Processor_Family_Development_executed.ipynb`: compact executed run record.
