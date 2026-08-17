# V8 Stage-1 spatial capacity development

This directory records the first converged V8 Stage-1 capacity comparison. Gaussian sigma is fixed at `0.10`; the experiment isolates point-MLP width (`point_hidden`) and gridded latent width (`latent_channels`). The run used commit `7e3dfb7cf3a339b08f1937f062b802a2d65ee46e` on branch `v8-model-development`.

## Scientific scope

- Training: ERA5 2013-2015
- Validation and early stopping: ERA5 2016
- ERA5 2017: not read
- August 2026 operational data: not read
- Common seed: 42
- Gaussian sigma: 0.10
- Maximum: 120 epochs
- Early-stopping patience: 10
- Reused immutable cache: `stormengine_dev_2013_2016`

All four configurations demonstrated validation convergence before comparison. The 64/64 baseline is the previously published converged sigma=0.10 model; three larger candidates were trained from scratch.

## Ranking

| Rank | point_hidden | latent_channels | Completed epochs | Best epoch | Best validation loss | Improvement over 64/64 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 96 | 96 | 88 | 78 | 0.3271428113667308 | 3.2634% |
| 2 | 64 | 96 | 81 | 71 | 0.3284111303977505 | 2.8884% |
| 3 | 96 | 64 | 109 | 99 | 0.3374235863409882 | 0.2234% |
| 4 | 64 | 64 | 96 | 86 | 0.338179004596152 | baseline |

The result shows that increasing `latent_channels` from 64 to 96 is the material capacity change. Increasing only `point_hidden` from 64 to 96 has little effect. The 96/96 candidate ranks first and improves all reported full-domain RMSE variables relative to 64/64. The simpler 64/96 candidate is close behind and may offer a better capacity-efficiency trade-off.

The next gate is a second-seed repeat of only the strongest configurations, 96/96 and 64/96, before Processor transfer. No Processor training was started as part of this record.

## Storage and reproducibility

The 0.99 GB development cache and all `best.pt`/`last.pt` files remain local and are intentionally excluded from Git. Their sizes and SHA-256 hashes are recorded in `manifest.json`.

## Contents

- `ph096_lat064/`, `ph064_lat096/`, `ph096_lat096/`: complete histories and converged summaries.
- `comparison.json`: baseline-inclusive ranking and domain/variable metrics.
- `manifest.json`: provenance and local checkpoint hashes.
- `StormEngine_V8_Stage1_Capacity_Development_executed.ipynb`: compact executed run record.
