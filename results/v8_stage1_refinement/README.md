# V8 Stage-1 spatial refinement

This directory records the controlled V8 Stage-1 refinement run performed on Windows on 2026-08-16. The code base was `952e0f1` (`Add V8 spatial refinement workflow`) on branch `v8-model-development`.

## Sigma 0.10 continuation

The published Stage-1 `sigma=0.10` model was resumed from epoch 60 with its optimizer, scheduler, and early-stopping state intact. Training used ERA5 2010-2015, model selection used 2016, and neither 2017 nor the August 2026 operational week was read.

- Target maximum: 90 epochs
- Early-stopped: epoch 85
- Best epoch: 75
- Best normalized validation loss: `0.3207312494271646`
- Source checkpoint SHA-256: `23cda69cb9481b21a8dbfc236ffc6915e81e06367081db982d91d69e86ab407e`

The final checkpoints remain local and are intentionally excluded from Git:

- `artifacts/v8_spatial_sigma010_extended/best.pt` — SHA-256 `5676383ba369f1b5e6b140e90c551c6f0e744d49ce04cb04a93bdf1bd1ac6f6a`
- `artifacts/v8_spatial_sigma010_extended/last.pt` — SHA-256 `e7cc97af0d95355f485f983ef0b16625f30075672dcb9d44df9ea35715f3ca60`

## Controlled sigma screens

All candidates started from the same random initialization (`seed=42`) and used the same capped budget: 15 epochs, 1,000 training batches per epoch, and 200 validation batches per epoch.

| Rank | Gaussian sigma | Best epoch | Best validation loss |
|---:|---:|---:|---:|
| 1 | 0.15 | 14 | 0.45815034009516237 |
| 2 | 0.10 | 14 | 0.46436164192855356 |
| 3 | 0.07 | 14 | 0.47129186008125545 |

`sigma=0.15` is the screening winner under the normalized sea-weighted 2016 validation loss. This is a screening result, not final model selection. The next gate is to train the strongest non-baseline candidate to full convergence and then compare full reconstruction and identical Processor-transfer pilots.

## Contents

- `continuation_sigma010/`: complete history and summary for the converged continuation.
- `sigma_screens/`: histories and summaries for all three candidates plus the automatic ranking.

Large checkpoints, ERA5 caches, NetCDF files, and other generated artifacts are intentionally not committed.
