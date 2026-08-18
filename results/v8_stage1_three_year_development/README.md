# V8 Stage-1 three-year sigma convergence

This directory records the first converged three-year architecture-development comparison between Gaussian SetConv widths `sigma=0.10` and `sigma=0.15`. The run used commit `e646869bc4b2d573fec6908fab84563725a630ae` on branch `v8-model-development`.

## Scientific scope

- Training: ERA5 2013-2015
- Validation and early stopping: ERA5 2016
- ERA5 2017: not read
- August 2026 operational data: not read
- Common seed: 42
- Common maximum: 120 epochs
- Common early-stopping patience: 10
- Station profile: 390 DPC plus sea support points

Both candidates started from scratch and demonstrated validation convergence before comparison.

## Convergence results

| Sigma | Completed epochs | Best epoch | Best validation loss | Early stopped |
|---:|---:|---:|---:|:---:|
| 0.10 | 96 | 86 | 0.338179004596152 | yes |
| 0.15 | 93 | 83 | 0.33817766665270294 | yes |

The automatic normalized sea-weighted ranking places `sigma=0.15` first, but its validation-loss advantage is only `0.0003956%`. The variable-level result is mixed: over sea, `sigma=0.15` improves MSL, u10, and t2m RMSE by 2.49%, 2.09%, and 4.70%, while worsening v10 and tp RMSE by 3.26% and 5.26%.

The comparison JSON therefore preserves the mechanical ranking, while the scientific interpretation remains that the result is marginal and mixed. Under the documented project policy, a marginal or strongly mixed result defaults to `sigma=0.10`, which already has a converged six-year checkpoint. No six-year `sigma=0.15` retraining or Processor training was started as part of this record.

## Reproducibility and storage

The reusable 2013-2016 development cache is approximately 0.99 GB and remains local under `DownloadDate/cache/stormengine_dev_2013_2016`. The cache, `best.pt`, and `last.pt` files are intentionally excluded from Git. Their sizes and SHA-256 hashes are recorded in `manifest.json`.

The committed cache identity manifest is corrected to match the deterministic Windows cache produced and verified by `scripts/build_development_cache.py`.

## Contents

- `sigma010/`: complete history and converged development summary for `sigma=0.10`.
- `sigma015/`: complete history and converged development summary for `sigma=0.15`.
- `comparison.json`: automatic ranking and per-domain metrics.
- `manifest.json`: provenance, local checkpoint hashes, and storage exclusions.
- `StormEngine_V8_Stage1_ThreeYear_Development_executed.ipynb`: compact executed run record.
