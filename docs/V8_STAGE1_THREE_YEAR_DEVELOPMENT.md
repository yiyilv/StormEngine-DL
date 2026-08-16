# V8 Stage-1 three-year development gate

The 15-epoch screen made `sigma=0.15` a promising but mixed candidate. A full
six-year run for every spatial alternative is unnecessarily expensive during
architecture selection. This gate compares `0.10` and `0.15` on the same
reduced development task:

- ERA5 2013--2015: training;
- ERA5 2016: validation and early stopping;
- ERA5 2017: not read;
- August 2026 operational week: not read;
- seed 42;
- maximum 120 epochs (a safety ceiling, not a fixed run length);
- early-stopping patience 10;
- full available training and validation batches each epoch.

Both candidates start from scratch. The existing six-year sigma=0.10 model is
not compared directly with these three-year candidates.

The development workflow uses one compact, immutable cache containing
2013--2016. Build it once from the already validated 2010--2017 cache. It is a
byte-for-byte temporal subset: no values are recomputed or renormalized.

```text
python -u scripts/build_development_cache.py build \
  --source-cache ../DownloadDate/cache/stormengine_2010_2017 \
  --output-cache ../DownloadDate/cache/stormengine_dev_2013_2016

python -u scripts/build_development_cache.py verify \
  --output-cache ../DownloadDate/cache/stormengine_dev_2013_2016
```

The repository already contains the expected cross-platform identity manifest;
the preflight validates the local derived cache against that committed file.

Run on Windows CUDA:

```text
python -u scripts/train_v8_reconstruction.py preflight --device cuda --config configs/v8_reconstruction_dev3y_sigma010.yaml
python -u scripts/train_v8_reconstruction.py develop --device cuda --config configs/v8_reconstruction_dev3y_sigma010.yaml

python -u scripts/train_v8_reconstruction.py preflight --device cuda --config configs/v8_reconstruction_dev3y_sigma015.yaml
python -u scripts/train_v8_reconstruction.py develop --device cuda --config configs/v8_reconstruction_dev3y_sigma015.yaml
```

Compare them:

```text
python -u scripts/compare_v8_spatial_development.py \
  artifacts/v8_spatial_dev3y_sigma010 \
  artifacts/v8_spatial_dev3y_sigma015 \
  --output artifacts/v8_spatial_dev3y_comparison.json
```

The comparator rejects different years, seeds, contracts, epoch caps, or batch
budgets. It also rejects a candidate that merely reached epoch 120 without
triggering early stopping. In that case raise the common ceiling and rerun both
candidates from scratch; do not rank unconverged checkpoints. The output is
architecture-selection evidence, not a final model result. Stop after
comparison and preserve both local checkpoints.

If `0.10` wins, the already converged six-year sigma=0.10 checkpoint can move
to the Processor-transfer gate. If `0.15` wins convincingly, train only that
winner once on 2010--2015 before transfer. A marginal or strongly mixed result
defaults to `0.10` because it is already converged and cheaper scientifically.
