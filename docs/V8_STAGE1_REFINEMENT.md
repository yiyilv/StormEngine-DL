# V8 Stage-1 refinement

The published 60-epoch `sigma=0.10` spatial run is the immutable baseline.
Refinement asks two bounded questions before Processor training:

1. was the baseline stopped by the epoch cap before convergence?;
2. does a narrower or wider SetConv Gaussian improve spatial representation?

This work does not repeat the already-established IDW forecast baseline and it
does not read 2017 or the August 2026 operational week.

## 1. Continue the baseline safely

The Windows computer must still contain both:

```text
artifacts/v8_spatial_pretraining/best.pt
artifacts/v8_spatial_pretraining/last.pt
```

Verify that `best.pt` has the published SHA-256
`23cda69cb9481b21a8dbfc236ffc6915e81e06367081db982d91d69e86ab407e`,
then continue from the epoch-60 `last.pt`:

```text
python -u scripts/train_v8_reconstruction.py train --device cuda \
  --config configs/v8_reconstruction_continue.yaml \
  --resume artifacts/v8_spatial_pretraining/last.pt
```

`max_epochs: 90` is the total target, so this runs at most 30 additional
epochs. Optimizer, scheduler, AMP scaler, history, and early-stopping state are
restored. Output is isolated in `artifacts/v8_spatial_sigma010_extended`; the
published 60-epoch directory is not overwritten. The earlier selected
`best.pt` is copied into the continuation directory before training, so it
remains selected if no later epoch improves validation.

## 2. Like-for-like sigma screen

Run every candidate from scratch. Do not resume a screen from the baseline.

```text
python -u scripts/train_v8_reconstruction.py screen --device cuda --config configs/v8_reconstruction_sigma007.yaml
python -u scripts/train_v8_reconstruction.py screen --device cuda --config configs/v8_reconstruction_sigma010.yaml
python -u scripts/train_v8_reconstruction.py screen --device cuda --config configs/v8_reconstruction_sigma015.yaml
```

Each candidate uses seed 42, 15 capped epochs, 1,000 training batches and 200
validation batches per epoch. The only intended contract difference is
`gaussian_sigma`.

Compare the three summaries:

```text
python -u scripts/compare_v8_spatial_screens.py \
  artifacts/v8_spatial_screen_sigma007 \
  artifacts/v8_spatial_screen_sigma010 \
  artifacts/v8_spatial_screen_sigma015 \
  --output artifacts/v8_spatial_sigma_comparison.json
```

The comparison utility rejects different splits, budgets, duplicate sigma
values, or contracts that differ in anything other than sigma. Ranking uses
the normalized sea-weighted 2016 validation loss and also reports physical-unit
sea RMSE for every target.

## 3. Selection gates

The screen is not final model selection. After inspection:

1. if neither `0.07` nor `0.15` is convincingly better, retain the converged
   `0.10` baseline;
2. otherwise train only the strongest non-baseline candidate to full
   convergence;
3. compare fully converged reconstruction metrics;
4. run identical short Processor-transfer pilots from the best two spatial
   checkpoints;
5. choose the Stage-2 initialization using 2016 future-forecast metrics, not
   reconstruction loss alone.

Do not delete or overwrite any selected checkpoint. Checkpoints remain local;
publish only summaries, histories, plots, and hashes.

