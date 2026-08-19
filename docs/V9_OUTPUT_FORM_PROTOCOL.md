# V9 output-form development protocol

## Why V9 starts from V7-B

The one-time 2017 comparison showed that the later V8 staged-training candidates did not
improve the frozen V7-B forecast. V7-B therefore remains the reference architecture and
initialisation source. V9 does not repeat the same staged-training recipe. It tests two
specific hypotheses that can address the observed lack of forecast skill:

1. predicting a change from the reconstructed current state may be easier than predicting
   each future field from zero; and
2. predicting all six lead times directly may avoid error accumulation in the existing
   autoregressive latent rollout.

These hypotheses form a controlled two-by-two experiment:

| Candidate | Forecast output | Temporal generation |
|---|---|---|
| A | absolute field | autoregressive |
| B | residual from reconstructed current field | autoregressive |
| C | absolute field | direct six-horizon |
| D | residual from reconstructed current field | direct six-horizon |

All four candidates use the same 390 coordinates, input variables, missingness simulator,
normalisation, sea-weighted loss, optimiser, training duration, and chronological split.

## Shared spatial-preservation constraint

Every candidate also reconstructs the dense current ERA5 grid from the final sparse input
hour. The same reconstruction decoder is initialised from frozen V7-B, and every candidate
uses the same auxiliary weight of `0.10`. Candidate ranking still uses future-forecast
validation loss only; reconstruction loss is a guardrail and reported diagnostic.

For residual candidates, the forecast decoder represents only an increment. Its transferred
feature layers are retained, but its final projection is zero-initialised so the initial
forecast equals the reconstructed current field instead of double-counting an absolute
V7-B forecast. The current reconstruction is supervised explicitly and cannot drift into an
unidentifiable second future-field decoder.

## Chronological data contract

- 2010–2025: cached ERA5 archive, using the frozen 2010–2015 normalisation statistics;
- 2020–2022: fast development training;
- 2023: validation, early stopping, and candidate ranking;
- 2024: one-time confirmation after the candidate and its settings are frozen;
- 2025: locked final test, not instantiated by the development trainer.

The old 2017 test result is historical evidence and is not used to tune V9. The V9 trainer
enforces the split above and only constructs datasets for 2020–2023.

## Initialisation and selection

The frozen V7-B checkpoint is verified by SHA-256. Shape-compatible encoder, ConvGRU, and
decoder tensors are transferred. A direct-horizon head is necessarily new. For residual
models, both the residual decoder and current-field reconstruction decoder start from the
V7-B decoder tensors. Every transferred tensor is recorded in the new checkpoint contract.

All four candidates first train with seed 42 until early stopping. The best two according to
2023 sea-weighted validation MSE are repeated with seed 43. Their two-seed mean and range
select exactly one candidate for the one-time 2024 confirmation. This screening result is
not yet the final scientific test result.

Before 2024 is read, the confirmation gate is frozen as follows:

- mean sea RMSE skill must be positive relative to frozen V7-B;
- at least 7 of the 12 sea `u10`/`v10` component-by-lead comparisons must be positive;
- current-field reconstruction degradation must remain at or below 3%;
- failure stops V9 without reading 2025.

If and only if the 2024 gate passes, 2025 is read once as the locked final test. No result
from 2024 or 2025 may trigger hyperparameter changes. The development runner implemented in
this stage constructs only 2020--2023 datasets; confirmation and final-test evaluators remain
separate commands so an overnight selection run cannot unlock either year accidentally.

## Windows execution

From the repository root in the existing `stormengine` environment:

```powershell
git pull
python scripts/build_training_cache.py --config configs/era5_2010_2025.yaml
python scripts/build_v7_cache_identity.py `
  --cache ..\DownloadDate\cache\stormengine_2010_2025 `
  --registry data\stations_registry.csv `
  --output data\manifests\v9_cache_identity_2010_2025.json
python scripts/run_v9_output_form_selection.py `
  --config configs/v9_dev_output_form.yaml `
  --warm-start artifacts\v7_b_2010_2017\best.pt `
  --device cuda
```

The four candidates run sequentially on one GPU. Running two training processes on the same
laptop GPU is not recommended because it makes timing and memory behaviour less reliable.
The runner performs preflight and smoke checks before every full run, preserves completed
runs, and writes `artifacts/v9_output_form/selection_protocol.json`.

If interrupted, resume an individual candidate with `scripts/train_v9_output_form.py` and
its `last.pt`. Do not use `--rerun` unless replacing an intentionally discarded experiment.

