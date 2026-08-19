# V8 frozen 2016 validation benchmark

This benchmark decides whether the provisional V8 Stage-3A candidate has
demonstrated forecast skill before the locked 2017 V8 test split is read.

## Frozen comparison contract

- Event thresholds: ERA5 2010--2015 target grids only.
- Verification: the same clean 2016 windows for every model, with a three-hour
  window stride, 12 input hours, and six forecast hours.
- Input support: the fixed 239 physical plus 151 model-derived marine
  coordinates. Historical values at all 390 coordinates remain ERA5 samples.
- Variables: `msl,u10,v10,t2m,tp`.
- Regions and leads: full, land, sea and +1 through +6 hours.
- Locked V8 test: 2017 is declared in the model configuration but is never
  instantiated by this evaluator. Every output records `test_years_read=[]`.

The five frozen models are V7-B, V8 Stage 2 seeds 42/43, and V8 Stage 3A seeds
42/43. Every local checkpoint is verified against the SHA-256 already recorded
in Git before it is loaded.

## Persistence baselines

`sparse_reconstruction_persistence` is the input-fair baseline. It passes the
last sparse input hour through the original Stage-2 Encoder and Decoder while
bypassing the Processor, then repeats that reconstructed current grid for all
six leads. It has the same sparse information as V8 and answers whether the
temporal Processor adds value beyond current-field reconstruction.

`dense_era5_persistence` repeats the complete current ERA5 grid for all six
leads. It has stronger information than any deployable sparse model and is
therefore a diagnostic reference, not the sole acceptance baseline.

RMSE skill is always

```text
1 - candidate RMSE / reference RMSE
```

Positive values mean the candidate is better.

## Pre-registered acceptance gate

For both Stage-3A seeds:

1. the mean sea RMSE skill across the five variables must be positive relative
   to the matching Stage-2 seed;
2. sea `u10` and `v10` must each beat sparse reconstruction persistence at no
   fewer than four of six leads;
3. +6 h sea wind RMSE must not be lower than +1 h RMSE, which catches an
   implausible reversed lead-time pattern;
4. the already-frozen simultaneous-reconstruction degradation must remain at
   or below 3%.

Event metrics are reported diagnostics, not extra thresholds tuned on 2016.
Failure keeps 2017 locked. Passing permits freezing the selected Stage-3A
checkpoint before the one-time V8 test.

## Event verification

All event thresholds are fixed from 2010--2015 sea cells:

- strong wind: 95th percentile of `sqrt(u10^2 + v10^2)`;
- training-distribution heavy precipitation: 95th percentile among wet cells
  with preceding-hour `tp >= 0.1 mm`;
- fixed precipitation event: `tp >= 5 mm` in the preceding hour;
- low pressure: 5th percentile of `msl`.

For every model and persistence baseline, the evaluator reports event-cell
RMSE, probability of detection (POD), false alarm ratio (FAR), critical success
index (CSI), and event-case peak-intensity bias, both aggregate and by lead.
`msl` remains a weak target because the operational point input has no
same-semantics sea-level-pressure channel.

## Windows execution

Run the notebook `notebooks/StormEngine_V8_2016_Benchmark.ipynb`. It first runs
checkpoint/data preflight and then the complete benchmark. The main output is:

```text
artifacts/v8_2016_benchmark/benchmark.json
results/v8_2016_validation_benchmark/
```

The notebook automatically publishes the lightweight benchmark, README, and
cross-platform manifest under `results/`. These files can be committed;
checkpoints, cache arrays, and ERA5 NetCDF files remain local.
