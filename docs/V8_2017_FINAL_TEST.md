# V8 one-time 2017 final test

This workflow performs the final locked comparison after both V8 Stage-3A
seeds passed the pre-registered 2016 validation gate. It is evaluation only:
no parameter is updated and no result may be used for another tuning cycle.

## Frozen contract

- Threshold source: ERA5 2010--2015 sea cells only.
- Test split: exactly 2017.
- Shared windows: stride 3, 12 input hours, 6 forecast hours.
- Shared input support: fixed 390-point `dpc_plus_sea` profile.
- Targets: `msl,u10,v10,t2m,tp` on the 31 x 33 grid.
- Regions: full, land, and sea.
- Leads: +1 through +6 hours.

The evaluator first verifies the SHA-256 and passing status of the frozen 2016
benchmark. It then verifies every checkpoint SHA-256 and model contract before
reading the test split. The compared frozen models are:

- V8 Stage 3A Seed 42;
- V7-B;
- input-fair sparse reconstruction persistence;
- stronger-information dense ERA5 persistence.

The output includes aggregate and lead-wise MAE/RMSE, RMSE skill, and the same
strong-wind, heavy-precipitation, and low-pressure event verification used in
2016. Dense persistence remains a diagnostic rather than a deployment-equivalent
gate.

## Pre-declared interpretation

The V8 replacement claim is supported only when both conditions hold:

1. the mean sea RMSE skill across all five targets is positive relative to
   frozen V7-B;
2. V8 wins at least 7 of the 12 sea `u10`/`v10` component-by-lead comparisons.

Event metrics remain separately reported scientific diagnostics. Regardless of
the outcome, no test-driven retraining follows. A negative result is retained
as evidence that V7-B remains the stronger frozen reference.

## Windows execution

Open and run:

```text
notebooks/StormEngine_V8_2017_Final_Test.ipynb
```

The notebook deliberately has one formal execution cell. It passes the explicit
`--acknowledge-one-time-test` flag, and the evaluator refuses to overwrite an
existing complete result. Do not add `--max-batches` to the formal run.

Outputs are written to:

```text
artifacts/v8_2017_final_test/benchmark.json
results/v8_2017_final_test/
```

Only the lightweight published result, README, and manifest belong in Git.
Checkpoints, cache arrays, and NetCDF files remain local.
