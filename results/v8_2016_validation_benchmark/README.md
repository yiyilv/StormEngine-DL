# V8 frozen 2016 validation benchmark

All models use the same clean 2016 windows; event thresholds use only 2010--2015, and the V8 2017 test split is not read.

- Both Stage-3A seeds passed: `True`
- Samples: `2923`
- Window stride: `3 h`

| seed | mean sea skill vs Stage 2 | fair-persistence u10 leads | fair-persistence v10 leads | reconstruction gate | passed |
|---:|---:|---:|---:|---|---|
| 42 | 4.15% | 6/6 | 6/6 | True | True |
| 43 | 3.28% | 6/6 | 6/6 | True | True |

`benchmark.json` contains full/land/sea MAE and RMSE, +1--+6 h metrics, RMSE skill, and wind/precipitation/low-pressure event verification.
