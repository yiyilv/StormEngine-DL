# V8 one-time 2017 final test

This is the locked 2017 comparison. Models and thresholds were frozen before the test was read; no post-test tuning is permitted.

- Samples: `2915`
- Window stride: `3 h`
- V8 replacement claim supported: `False`
- Mean sea RMSE skill of V8 vs V7-B: `-1.11%`
- Positive sea-wind component/leads: `0/12`

| model | msl sea RMSE | u10 sea RMSE | v10 sea RMSE | t2m sea RMSE | tp sea RMSE |
|---|---:|---:|---:|---:|---:|
| v7_b | 4.1910 | 1.9837 | 1.7411 | 1.2073 | 0.3015 |
| v8_stage3a_seed42 | 4.1181 | 1.9916 | 1.7808 | 1.2858 | 0.2957 |

`benchmark.json` contains full/land/sea MAE and RMSE, +1--+6 h metrics, persistence skills, and frozen-threshold event verification.
