# Metric review — V8 ConvGRU overnight selection

The automated recommendation is scientifically consistent with the declared selection rule: when the two-seed mean normalized validation losses differ by less than 1%, prefer the lower-complexity candidate.

## Primary comparison

| Candidate | Parameters | Seed 42 | Seed 43 | Two-seed mean | Sample std |
|---|---:|---:|---:|---:|---:|
| L3K5 | 4,149,125 | 0.1311568623 | 0.1310025233 | 0.1310796928 | 0.0001091342 |
| L3K3 | 1,494,917 | 0.1317562490 | 0.1318714298 | 0.1318138394 | 0.0000814452 |

L3K5 has a 0.56% lower mean validation loss, while using about 2.78 times as many parameters. Both candidates are stable across the two seeds.

## Per-variable review

For two-seed mean sea RMSE, L3K5 is better for `msl` (2.52%), `u10` (1.50%), `v10` (0.82%), and `tp` (0.31%). L3K3 is better for sea `t2m` (0.54%). For full-domain RMSE, L3K5 is better for `msl`, `u10`, `v10`, and `t2m`, while L3K3 is better for `tp` by 0.60%.

The gains are small relative to the parameter increase and are not uniform across every variable/region. Therefore **L3K3 remains the provisional Processor choice** for subsequent joint-model development. This is not a final end-to-end model claim; 2017 remains locked and unread.

## Validation

- Result manifest: all 17 listed files match byte counts and SHA-256 hashes.
- Result JSON files: all parse successfully.
- Executed notebook: zero saved error outputs.
- Tests: 6 passed (`test_convgru_overnight_selection.py` and `test_dense_processor.py`).
