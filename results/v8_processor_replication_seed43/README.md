# V8 dense Processor replication — seed 43

This directory records the seed-43 replication of the V8 dense Processor family screen, together with the same-window 2016 dense-persistence baseline and the combined seed-42/43 comparison. The run used commit `31145251e0fd5421cb37d1a02689ba20d316d7b2` on branch `v8-model-development`.

## Scientific scope

- Training: ERA5 2013-2015
- Validation and early stopping: ERA5 2016
- ERA5 2017: not read
- August 2026 operational data: not read
- Sparse stations, DPC, Open-Meteo, Encoder, and Decoder: not read
- History: 12 hours
- Forecast horizon: 6 hours
- Development stride: 3 hours
- Seeds compared: 42 and 43
- Maximum: 120 epochs
- Early-stopping patience: 10

## Seed-43 results

| Family | Parameters | Completed epochs | Best epoch | Best validation loss |
|---|---:|---:|---:|---:|
| ConvGRU | 996,965 | 62 | 52 | 0.13434550805841802 |
| Factorized ViT | 1,995,173 | 53 | 43 | 0.13715949531420657 |

Both candidates demonstrated validation convergence. ConvGRU wins seed 43, reproducing the seed-42 ranking.

## Two-seed conclusion

| Family | Mean validation loss | Between-seed standard deviation | Mean skill vs persistence |
|---|---:|---:|---:|
| ConvGRU | 0.1346995931022032 | 0.0005007518711544837 | 55.4219% |
| Factorized ViT | 0.1373931824870896 | 0.00033048356924381784 | 54.5304% |

Dense persistence has validation loss `0.30216525403446837`. ConvGRU wins both seeds, has a two-seed mean loss approximately `1.9605%` lower than Factorized ViT, and uses approximately half the parameters. The family ranking is therefore consistent across the two development seeds.

This is still a 2016 development decision, not a 2017 test result. The next stage should use ConvGRU as the preferred Processor family while retaining the published ViT evidence for the later end-to-end compatibility check if needed.

## Storage and reproducibility

Seed-43 `best.pt` and `last.pt` files remain local and are intentionally excluded from Git. Their sizes and SHA-256 hashes are recorded in `manifest.json`. Seed-42 checkpoint hashes are recorded in the preceding `v8_processor_family_development_seed42` result package.

## Contents

- `convgru_seed43/`, `factorized_vit_seed43/`: complete histories and converged summaries.
- `persistence_2016.json`: same-window dense-persistence baseline.
- `two_seed_comparison.json`: seed means, variation, persistence skill, and ranking consistency.
- `manifest.json`: provenance and local seed-43 checkpoint hashes.
- `StormEngine_V8_Processor_Replication_executed.ipynb`: compact executed run record.
