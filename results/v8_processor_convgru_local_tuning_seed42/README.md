# V8 ConvGRU local tuning — seed 42

This record freezes the first local ConvGRU depth/kernel search after ConvGRU won the Processor-family comparison on seeds 42 and 43.

## Fixed experiment contract

- ERA5 2013–2015 development training
- ERA5 2016 validation and model selection
- 2017 was not read
- seed: 42
- latent channels: 96
- primary metric: normalized sea-weighted 2016 validation loss
- reused baseline: 2 layers, 3x3 kernel
- newly trained candidates: 1 layer/3x3, 3 layers/3x3, 2 layers/5x5

## Ranking

| Rank | Layers | Kernel | Parameters | Best epoch | Completed epochs | Best validation loss |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 3x3 | 1,494,917 | 44 | 54 | 0.1317562490 |
| 2 | 2 | 5x5 | 2,766,437 | 35 | 45 | 0.1330885594 |
| 3 | 2 | 3x3 | 996,965 | 46 | 56 | 0.1350536781 |
| 4 | 1 | 3x3 | 499,013 | 54 | 64 | 0.1369023472 |

Relative to the 2-layer/3x3 baseline, the 3-layer/3x3 candidate lowers validation loss by about 2.44%. At two layers, changing the kernel from 3x3 to 5x5 lowers it by about 1.46%.

## Decision

The best configuration in this seed is 3 layers with a 3x3 kernel. Because both added depth and a larger kernel helped in their controlled comparisons, the scripted gate requests one interaction candidate: 3 layers with a 5x5 kernel. Replication should wait until that interaction run is evaluated.

## Published files

- executed tuning notebook
- final comparison JSON
- development histories and summaries for the three newly trained candidates
- checkpoint manifest with local SHA-256 values

The large `best.pt` and `last.pt` files remain on the Windows workstation and are intentionally not committed to ordinary Git.
