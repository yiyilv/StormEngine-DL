# V8 Stage-1 spatial capacity development

Gaussian sigma is fixed at `0.10` after the converged three-year sigma result
showed no meaningful difference from `0.15`. This gate compares a complete
two-by-two capacity design:

| point_hidden | latent_channels | status |
|---:|---:|---|
| 64 | 64 | existing converged baseline |
| 96 | 64 | new candidate |
| 64 | 96 | new candidate |
| 96 | 96 | new candidate |

All configurations use ERA5 2013--2015 for training, ERA5 2016 for validation
and early stopping, seed 42, maximum 120 epochs, patience 10, the same compact
cache, and the same missingness augmentation. ERA5 2017 and the August 2026
operational week are not read.

Run `notebooks/StormEngine_V8_Stage1_Capacity_Development.ipynb` on Windows
CUDA. It verifies the cache, preflights all three new candidates, runs them
sequentially, resumes an interrupted candidate from its own `last.pt`, skips a
completed candidate, and finally compares all four converged configurations.

The primary ranking is normalized sea-weighted 2016 validation loss. Physical
full/land/sea RMSE and effect sizes relative to `64/64` must also be inspected.
The automatic top two are not yet final Processor initializations: if their
advantage is small, repeat only the strongest configurations with a second seed
before claiming a capacity improvement.
