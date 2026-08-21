# V9-A warm-start refinement (2023 development only)

This controlled diagnostic was run after the V9 output-form screen showed that the
field/autoregressive candidate converged unusually early. It measures the untouched
V7-B warm start, then changes only fine-tuning learning rate and reconstruction weight.

- Training years: `2020--2022`
- Validation and model-selection year: `2023`
- Confirmation year `2024`: **not read**
- Locked final-test year `2025`: **not read**
- Epoch-0 validation MSE: `0.337319`
- Original V9-A two-seed mean: `0.316049`
- Refined V9-A two-seed mean: `0.310496`
- Relative validation-MSE improvement: `1.76%`

## Frozen candidate for one-time 2024 confirmation

`A_lr25e6_recon000`: field output, autoregressive forecasting, learning rate
`2.5e-5`, reconstruction-loss weight `0`, primary seed `42`.

- Best epoch: `14`
- Primary validation MSE: `0.310793`
- Replication validation MSE: `0.310198`
- Checkpoint SHA-256: `e8486b695d03e4a27f8ba5921eb3bb77e5e2d66c649f02ad289392576aeab7c3`

The checkpoint itself is intentionally not tracked. `checkpoint_manifest.json` records
all local checkpoint paths, sizes, and hashes. `histories.json` preserves every epoch,
and `metrics.json` is the compact scientific summary.
