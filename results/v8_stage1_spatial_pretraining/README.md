# V8 Stage-1 spatial pretraining

V8 Stage-1 pretrains the mask-aware spatial Encoder and Decoder by
reconstructing the simultaneous ERA5 grid from the final sparse input hour.
The temporal Processor is absent. The input contract is the frozen V7-B set of
239 physical DPC/MeteoHub stations plus 151 model-derived marine points, with
`u10,v10,i10fg,t2m,tp` values, per-variable masks, and observation ages.

## Data and model-selection boundary

- ERA5 2010-2015: training.
- ERA5 2016: validation and checkpoint selection.
- ERA5 2017: not read.
- August 2026 operational week: not read.
- Grid: Adriatic 31x33.
- Processor used: no.
- Maximum epochs: 60.

## Training result

- Completed all 60 epochs in 2.397 hours.
- Mean epoch time: 143.8 seconds.
- Best epoch: 58.
- Best validation loss: `0.3241049153`.
- Final train loss: `0.3124605419`.
- Final validation loss: `0.3246199589`.
- Final validation/train gap: `0.01216`.
- No clear overfitting was observed; the best checkpoint occurred near the end.

The learning rate decreased from `1e-4` to `5e-5` at epoch 39 and to `2.5e-5`
at epoch 55. Validation continued improving after both reductions.

## 2016 validation reconstruction RMSE

| variable | full | land | sea |
|---|---:|---:|---:|
| msl | 5.4251 | 5.5935 | 5.1935 |
| u10 | 1.5483 | 1.1371 | 1.9648 |
| v10 | 1.2946 | 1.0814 | 1.5321 |
| t2m | 1.9787 | 2.3375 | 1.3649 |
| tp | 0.3058 | 0.3428 | 0.2485 |

Relative to the five-epoch capped pilot, sea RMSE improved by 31.83% for
`u10`, 37.10% for `v10`, 39.82% for `t2m`, and 6.83% for `tp`. `msl` improved
but remains a weak target because no same-semantics pressure input is present.

These are simultaneous spatial-reconstruction metrics, not +1 through +6-hour
forecast metrics, so they must not be directly ranked against V7 forecast RMSE.

## Frozen local checkpoint

Stage-2 must initialize its Encoder and Decoder from:

```text
artifacts/v8_spatial_pretraining/best.pt
```

The checkpoint is 996,884 bytes with SHA-256
`23cda69cb9481b21a8dbfc236ffc6915e81e06367081db982d91d69e86ab407e`.
It remains local and is intentionally excluded from ordinary Git. `last.pt` is
also local and must not replace the selected epoch-58 checkpoint.

## Published artifacts

- `train_summary.json`: selected epoch and full/land/sea physical-unit metrics.
- `history.json`: all 60 train/validation losses, learning rates, and timings.
- `pilot_summary.json` and `pilot_history.json`: capped-pilot record.
- `smoke_summary.json`: two-batch pipeline record.
- `training_validation_curve.png`: full learning curve.
- `StormEngine_V8_Workflow_executed.ipynb`: executed preflight/smoke/pilot record.
- `manifest.json`: code, notebook, result, and local checkpoint hashes.
