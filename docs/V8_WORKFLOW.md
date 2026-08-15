# V8 model-development workflow

V8 starts from the frozen V7-B data contract and does not modify V7 results.
It uses 239 physical DPC/MeteoHub coordinates plus 151 model-derived marine
support coordinates, five input variables (`u10,v10,i10fg,t2m,tp`), per-variable
masks and ages, and the same five target fields (`msl,u10,v10,t2m,tp`).

The development split remains immutable:

- 2010--2015: training;
- 2016: validation and model decisions;
- 2017: locked until the complete V8 pipeline is frozen;
- the August 2026 operational week: locked deployment case study, not tuning data.

## Stage 1: mask-aware spatial pretraining

`scripts/train_v8_reconstruction.py` uses only the final hour of each 12-hour
history and reconstructs the dense grid at that same valid time. The temporal
Processor is absent. This isolates and pretrains the Encoder and Decoder using
the exact V7-B/V8 input contract.

Run in this order on Windows CUDA:

```text
python -u scripts/train_v8_reconstruction.py preflight --device cuda
python -u scripts/train_v8_reconstruction.py smoke --device cuda --output-dir artifacts/v8_spatial_smoke
python -u scripts/train_v8_reconstruction.py pilot --device cuda --output-dir artifacts/v8_spatial_pilot
```

The pilot is a pipeline and learning-direction check, not a reportable model
result. Run the full validation experiment only after its loss is finite and
decreasing:

```text
python -u scripts/train_v8_reconstruction.py train --device cuda
```

The full run writes `best.pt`, `last.pt`, `history.json`, and
`train_summary.json` under `artifacts/v8_spatial_pretraining`. Checkpoints stay
outside ordinary Git. The checkpoint contract permits strict transfer of only
the Encoder and Decoder into the future forecasting model; the Processor is
initialized and trained separately.

MSL remains in the reconstruction output so Decoder shapes transfer exactly.
It must be interpreted separately because the present deployment-compatible
inputs contain no station variable equivalent to ERA5 mean-sea-level pressure.
Raw DPC station pressure must not be added without sea-level correction.

## Later stages

1. initialize the V8 forecast Encoder and Decoder from Stage 1;
2. train the temporal Processor on 2010--2015 and select with 2016 only;
3. jointly fine-tune Encoder, Processor, and Decoder;
4. compare against frozen V7-B on 2016 before unlocking the 2017 test;
5. run 2017 once after the architecture and hyperparameters are frozen;
6. finally repeat the fixed August 2026 operational evaluation.

