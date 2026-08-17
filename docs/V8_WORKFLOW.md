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

## Stage 2: Processor-only family development

Before connecting the sparse interface, isolate temporal forecasting skill with
the dense ERA5 contract `12 x [msl,u10,v10,t2m,tp] -> 6 x the same fields`.
This stage deliberately reads no DPC, Open-Meteo, sparse point values, Encoder,
or Decoder checkpoints.

The reduced-cost family screen uses 2013--2015 training, 2016 validation, and a
three-hour window stride from the validated development cache. Both candidates
use the same normalized sea-weighted loss and validation early stopping:

- a two-layer, 96-channel, 3x3 ConvGRU;
- a factorized temporal/spatial ViT with 4x4 patches.

Run `notebooks/StormEngine_V8_Processor_Family_Development.ipynb`. It profiles
both candidates before optionally launching them in separate Windows console
processes. Parallel training is allowed only when the conservative combined
CUDA-memory check passes. Sharing one GPU may still make two simultaneous jobs
slower, so this is a scheduling convenience rather than a speed guarantee.

The first comparison is a seed-42 family screen, not final model selection.
Repeat both families with a second seed before choosing a family. If the
ranking is unstable or effectively tied, prefer the smaller model and carry
both into the later end-to-end compatibility gate. The 2017 test remains unread.

After seed 42 finishes, run
`notebooks/StormEngine_V8_Processor_Replication.ipynb`. It computes persistence
on the identical stride-3 2016 windows (persistence has no training phase),
runs both families with seed 43, and reports mean validation loss, between-seed
variation, skill relative to persistence, and ranking consistency. These
development values must not be compared numerically with the old 2017 V6/V7
scores; direct comparison is deferred until the retained V8 system is trained
on the full development split and evaluated once on the same 2017 test.

## Later stages

1. tune a small, preregistered parameter set for the retained Processor family;
2. initialize the V8 forecast Encoder and Decoder from Stage 1 and insert the retained Processor;
3. test Processor ranking under generic missingness and the deployment-compatible sparse interface;
4. jointly fine-tune Encoder, Processor, and Decoder;
5. compare against frozen V7-B on 2016 before unlocking the 2017 test;
6. run 2017 once after the architecture and hyperparameters are frozen;
7. finally repeat the fixed August 2026 operational evaluation.
