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

## Stage 2b: ConvGRU local tuning

The two-seed family screen retained ConvGRU. Keep `latent_channels=96` fixed so
the Processor remains compatible with the selected spatial latent interface.
Run `notebooks/StormEngine_V8_ConvGRU_Local_Tuning.ipynb` to reuse the converged
2-layer/3x3 seed-42 result and add three local candidates: 1-layer/3x3,
3-layer/3x3, and 2-layer/5x5. All other data, optimization, split, seed, loss,
and early-stopping settings remain identical.

This adaptive design first estimates depth effects at kernel 3 and the local
kernel effect at depth 2. If both a non-default depth and kernel 5 improve the
baseline, run their interaction combination before replication. Otherwise,
repeat only a meaningfully better winner with seed 43; if no effect is
meaningful, retain the already replicated 2-layer/3x3 baseline. Do not use 2017
or start end-to-end fine-tuning during this local parameter decision.

## Stage 2c: sparse Processor-only transfer gate

The dense development gate retained ConvGRU L3K3. Formal Stage 2 now tests
that temporal choice inside the deployment-compatible sparse interface. It
compares the two converged Stage-1 spatial candidates (`point_hidden / latent`
of `64/96` and `96/96`) with seeds 42 and 43. For every run:

- the matching Stage-1 `best.pt` is loaded with an exact reconstruction
  contract check;
- Encoder and Decoder parameters are frozen and tensor-hashed;
- the L3K3 Processor is initialized from the Stage-2 seed, independently of
  Encoder-construction RNG consumption;
- only `processor.*` parameters enter the optimizer;
- training uses 2013--2015, validation uses 2016, and 2017 is never
  instantiated;
- generic V7/V8 missingness augmentation is active only in training;
- checkpoints contain the complete model so the selected `best.pt` can become
  the starting point for Stage 3.

The shared development contract uses stride 3 and batch size 8 to keep the
four-run transfer gate tractable while matching the preceding dense Processor
screen. Validation remains clean and chronological.

On Windows CUDA, retain these local Stage-1 checkpoints at their original
artifact paths:

```text
artifacts/v8_spatial_dev3y_ph064_lat096/best.pt
artifacts/v8_spatial_dev3y_ph096_lat096/best.pt
```

Then open `notebooks/StormEngine_V8_Stage2_Processor_Only.ipynb` and run its
single code cell, or invoke:

```text
python -u scripts/run_v8_stage2.py --phase all --device cuda
```

The runner performs preflight, smoke and capped pilot checks before launching
the four formal runs sequentially. Completed runs are validated and skipped;
interrupted formal runs resume from `last.pt`. Selection uses the two-seed
mean normalized sea-weighted 2016 validation loss. If the mean gap is below
1%, the smaller spatial MLP is preferred. Per-variable and per-lead-hour
metrics must still be reviewed before Stage 3.

## Stage 3: six-year gradual unfreezing

Stage 2 retained `PH64-LAT96` with ConvGRU L3K3. Stage 3 expands training to
2010--2015 while keeping 2016 as the only development validation year and
leaving 2017 uninstantiated. It deliberately avoids an abrupt equal-rate
unfreeze:

1. **Stage 3A** loads each retained Stage-2 `best.pt`, keeps Encoder frozen,
   and trains Processor plus Decoder for at most 10 adaptation epochs. The
   initial learning rates are `5e-5` and `1e-5`, respectively.
2. **Stage 3B** loads the matching Stage-3A `best.pt` and jointly fine-tunes
   Encoder, Processor, and Decoder with discriminative learning rates of
   `5e-6`, `3e-5`, and `1e-5`.

Both seed-42 and seed-43 lineages are retained. Optimizer and scheduler state
are rebuilt at each phase boundary because the trainable modules and data span
change. Generic missingness remains active only in training; validation is
clean and chronological.

The same clean 2016 simultaneous reconstruction diagnostic is evaluated on
the original Stage-2 source and on each selected Stage-3 checkpoint. Stage 3B
must remain within a preregistered 3% normalized reconstruction-loss
degradation relative to the original Stage-2 spatial state. Failing that gate
blocks 2017 and triggers a reconstruction-auxiliary-loss experiment; the
auxiliary objective is not added unless the diagnostic demonstrates a need.

On Windows CUDA, retain these local Stage-2 checkpoints:

```text
artifacts/v8_stage2_dev3y_ph064_lat096_seed42/best.pt
artifacts/v8_stage2_dev3y_ph064_lat096_seed43/best.pt
```

Open `notebooks/StormEngine_V8_Stage3_Gradual_Unfreezing.ipynb` and run its
single code cell, or invoke:

```text
python -u scripts/run_v8_stage3.py --phase all --device cuda
```

The two phases and two seeds run sequentially. A repeated invocation skips
validated completed runs and resumes interrupted formal runs from `last.pt`.
Lightweight comparison results are published under
`results/v8_stage3_6y_gradual_unfreezing/`; checkpoints remain local.

## Later stages

1. run the two-seed Stage-3A/Stage-3B gradual-unfreezing workflow;
2. verify the reconstruction-preservation gate and review variable/region/
   lead-hour metrics;
3. compare Stage 2 and Stage 3 against frozen V7-B on 2016;
4. run 2017 once after the complete V8 architecture and hyperparameters are
   frozen;
5. finally repeat the fixed August 2026 operational evaluation.
