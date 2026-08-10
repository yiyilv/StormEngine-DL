# V7 mask-aware development gate

This directory records lightweight checks used to decide whether a V7 candidate
may enter a longer pilot. It does not contain model checkpoints or training
caches.

## Frozen input contract

- 239 enabled physical coastal stations; the 151 virtual marine support points
  are excluded.
- Five ERA5/DPC shared inputs: `u10`, `v10`, `i10fg`, `t2m`, and hourly `tp`.
- Per-variable `value_mask[T,N,C]` and normalized
  `observation_age[T,N,C]` channels.
- Observation age is in hours and capped at 1 (the 60-minute operational
  latest-at-or-before limit).
- Station pressure is excluded because official `B10004` station pressure must
  not be silently treated as ERA5 mean-sea-level pressure.
- The V6 390-point checkpoint is deliberately incompatible and remains frozen.

## Candidate ladder

- `v7_a0.yaml`: five variables plus per-variable masks, without age channels.
- `v7_a1.yaml`: five variables plus masks and age channels; primary candidate.
- `v7_a2.yaml`: A1 with gust removed; controlled sparse-gust ablation.

All candidates reuse the immutable 2010-2017 ERA5 memmap cache through a
validated sidecar identity. Training uses generic randomized missingness; the
pinned contiguous DPC windows are reserved for validation and stress testing.
No duplicate multi-gigabyte cache is required.

## Promotion rules

1. Unit and forward checks must pass with finite tensors and predictions.
2. A 1-10 batch smoke run must complete forward, backward, and validation.
3. The 200-batch benchmark must establish feasible runtime and memory.
4. A capped 2010-2012 pilot may then run for at most 5 epochs (300 training and
   75 clean-validation batches per epoch).
5. Only promising candidates proceed to reconstruction diagnostics and then one
   full 2010-2017 experiment.

`benchmark_rtx4060_20260810.json` is the five-channel A1 speed gate. Local `*.pt`, ERA5
`*.nc`, cache memmaps, and detailed predictions remain outside Git.
