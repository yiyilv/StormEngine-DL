# V7-A workflow

V7-A is the physical-station deployment baseline: 239 DPC/MeteoHub stations,
12 input hours, five input variables (`u10`, `v10`, `i10fg`, `t2m`, `tp`), and
five target fields for leads +1 through +6 hours. It contains no virtual sea
points, Open-Meteo inputs, pressure, or humidity.

The canonical interactive entry point is
`notebooks/StormEngine_V7A_Workflow.ipynb`. Its stages are intentionally
separate:

1. V7 unit tests and preflight.
2. Smoke training and a 200-batch speed benchmark.
3. Five-epoch capped pilot.
4. Full 2010-2015 training with 2016 model selection.
5. Frozen 2017 clean and missing-input evaluation.
6. 152-window real DPC replay for 2026-08-01 through 2026-08-08.
7. Frozen-manifest creation.

Formal training, full evaluation, and replay are disabled by default in the
notebook. The user must enable each gate explicitly after inspecting the prior
stage.

Training uses generic randomized variable dropout, whole-station dropout,
contiguous network outages, and true one-hour lagged values for simulated age.
The empirical DPC week is never a training template. It is used by the DPC
Adapter and the final runtime replay.

Large local outputs under `artifacts/v7_a_2010_2017/`, especially `best.pt`,
`last.pt`, and optional prediction arrays, must remain outside ordinary Git.
The small frozen manifest records their hashes for synchronization.

