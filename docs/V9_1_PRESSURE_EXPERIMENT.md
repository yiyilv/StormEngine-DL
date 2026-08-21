# V9.1 pressure ablation and final event verification

This workflow completes two previously open scientific requirements without
modifying the frozen V9-A checkpoint.

## Frozen V9-A event extension

`evaluate_v9_2025_events.py` replays the exact frozen 2025 checkpoint and
windows. It verifies that replayed full/land/sea MAE and RMSE match the already
published benchmark before adding strong-wind, heavy-precipitation, and
low-pressure POD, FAR, CSI, event-conditioned RMSE, peak-intensity bias, and
lead-hour metrics. Thresholds remain derived from 2010--2015, exactly as in the
earlier frozen benchmark. This is metric completion, not training or post-test
tuning.

## Controlled pressure experiment

The pressure experiment reserves a new chronological split:

- training: 2015--2017;
- validation and model selection: 2018;
- acknowledged one-time test: 2019.

The compact three-year development period is deliberate: this is a paired
input ablation rather than the final production retraining. It reduces the four
run training cost while preserving multiple annual cycles and a strictly later
validation/test chronology. A successful pressure contract can subsequently be
used in the final long-record training.

The paired control uses `u10,v10,i10fg,t2m,tp`. The pressure candidate adds
`msl` and changes nothing else. Both use the field-autoregressive V9 form, the
same V7-B warm start, two seeds, training budget, missingness augmentation,
loss, and early stopping. Common point-MLP columns are copied exactly; the new
MSL value, mask, and age columns start at zero.

Pressure is structurally available only at the 13 physical stations validated
by the DPC pressure-reduction audit and at all 151 model-derived marine support
points. Every other physical pressure cell has `value_mask=0`. Random
variable/station/network missingness is subsequently applied during training.

The frozen 2018 gate requires positive full- and sea-domain MSL RMSE skill for
both seeds, improved low-pressure event RMSE for both seeds, and no more than a
1% two-seed mean degradation across the four non-MSL sea RMSEs. Only a passed
full validation result unlocks 2019. The 2019 result is reported once and cannot
trigger tuning.

## Windows order

Run `notebooks/StormEngine_V9_1_Pressure_Experiment.ipynb` in order:

1. frozen V9-A 2025 event extension;
2. paired preflight;
3. paired smoke run;
4. four sequential full runs (control/pressure, seeds 42/43);
5. frozen 2018 validation;
6. inspect the gate, and only if it passed, acknowledge the one-time 2019 test.

Checkpoints, caches, and bulk predictions stay outside Git. Published JSON,
README, manifests, and the executed notebook may be committed after review.
