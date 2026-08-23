# V9.2 final 16-year production refit

The frozen V9.2 strong design was refitted for exactly one complete epoch on all
ERA5 windows from 2010 through 2025. The epoch count and event-loss settings were
selected beforehand using the 2015--2017 development training and 2018
validation experiment.

## Training record

- years: 2010--2025 (all 16 years);
- samples: 46,747;
- batches: 2,922;
- event windows: 374 (0.8001% before weighted sampling);
- mean total loss: 0.543399;
- mean field loss: 0.404972;
- mean event loss: 0.138427;
- checkpoint SHA-256: `ea0afde51397bd88bc6aeca45e452cabadfd956821112c6828eb826fdc03e86f`.

The checkpoint is stored locally at
`artifacts/v9_2_event_aware_final16/seed_42/final.pt`. It is not included in
ordinary Git history. The file is 2,421,073 bytes and all 25 floating-point
state tensors passed a finite-value integrity check.

## Scientific interpretation

This checkpoint is the production model, not a new independent test result.
Every year from 2010 through 2025 participated in its refit, so none of those
years may be used to claim held-out accuracy. The frozen 2018 development
comparison justifies the architecture and schedule; a new 2026 or later target
period is required for independent evaluation.

Frozen V9.1 remains the continuous-field baseline and fallback. V9.2 improves
storm sensitivity but retains the documented unresolved risk that 2018 extreme
rain and compound-storm cases were not detected.
