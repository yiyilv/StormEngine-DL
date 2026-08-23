# V9.2 Final16 independent 2026 operational evaluation

This evaluation uses 152 operational forecast windows from 2026-08-01 through
2026-08-08. The inputs are real DPC/MeteoHub physical observations plus
Open-Meteo marine support, including the separately aligned MSL channels. ERA5T
is the gridded reference target; it is not direct station truth.

## Input coverage

- physical DPC variable-cell coverage: 27.15%;
- marine Open-Meteo variable-cell coverage: 100%;
- combined six-variable coverage: 55.35%;
- combined MSL coverage: 42.05%;
- physical corrected-MSL coverage: 5.43%;
- marine Open-Meteo MSL coverage: 100%.

## Sea RMSE

| variable | frozen V9.1 | V9.2 Final16 | interpretation |
|---|---:|---:|---|
| msl | 0.8368 | 0.8591 | V9.2 +2.7% |
| u10 | 1.5568 | 1.5677 | V9.2 +0.7% |
| v10 | 1.6606 | 1.6498 | V9.2 -0.7% |
| t2m | 1.0670 | 1.0227 | V9.2 -4.2% |
| tp | 0.0240 | 0.0286 | V9.2 worse, small absolute scale |

Across the full grid, V9.2 improves MSL RMSE from 1.1167 to 1.0513 (about
5.9%) and slightly improves v10. It is worse for u10, t2m, and tp. The result is
therefore mixed rather than a uniform replacement of V9.1.

## Event limitation

ERA5T contains no sea grid cells or forecast cases crossing the project's
six-hour storm, extreme-rain, extreme-wind, or compound thresholds during this
week. POD/FAR/CSI are consequently undefined. This calm-period result verifies
stable real-input operation and ordinary-field accuracy, but cannot validate the
event-aware objective. V9.1 should remain a fallback until an eventful, untouched
2026 period is evaluated.

Complete field, lead-hour, coverage, checkpoint, and physical-event records are
in `metrics.json`.
