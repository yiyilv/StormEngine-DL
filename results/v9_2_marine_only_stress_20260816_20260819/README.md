# V9.2 marine-only strong-weather stress test

## Status

This is a **no-truth operational stress test**, not an accuracy evaluation.
All 239 physical DPC stations are deliberately masked as offline. The only
available input is the complete 151-point Open-Meteo/ICON-2I marine support
network for 2026-08-16 00:00 through 2026-08-19 00:00 UTC.

Both frozen checkpoints completed all 56 forecast windows without NaN or a
runtime failure.

## Input stress

- Maximum marine input gust: 31.2 m/s
- Maximum marine input hourly precipitation: 37.6 mm
- Hours with at least one gust >= 20 m/s: 9
- Hours with at least one hourly precipitation value >= 5 mm: 14

These values describe model-derived input support and are not verification
truth.

## Predicted response

| Frozen model | Maximum predicted sea wind | Maximum predicted sea 6 h precipitation | Forecast cases above 10 mm rain |
|---|---:|---:|---:|
| V9.2 Final16 | 13.22 m/s | 28.01 mm | 4 |
| V9.1 | 10.76 m/s | 18.35 mm | 1 |

Neither model predicted a sea grid cell above the frozen 15 m/s strong-wind or
30 mm/6 h heavy-rain threshold. V9.2 responds more strongly than V9.1, but the
complete physical-network outage produces conservative extreme amplitudes.

## Interpretation

This result demonstrates numerical stability and a stronger event-sensitive
response in V9.2 under a severe missing-network scenario. It does **not** show
whether either prediction is correct, because no independent DPC or ERA5T truth
is used. It also must not replace the existing 2018 independent ERA5 event
evaluation, which remains the evidence for event skill.

The diagnostic suggests retaining explicit physical-network outage scenarios
in deployment monitoring. It does not justify retraining or changing the frozen
thresholds by itself.

