# Original six-hour physical-event evaluation

## Purpose

This post-freeze evaluation checks the event definitions proposed in the
original StormEngine architecture document without retraining or changing any
forecast checkpoint. It replays the frozen 2025 model comparison on the same
windows and verifies that the ordinary MAE/RMSE values exactly reproduce the
published one-time test before calculating additional event metrics.

The evaluation compares:

- the frozen V9-A forecast;
- the frozen V7-B reference;
- input-fair sparse reconstruction persistence;
- information-strong dense ERA5 persistence.

All event statistics are restricted to sea grid cells because marine forecast
performance is the primary project objective.

## Six-hour fields

The five target channels remain `msl`, `u10`, `v10`, `t2m`, and hourly `tp`.
No new model output is introduced. For every forecast origin, the evaluator
constructs:

```text
tp_6h_mm = sum(max(tp(+lead), 0)), lead = 1...6
max_wind_speed_ms = max(sqrt(u10(+lead)^2 + v10(+lead)^2)), lead = 1...6
```

Negative predicted hourly precipitation is clipped to its physical lower bound
before accumulation. ERA5 target precipitation is processed identically.

## Fixed event definitions

| Event | Definition |
|---|---|
| Rain | `tp_6h_mm > 10` |
| Heavy rain | `tp_6h_mm > 30` |
| Extreme rain | `tp_6h_mm > 50` |
| Strong wind | `max_wind_speed_ms > 15` |
| Extreme wind | `max_wind_speed_ms > 20` |
| Storm (OR interpretation) | heavy rain OR strong wind |
| Compound storm (AND interpretation) | heavy rain AND strong wind |
| Extreme weather | extreme rain OR extreme wind |

Both OR and AND storm definitions are retained because the original document
uses inconsistent wording in its training-label paragraph and severity table.
Reporting the two definitions makes the interpretation explicit instead of
silently choosing one after seeing the result.

## Metrics

For each model and event, the evaluator reports two categorical views:

- grid-cell POD, FAR, and CSI, measuring event-region localization;
- forecast-case POD, FAR, and CSI, where one case is positive if any sea cell
  exceeds the event definition during the six-hour window.

It additionally reports event-conditioned RMSE and peak-intensity bias for
`tp_6h_mm`, `max_wind_speed_ms`, or both, depending on the event.

## Windows command

From the repository root in the CUDA environment:

```text
python -u scripts/run_v9_1_pressure_experiment.py physical-events --device cuda
```

The command uses `configs/v9_2025_physical_event_extension.yaml` and writes
local artifacts to:

```text
artifacts/v9_2025_physical_event_extension/
```

The lightweight Git-ready result is published to:

```text
results/v9_2025_physical_event_extension/
```

The existing frozen percentile-event result is not overwritten. A bounded
pipeline check can be run directly with `evaluate_v9_2025_events.py`, the new
configuration, and `--max-batches 2`; bounded output is not a scientific
result.

## Scope boundary

This evaluator satisfies the report-level requirement to assess the original
physical events. It does not add a learned event-classification head and does
not use DPC Radar SRI or HRD confirmation. Those remain separate extensions if
the final course assessment requires joint regression-classification training
or radar-confirmed operational labels.
