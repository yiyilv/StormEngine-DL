# StormEngine V9.2 Final16 model card

## Intended use

V9.2 Final16 is the frozen production candidate for six-hour Adriatic weather
grid forecasting. It consumes the preceding 12 hourly observations and predicts
`msl`, `u10`, `v10`, `t2m`, and hourly `tp` for lead hours +1 through +6 on the
fixed 31 x 33 project grid.

The model is intended for research evaluation and controlled operational replay.
It is not a public-warning system and must not be treated as the sole source for
safety-critical decisions.

## Frozen input contract

- 239 physical DPC/MeteoHub coastal stations.
- 151 Open-Meteo ICON-2I marine support points.
- Fixed station order from `data/stations_registry.csv`.
- Input variables, in order: `msl,u10,v10,i10fg,t2m,tp`.
- Every variable supplies a normalized value, a per-variable validity mask, and
  an observation age.
- DPC station pressure is reduced to MSL with the documented temperature,
  humidity, elevation, and fallback rules; it is never silently treated as MSL.
- Missing values are zero-filled only after normalization and always have mask 0.
- No observation after the forecast origin may enter its 12-hour history.

## Training

- Training period: 2010--2025 inclusive.
- Training windows: 46,747.
- Fixed production refit: one complete epoch, selected before the all-year run
  from the frozen development schedule.
- Event-aware loss emphasizes six-hour sea wind and precipitation events.
- Final checkpoint SHA-256:
  `ea0afde51397bd88bc6aeca45e452cabadfd956821112c6828eb826fdc03e86f`.

All 16 years enter the final refit. Consequently, the production checkpoint has
no internal holdout and makes no independent accuracy claim by itself.

## Evidence

- The 2018 development holdout supplies the independent evidence that V9.2
  improves event detection relative to V9.1.
- The 2026-08-01--08 ERA5T evaluation tests ordinary operational inputs and
  field error; that week contains no frozen-threshold sea event.
- The 2026-08-16--19 marine-only test masks every physical station and verifies
  numerical stability under total DPC network outage. It has no target truth.

V9.2 improves event sensitivity but can increase false alarms. Its ordinary
field results are mixed rather than uniformly better than V9.1. Under complete
physical-network outage it remains stable but produces conservative event
amplitudes.

## Deployment

The checkpoint remains a local external artifact under:

```text
artifacts/v9_2_event_aware_final16/seed_42/final.pt
```

Generate and verify the immutable dependency manifest with:

```powershell
python -u scripts\freeze_v9_2_final16.py
```

Run target-free prediction with `scripts/predict_v9_2_final16.py`. The command
requires four aligned tensors: physical observations, corrected physical MSL,
marine support variables, and marine MSL. It writes a compressed forecast NPZ
and a JSON provenance record. Prediction output is a local operational artifact
and is not committed to ordinary Git. Hourly precipitation is clipped to its
physical lower bound of zero at the production output boundary; the other
channels are not silently clipped.

## Known limitations

- Six-hour horizon only; the model does not implement the later 48-hour goal.
- MSL quality depends on sparse physical pressure anchors and Open-Meteo marine
  pressure support.
- DPC variable coverage is heterogeneous and varies by network and sensor type.
- Open-Meteo is model-derived support, not a physical observation network.
- Event skill outside the evaluated periods remains uncertain.
- Forecasts require monitoring for source outages, distribution shift, missing
  pressure support, and implausible spatial fields.
