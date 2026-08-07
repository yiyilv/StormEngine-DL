# StormEngine-DL

End-to-end deep-learning pipeline for short-range weather forecasting over the
Adriatic domain. The model converts sparse observations at weather-station
coordinates into future gridded atmospheric fields.

## Model flow

```text
past station sequences
    -> SetConv spatial encoder
    -> ConvGRU temporal processor
    -> CNN field decoder
    -> future gridded weather fields
```

The initial supported grid is the native ERA5 Adriatic grid:

- domain: 39.0-46.5 N, 12.0-20.0 E
- resolution: 0.25 degrees
- shape: 31 x 33
- history: 12 hourly steps
- forecast horizon: configurable, initially 6 and later 48 hours

## Repository layout

```text
configs/                 experiment configuration
data/                    versioned metadata only; large arrays stay external
src/stormengine_dl/      reusable pipeline code
tests/                   smoke and shape tests
```

Large ERA5 files are not committed. The default config points to the sibling
`../DownloadDate` directory and can be overridden for another machine.

## Current status

The first milestone provides a runnable tensor-level skeleton for the complete
Encoder-Processor-Decoder path. Data loading, station-registry validation,
training, baselines, evaluation, and inference will be added incrementally.

## Quick smoke test

Using the existing StormEngine virtual environment:

```bash
PYTHONPATH=src ../StormEngine/stormengine-env/bin/python -m unittest discover -s tests -v
```

