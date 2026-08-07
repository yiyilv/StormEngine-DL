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

## ERA5 data validation

Build a manifest before constructing training sequences:

```bash
PYTHONPATH=src ../StormEngine/stormengine-env/bin/python scripts/build_era5_manifest.py \
  --root ../DownloadDate \
  --output data/manifests/era5_manifest.csv \
  --deep
```

The manifest retains incomplete months and records missing variables, unexpected
hour counts, time discontinuities, mismatched grids, read errors, and NaN values.
Training code must consume only rows where `valid` is true.

`Era5SequenceDataset` then builds continuous history/forecast windows across
month boundaries. For the first reproducible experiment it samples ERA5 fields
at verified DPC station coordinates to form sparse inputs and uses the complete
ERA5 grid as the target. It converts pressure to hPa, temperature to degrees C,
precipitation to mm, and hourly solar-radiation energy to W/m2. Monthly arrays
are loaded lazily with a bounded cache.

## Station profiles

`data/stations_registry.csv` keeps physical land stations, virtual Adriatic sea
coordinates, and disabled legacy coastal anchors in one traceable catalog. Three
profiles support controlled experiments:

- `land_only`: all available DPC/regional physical stations in the domain
- `sea_only`: virtual Adriatic coordinates, intended for Open-Meteo variables
- `dpc_plus_sea`: the combined operational design and default configuration

Legacy `ARTA_VIRTUAL`, `ARPAM_VIRTUAL`, and `ARPA_PUGLIA_VIRTUAL` rows remain in
the catalog for provenance but are disabled. They are never presented as DPC
physical observations.

The registry also recovers physical `dpcn-puglia` stations directly from the
raw MeteoHub JSON because the earlier aggregated CSV omitted that network.

## Current status

The repository now provides the complete tensor-level Encoder-Processor-Decoder
skeleton, a validated ERA5 manifest, and cross-month sequence loading. The next
milestones are the verified station registry, train-only normalization,
training/baseline evaluation, and inference.

## Quick smoke test

Using the existing StormEngine virtual environment:

```bash
PYTHONPATH=src ../StormEngine/stormengine-env/bin/python -m unittest discover -s tests -v
```
