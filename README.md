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

The currently downloaded pilot data cover 2010, 2011, and January-February
2012. `configs/pilot.yaml` therefore uses 2010 for training, 2011 for
validation, and the available 2012 months for testing. Check the complete
390-coordinate input path before training with:

```bash
PYTHONPATH=src ../StormEngine/stormengine-env/bin/python \
  scripts/check_data_pipeline.py --config configs/pilot.yaml
```

`configs/base.yaml` records the intended final year split and requires the
remaining ERA5 months to be downloaded before it can be used.

## Station profiles

`data/stations_registry.csv` keeps physical land stations, virtual Adriatic sea
coordinates, and disabled legacy coastal anchors in one traceable catalog. The
land profile is now built from the versioned official-source snapshot
`data/source_snapshots/official_station_catalog_2026-08-07.csv`, rather than the
earlier 485-coordinate team aggregate. Three profiles support controlled experiments:

- `land_only`: all available DPC/regional physical stations in the domain
- `sea_only`: virtual Adriatic coordinates, intended for Open-Meteo variables
- `dpc_plus_sea`: the combined operational design and default configuration

Legacy `ARTA_VIRTUAL`, `ARPAM_VIRTUAL`, and `ARPA_PUGLIA_VIRTUAL` rows remain in
the catalog for provenance but are disabled. They are never presented as DPC
physical observations.

The official source snapshot contains 1,377 candidate physical coordinates
inside the ERA5 rectangle: 1,330 stations observed in three official MeteoHub
query windows and 47 official Polaris-linked Abruzzo stations. The training
registry then applies a shared Shapely polygon representing a 20 km corridor
around the Italian Adriatic shoreline. It retains 239 coastal physical stations.
Virtual support nodes are selected separately: the central Adriatic points and
the Slovenian, Croatian, Montenegrin, and northern Albanian shore are retained,
while Greece, the Ionian group, and Albania south of 40.45 N are excluded. This
retains 151 virtual support coordinates, so the default `dpc_plus_sea` profile
contains 390 points. The full rectangle remains an ERA5 download/grid extent; it is not
used as the station-selection geometry.

This count is deliberately described as an official-source snapshot, not a
claim that every installed station was online. MeteoHub's observations endpoint
only reveals stations that returned data in a selected time window. Abruzzo is
also a documented gap: the regional authority states that its complete
telemetered network contains 119 stations, while the accessible official API
exposes 47 Polaris-linked coordinates and the public network-map link currently
redirects to login. The exact queries, hashes, counts, and limitations are in
`data/source_snapshots/official_station_catalog_2026-08-07.meta.json`.

The coastal rule follows the original DPC workflow at two independent stages:

1. `scripts/build_station_registry.py` filters station metadata before a station
   is admitted to a training profile.
2. `scripts/filter_coastal_observations.py` reapplies the same point-in-polygon
   test to every real-time measurement coordinate before values enter the model.

The shoreline vertices, polygon construction, and version identifier live in
`src/stormengine_dl/data/coastal_filter.py`. Counts and the selected buffer are
recorded in `data/stations_registry.meta.json`.

Apply the measurement-level check with:

```bash
PYTHONPATH=src ../StormEngine/stormengine-env/bin/python \
  scripts/filter_coastal_observations.py \
  --input observations.csv \
  --output observations_coastal.csv
```

Rebuild the compact catalog from downloaded official API responses with:

```bash
PYTHONPATH=src ../StormEngine/stormengine-env/bin/python \
  scripts/build_official_station_catalog.py \
  --meteohub-json snapshot-00.json snapshot-06.json snapshot-11.json \
  --abruzzo-json abruzzo-stations.json \
  --output data/source_snapshots/official_station_catalog_2026-08-07.csv
```

## Current status

The repository now provides the tensor-level Encoder-Processor-Decoder skeleton,
a validated ERA5 manifest, cross-month sequence loading, and a traceable
official-source station snapshot. The remaining station-data task is to obtain
the non-public portion of the current Abruzzo station registry; model work can
continue using the explicitly labelled public subset. The next pipeline
milestones are train-only normalization, training/baseline evaluation, and
inference.

## Quick smoke test

Using the existing StormEngine virtual environment:

```bash
PYTHONPATH=src ../StormEngine/stormengine-env/bin/python -m unittest discover -s tests -v
```
