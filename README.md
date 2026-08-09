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

The validated local archive now covers every hour from 2010 through 2017: 96
complete monthly pairs, 70,128 hourly steps, one consistent 31 x 33 grid, and
no NaN values. `configs/era5_2010_2017.yaml` uses 2010-2015 for training, 2016
for validation, and 2017 as a held-out test year. `configs/pilot.yaml` remains
as the smaller 2010/2011/2012 development split. Check the complete
390-coordinate input path with:

```bash
PYTHONPATH=src ../StormEngine/stormengine-env/bin/python \
  scripts/build_training_cache.py --config configs/era5_2010_2017.yaml

PYTHONPATH=src ../StormEngine/stormengine-env/bin/python \
  scripts/check_data_pipeline.py --config configs/era5_2010_2017.yaml
```

The first command is a one-time preprocessing step. It converts the 96 monthly
NetCDF pairs into normalized hourly station inputs and dense target grids stored
as NumPy memory-mapped arrays. The cache is about 2 GB, stays beside the ERA5
archive under `DownloadDate/cache/stormengine_2010_2017`, and is not committed
to Git. Training refuses to fall back to repeated per-window NetCDF work when
this configured cache is missing, preventing an accidentally very slow run.

`configs/base.yaml` records the intended final year split and requires the
remaining ERA5 months to be downloaded before it can be used.

V6-style preprocessing is now reproducible outside the notebook. Statistics
are fitted on the training years only, avoiding the full-dataset leakage in the
exploratory V6 normalizer. The decoder receives a Natural Earth 10m land-sea
mask and a metric-aware normalized distance-to-nearest-input field. Generate
the pilot artifacts with:

```bash
PYTHONPATH=src ../StormEngine/stormengine-env/bin/python \
  scripts/fit_normalization.py --config configs/pilot.yaml

PYTHONPATH=src ../StormEngine/stormengine-env/bin/python \
  scripts/build_static_fields.py --config configs/pilot.yaml
```

Cartopy is needed only to rebuild the Natural Earth mask; install the optional
dependency with `pip install -e '.[static]'`. Runtime loading uses the compact
versioned NPZ file and does not require Cartopy.

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

The repository now provides the Encoder-Processor-Decoder model, validated
cross-month ERA5 sequences, train-only normalization, reproducible static
fields, a traceable official-source station snapshot, and a resumable V6
training/evaluation entry point. The remaining station-data task is to obtain
the non-public portion of the current Abruzzo station registry. The next model
milestones are a full CUDA pilot run, baseline comparison, and operational
inference with real-time observations.

## V6 end-to-end training

The new V6 keeps the original mean-normalized sea-weighted MSE definition
(`sea_weight: 2.0`) and adds chronological train/validation/test years,
mixed-precision CUDA training, gradient clipping, ReduceLROnPlateau, early
stopping, resumable `last.pt`, best-model `best.pt`, and denormalized MAE/RMSE
for the full grid, land, and sea. Training selects checkpoints only with the
validation years; it does not inspect the held-out test set by default. Run it
from the repository root with:

```bash
python scripts/train.py --config configs/era5_2010_2017.yaml --device cuda
```

Resume an interrupted Windows run with:

```bash
python scripts/train.py --config configs/era5_2010_2017.yaml --device cuda \
  --resume artifacts/v6_2010_2017/last.pt
```

Inspect every forecast lead on validation data while developing the model:

```bash
python scripts/evaluate.py \
  --config configs/era5_2010_2017.yaml \
  --checkpoint artifacts/v6_2010_2017/best.pt \
  --split validation --device cuda
```

Only after the architecture and hyperparameters are frozen, replace
`--split validation` with `--split test`. The evaluator writes aggregate and
lead-hour MAE/RMSE for the full grid, land, and sea, plus representative
denormalized forecast/target arrays. `notebooks/StormEngine_V6_Evaluation.ipynb`
plots these metrics and example error maps. `train.py --evaluate-test` remains
available only for isolated pipeline smoke checks.

On Windows, keep `num_workers: 0`; this avoids Jupyter multiprocessing spawn
issues. The RTX 4060 starting batch size is 16. If CUDA runs out of memory,
reduce `batch_size` to 8 or 4 without changing the data split or model. Before
a full run, use the notebook's isolated smoke command (two training batches and
one evaluation batch), then its bounded 2010-2012 pilot (200 training batches
per epoch for three epochs). These are pipeline and speed checks, not scientific
results. Only after both complete should the full 2010-2017 cell be started.
The training command prints batch progress, elapsed time, and ETA throughout
each epoch.

`notebooks/StormEngine_V6_EndToEnd.ipynb` performs the cache build, preflight,
smoke run, medium pilot, full resumable training, and validation inspection in
that order. Its subprocess output is streamed live in Jupyter. It writes local
configuration files ignored by Git, so machine-specific drive paths are not
committed.

### Persistence baselines

After freezing the V6 test result, evaluate two non-learned references on exactly
the same windows and land/sea masks:

```bash
python scripts/evaluate_baselines.py \
  --config configs/era5_2010_2017.yaml \
  --split test \
  --v6-metrics results/v6_2010_2017_baseline/metrics_by_lead.json \
  --output-dir artifacts/v6_2010_2017/baselines_test
```

`dense_grid_persistence` repeats the last complete ERA5 grid and is deliberately
a strong reference with more information than V6. `sparse_idw_persistence` uses
only the last-hour values at the same enabled coordinates as V6, interpolates the
four shared target variables with metric-aware inverse-distance weighting, and
uses a zero-precipitation forecast because `tp` is not an input variable. Positive
skill (`1 - V6_RMSE / baseline_RMSE`) means V6 improves on that reference. The
same workflow is available in `notebooks/StormEngine_V6_Baselines.ipynb`.

## Quick smoke test

Using the existing StormEngine virtual environment:

```bash
PYTHONPATH=src ../StormEngine/stormengine-env/bin/python -m unittest discover -s tests -v
```
