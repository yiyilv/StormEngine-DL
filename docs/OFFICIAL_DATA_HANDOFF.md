# Official observation data handoff

## Purpose

This document hands off the next StormEngine-DL data-engineering stage. The
work must remain aligned with the project objective: turn sparse coastal
physical observations and model-derived marine support inputs into future
gridded weather fields over the Adriatic.

The current model flow is:

```text
12 hourly sparse point sets
    -> SetConv spatial encoder
    -> ConvGRU temporal processor
    -> CNN field decoder
    -> +1 ... +6 h fields on the 31 x 33 ERA5 Adriatic grid
```

The ERA5 rectangle is 39.0--46.5 N, 12.0--20.0 E at 0.25 degrees. It is the
download and prediction domain, not the physical-station selection geometry.
Physical inputs are selected with the shared 20 km Italian Adriatic coastal
corridor in `src/stormengine_dl/data/coastal_filter.py`.

## Do not change the scientific baseline silently

The frozen V6 experiment uses `configs/era5_2010_2017.yaml`:

- train: 2010--2015;
- validation: 2016;
- held-out test: 2017;
- input variables: `msl`, `u10`, `v10`, `i10fg`, `t2m`;
- targets: `msl`, `u10`, `v10`, `t2m`, `tp`;
- station profile: `dpc_plus_sea`;
- 239 physical coastal coordinates plus 151 virtual sea coordinates.

Training currently samples ERA5 at every enabled coordinate. It does not yet
train on true DPC/MeteoHub observations. The one-week official snapshot in this
handoff is for schema validation, operational-adapter development, coverage
analysis, and later calibration. It is not long enough to replace the ERA5
training archive.

The frozen V6 checkpoint and baseline results must remain comparable. Any
change to point channels, masks, or source features belongs in a new experiment
configuration and requires retraining.

## Repository and branch state

Repository:

```text
/Users/a1-6/CoursesCode/Sem4/GeoProject/StormEngine-DL
```

The handoff is prepared on `component-diagnostics`. Before changing anything,
inspect `git status` and preserve all existing user changes. Large data under
`data_external/` are intentionally ignored and must never be committed.

Relevant files:

```text
data/stations_registry.csv
data/stations_registry.meta.json
src/stormengine_dl/data/coastal_filter.py
src/stormengine_dl/data/official_observations.py
scripts/audit_official_observations.py
tests/test_official_observations.py
```

## Station design

The default `dpc_plus_sea` profile contains:

- 239 official-source physical coastal coordinates;
- 151 virtual Adriatic support coordinates;
- 390 total fixed input coordinates.

The virtual points are not physical stations. Their present pretraining values
are ERA5 samples; the planned operational values are Open-Meteo model-derived
marine inputs. Keep an explicit physical/virtual source distinction. Disabled
legacy `ARTA_VIRTUAL`, `ARPAM_VIRTUAL`, and `ARPA_PUGLIA_VIRTUAL` rows are
provenance only and must not be re-enabled as physical observations.

Offline stations should not be deleted from the fixed registry merely because
they are absent during one week. Registry membership and time-varying
availability are separate concepts.

## Frozen official snapshot

All production extracts below use the same half-open operational interval:

```text
2026-08-01 00:00:00 UTC <= time <= 2026-08-08 00:00:00 UTC
```

The portal requests retained all products, levels, and timerange semantics,
selected JSON output, and enabled the MeteoHub quality-control filter. The only
exception is the FVG `no_qc` control file.

Raw files are external to Git:

```text
data_external/meteohub/raw/20260801_20260808/
```

The authoritative filenames, sizes, line counts, hashes, network membership,
and request names are in
`data/manifests/meteohub_20260801_20260808.json`.

### Validated coverage

ARPAFVG:

- 154 stations in the extract;
- all 22/22 project coastal coordinates observed;
- project coverage: pressure 6, wind direction 13, wind speed 14, gust 14,
  temperature 13, humidity 11, precipitation 18.

The FVG QC comparison retained 77,729 of 77,734 variable measurements
(99.9936%). It removed five simultaneous zero values at `Monte Lussari sm` on
2026-08-03 08:00 UTC: pressure, humidity, wind direction, wind speed, and gust.
No selected FVG coastal observation was removed.

Central/Emilia-Romagna batch:

```text
dpcn-marche 54/54
dpcn-molise 4/4
agrmet 3/3
boa 2/6
simnbo 14/14
spdsra 21/21
urbane 1/1
marefe 9/9
```

The only missing registered coordinates are four BOA Cervia/Cattolica port or
radar stations. The current BOA extract contains `Calipso` and `Nausicaa 2`.
Record this as online status; do not delete the four coordinates from the
registry without evidence that they have been retired.

Puglia:

- 157 stations in the extract;
- all 61/61 project coastal coordinates observed;
- among project coordinates: temperature 56, precipitation 58, humidity 29,
  wind speed 11, wind direction 10;
- the dataset provides no `B10004` pressure or `B11041` gust in this window.

Veneto:

- 214 stations in the extract;
- all 22/22 project coastal coordinates observed;
- among project coordinates: temperature 15, precipitation 15, humidity 14,
  wind speed 9, wind direction 9, gust 2, pressure 0.

The wider Veneto network has pressure sensors, but none of the 22 selected
coastal coordinates supplies pressure in this snapshot.

Across MeteoHub-backed project networks, 213 of 217 registered coordinates are
observed. The remaining 22 physical project coordinates are the separate
Abruzzo Polaris subset and have no observation file in this snapshot.

## Corrected June interpretation

The June audit originally attributed the missing FVG data to a possible
quality-filter effect. It has now been corrected because later portal probes
showed:

```text
2026-06-15--22: 0 FVG records
2026-07-01--08: 0 FVG records
FVG supply begins around 2026-07-27
```

The June absence is therefore a historical-availability limitation, not a QC
rejection. Keep the old audit as a dated snapshot with this corrected explanation.

## Existing BUFR mapping

`official_observations.py` currently maps:

```text
B10004 pressure (Pa -> hPa)
B11001 wind direction (degree)
B11002 wind speed (m s-1)
B11041 maximum wind gust (m s-1)
B11043 gust direction (degree)
B12101 air temperature (K -> degree C)
B13003 relative humidity (%)
B13011 precipitation amount (mm equivalent)
B13013 snow depth (m)
B13215 river level (m)
B14198 downward global visible irradiance (W m-2)
```

Unmapped descriptors must remain in normalized output with their original
timerange and level metadata. Do not silently discard them.

## Critical scientific constraints

### Pressure is not automatically ERA5 MSL

`B10004` is station-level pressure and cannot be relabelled as ERA5 `msl`
without a documented sea-level correction. Retain
`station_pressure_hpa`, elevation, and source metadata. If an MSL estimate is
implemented, emit both original and corrected values and validate the formula.

### Convert wind direction before aggregation

Use the meteorological convention:

```text
u = -speed * sin(direction)
v = -speed * cos(direction)
```

Do not average direction angles directly. Convert to vectors first.

### Preserve precipitation timerange semantics

Use `timerange_indicator`, start/end seconds, and `aggregation_seconds`.
Prefer explicit one-hour accumulations. Shorter non-overlapping accumulations
may be combined; overlapping or ambiguous intervals must be rejected or masked.
Never double-count cumulative reports.

### Prevent future leakage

For an hourly feature at time `t`, do not use an observation after `t`.
Implement a configurable latest-at-or-before rule with a maximum age (initially
60 minutes), and output source timestamp and observation age. Hourly gust may
use the preceding-hour maximum when its timerange semantics support that rule.

### A station mask is insufficient

The current encoder accepts `point_mask: [B,T,N]`, but real observations are
missing per variable. Puglia can have temperature and precipitation without
pressure or gust. Filling missing values with zero while marking the station
valid is scientifically wrong.

The observation adapter should produce at least:

```text
values           [T,N,C]
value_mask       [T,N,C]
observation_age  [T,N,C]
station_present  [T,N]
coordinates      [N,2]
station_static   [N,F]
source_type      [N or T,N]
```

If the encoder consumes variable masks/age as channels, that changes the model
input contract and invalidates old checkpoints. Keep the frozen V6 baseline and
train a new experiment.

## Required implementation sequence

1. Validate every raw file against the versioned manifest.
2. Fix audit import hygiene. Running the official JSON audit should not require
   importing PyTorch, and Shapely should only be required for spatial filtering.
3. Generate `data/audits/meteohub_official_20260801_20260808/` with network,
   station, variable, timerange, project-coverage, and QC-comparison tables.
4. Preserve the corrected June FVG explanation when regenerating audit outputs.
5. Produce a normalized long-form observation table with station, coordinate,
   elevation, UTC time, BUFR code, canonical variable/value/unit, timerange,
   level, QC status, and source file.
6. Deduplicate on station, time, BUFR code, timerange, and level. Later official
   revisions may replace earlier values, but revisions must be counted.
7. Reapply the shared coastal polygon at measurement level.
8. Implement no-future-leak hourly alignment, wind vector conversion, and
   timerange-aware precipitation.
9. Construct fixed-registry tensors with per-variable masks and ages.
10. Add unit, leakage, timerange, fixed-order, partial-variable, BOA-offline,
    Puglia-no-pressure, and Veneto-no-coastal-pressure tests.

## Acceptance criteria

- all five files match manifest sizes, line counts, and SHA-256 hashes;
- every JSONL line parses;
- the audit runs in a lightweight data environment;
- 213/217 MeteoHub-backed project coordinates are identified;
- four BOA absences and 22 Abruzzo observation gaps are explicit;
- raw station pressure is never silently presented as MSL;
- wind direction is converted using meteorological vector conventions;
- precipitation respects source accumulation windows;
- no value after an hourly target time is used;
- fixed station ordering comes from `stations_registry.csv`;
- partial variables are represented with `[T,N,C]` masks;
- no unmasked placeholder zero or NaN reaches the model;
- raw data remain outside Git;
- all relevant tests pass;
- model-interface changes are documented as checkpoint-incompatible and use a
  new experiment configuration.

## Deferred work

- obtain official Abruzzo observation values for the 22 selected Polaris
  coordinates and, if possible, the complete stated 119-station network;
- obtain historical Open-Meteo/model-run values at the 151 marine support
  coordinates;
- quantify ERA5-to-DPC and ERA5-to-Open-Meteo distribution shifts;
- compare real-only with real-plus-virtual marine experiments;
- evaluate adding precipitation as a sparse input in a controlled retraining;
- keep `ssrd` optional because official radiation coverage is sparse.
