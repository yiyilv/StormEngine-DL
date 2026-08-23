# V9.2 independent 2026 event validation

## Purpose

The 2026-08-01--08 operational evaluation contained no ERA5T sea events at the
frozen V9.2 thresholds. It tested operational stability and ordinary field
error, but it could not test the event-aware objective. The next evaluation is
therefore a separate, post-training event window.

## Frozen candidate window

- Input/target endpoint period: **2026-08-16 00:00 through 2026-08-19 00:00 UTC**
- Primary event date: **2026-08-17**
- Training remains frozen at 2010--2025; this period is not used for fitting or
  model selection.
- The date was selected before seeing ERA5T targets. [ANSA reporting published
  on 17 August 2026](https://www.ansa.it/english/newswire/english_service/2026/08/17/italy-to-see-more-thunderstorms-temperatures-to-drop-across-many-regions-3_785b67ef-587c-48e0-8648-d0de47885547.html)
  described thunderstorms affecting northeast Italy and the Adriatic coast on
  17 August.

The downloaded ICON-2I/Open-Meteo support data confirms that the period is a
useful operational stress test, without being used as target truth:

- maximum marine-point gust: 31.2 m/s;
- 9 hourly valid times contain at least one marine point with gust >= 20 m/s;
- maximum marine-point hourly precipitation: 37.6 mm;
- 14 hourly valid times contain at least one marine point with precipitation
  >= 5 mm.

These are model-derived input diagnostics. Event labels and prediction scores
must come only from the independently downloaded ERA5T target grids.

## Prepared Open-Meteo data

Raw and processed arrays are intentionally outside Git:

```text
data_external/open_meteo/raw/20260816_20260819/icon2i/
data_external/open_meteo/processed/20260816_20260819/
  icon2i_hourly_marine.npz
  icon2i_hourly_marine_pressure.npz
```

The Git-tracked manifests and audit tables record checksums, coverage, time
alignment, and coordinate displacement. All 151 points and all six input
variables have complete coverage for the 73 retained hourly endpoints.

## Remaining external inputs

### MeteoHub/DPC portal export

Export the same QC-approved networks and variables used in the official handoff
for **2026-08-16 00:00 through 2026-08-19 00:00 UTC**. Keep each portal response
as JSON/JSONL and do not manually edit observations. The required contract
variables are wind speed/direction, gust, temperature, precipitation, station
pressure, and relative humidity where available. The fixed 239-station registry
and the existing station elevation audit remain authoritative.

Build the physical tensor with:

```powershell
python -u scripts\build_operational_tensors.py <QC_JSON_FILES...> `
  --from-utc 2026-08-16T00 `
  --to-utc 2026-08-19T00 `
  --output data_external\meteohub\processed\20260816_20260819\official_hourly_physical.npz
```

### ERA5T targets

Download instantaneous `msl,u10,v10,t2m` and accumulated `tp` over the same
project grid and endpoint period. Do not use ERA5T to fill the operational
inputs. Store the two NetCDF files under:

```text
..\DownloadDate\era5t_20260816_20260819\
  era5t_adriatic_20260816_20260819_instant.nc
  era5t_adriatic_20260816_20260819_accum.nc
```

The current Windows environment does not have `cdsapi` or a `.cdsapirc`, so the
ERA5T download requires CDS setup or externally supplied files.

## Evaluation rule

After DPC and ERA5T are present, build corrected DPC MSL, then run the frozen
V9.2 evaluator with explicit paths. Report field RMSE/MAE and event POD/FAR/CSI
against ERA5T. If ERA5T still contains no frozen-threshold sea events, record
that fact and select a new period without changing thresholds based on model
predictions.
