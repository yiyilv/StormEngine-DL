# V7 operational ERA5T evaluation, 2026-08-01 to 2026-08-08

This is the first same-window gridded accuracy evaluation of the frozen V7-A
and V7-B models using real operational inputs. Both models forecast the same
152 windows and are evaluated against the same independent ERA5T target grids.

## Evaluation contract

- Input period: 2026-08-01 00:00 through 2026-08-08 00:00.
- Analysis windows: 2026-08-01 11:00 through 2026-08-07 18:00.
- History: 12 hours.
- Forecast leads: +1 through +6 hours.
- Windows: 152/152 completed on CUDA.
- V7-A input: 239 physical DPC/MeteoHub stations.
- V7-B input: the same 239 physical stations plus 151 model-derived
  Open-Meteo marine points.
- Target: ERA5T `msl,u10,v10,t2m,tp` on the Adriatic 31x33 grid.

The comparison measures the value of the complete V7-B marine-support pathway.
It does not treat Open-Meteo as in-situ truth. Dense ERA5T persistence repeats
the last complete target grid and therefore has a stronger information set than
the deployable sparse-input models.

## Aggregate full-domain RMSE

| variable | V7-A | V7-B | V7-B skill over V7-A |
|---|---:|---:|---:|
| msl | 1.7576 | 1.9541 | -11.18% |
| u10 | 1.5791 | 1.3500 | +14.51% |
| v10 | 1.8313 | 1.3988 | +23.62% |
| t2m | 2.7020 | 2.1166 | +21.67% |
| tp | 0.1193 | 0.1194 | -0.10% |

## Sea-domain result

V7-B improved sea-domain RMSE relative to V7-A by 3.58% for `msl`, 17.77% for
`u10`, 30.48% for `v10`, 23.61% for `t2m`, and 10.88% for `tp`. This confirms
that the marine-support pathway materially improves the intended sea-domain
forecast, especially wind.

The main regression is land-domain `msl` (-21.33% skill versus V7-A). Neither
V7 model receives pressure as an input, so pressure should be treated as a
separate follow-up design problem rather than evidence that the marine wind and
temperature pathway failed.

## Artifacts

- `metrics.json`: complete MAE/RMSE results for V7-A, V7-B, and dense ERA5T
  persistence, split by full/land/sea and by lead hour.
- `manifest.json`: exact code, model, target-file, and result hashes.

The ERA5T NetCDF files and model checkpoints remain local and are not committed
to ordinary Git.
