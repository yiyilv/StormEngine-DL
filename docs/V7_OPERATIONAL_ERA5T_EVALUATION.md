# V7 operational ERA5T evaluation

## Purpose

This evaluation closes the gap left by the 2026 operational replays. It compares
the two frozen model systems on the same 152 real-input windows and the same
future ERA5T target grids:

- V7-A: 239 physical DPC/MeteoHub stations;
- V7-B: the same 239 stations plus 151 model-derived Open-Meteo marine points;
- target: same-period ERA5T `msl,u10,v10,t2m,tp` grids;
- history: 12 hours;
- forecast: +1 through +6 hours;
- domain: 31 x 33 Adriatic grid.

The primary comparison is V7-B versus V7-A, especially sea-domain RMSE. Dense
ERA5T persistence is also reported, but it uses a complete analysis grid and is
therefore a stronger-information diagnostic rather than a deployment-equivalent
baseline.

## Required local-only artifacts

These files are intentionally not stored in Git:

```text
artifacts/v7_a_2010_2017/best.pt
artifacts/v7_b_2010_2017/best.pt
data_external/meteohub/processed/20260801_20260808/official_hourly_physical.npz
data_external/open_meteo/processed/20260801_20260808/icon2i_hourly_marine.npz
<ERA5T>/era5t_adriatic_20260801_20260808_instant.nc
<ERA5T>/era5t_adriatic_20260801_20260808_accum.nc
```

Checkpoint and input hashes are recorded in the two frozen manifests under
`results/v7_a_2010_2017_frozen/` and `results/v7_b_2010_2017_frozen/`.

The verified ERA5T files contain 192 continuous hourly grids from
2026-08-01 00:00 through 2026-08-08 23:00 UTC. Only the exact timestamps needed
by the common DPC/Open-Meteo window are selected; no approximate or future-time
matching is allowed.

## Windows run

From PowerShell in the repository root:

```powershell
git fetch origin
git switch era5t-evaluation
git pull --ff-only origin era5t-evaluation

conda activate stormengine

python -u scripts\evaluate_v7_operational_era5t.py `
  --era5t-instant "D:\Documents\py_projects\StormEngine-DL\DownloadDate\era5t_20260801_20260808\era5t_adriatic_20260801_20260808_instant.nc" `
  --era5t-accum "D:\Documents\py_projects\StormEngine-DL\DownloadDate\era5t_20260801_20260808\era5t_adriatic_20260801_20260808_accum.nc" `
  --device cuda
```

The script validates checkpoint contracts, station order, common timestamps,
grid coordinates, latitude orientation, units, and finite outputs before saving:

```text
artifacts/v7_operational_era5t_20260801_20260808/metrics.json
```

Positive `rmse_skill.v7_b_over_v7_a` means V7-B has lower RMSE. Results are
reported for full, land, and sea domains and separately for leads +1 to +6.

## Interpretation limit

This is a same-period historical operational-input evaluation. It can quantify
whether the complete V7-B marine-support pathway improves over V7-A against
ERA5T. It does not turn Open-Meteo into physical observations and does not by
itself identify how much improvement comes from point placement, model source,
or calibration. Those require separate ablations.
