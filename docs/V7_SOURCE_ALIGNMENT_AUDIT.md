# V7 source alignment audit

## Question

The operational V7-B evaluation shows lower ERA5T RMSE than V7-A over the sea,
but DPC, Open-Meteo ICON-2I, and ERA5T are different observing/modelling
systems. This audit measures those source differences before attributing all of
the V7-B gain to the neural model.

For every common hour from 2026-08-01 00:00 through 2026-08-08 00:00 it
compares, at identical coordinates:

```text
DPC physical observations - bilinearly sampled ERA5T
Open-Meteo ICON-2I values - bilinearly sampled ERA5T
```

The variables are `u10,v10,i10fg,t2m,tp`. Mean-sea-level pressure is excluded
because DPC contains station pressure, which is not directly comparable.

DPC statistics are reported for all valid cells and for observations no more
than five minutes old. Open-Meteo is evaluated both at the coordinates supplied
to V7-B and at the provider-returned model-grid coordinates, separating
cell-selection displacement from value differences.

Outputs include count, bias, MAE, RMSE, correlation, mean, standard deviation,
and 5th/50th/95th percentiles for each source and ERA5T.

## Windows run

From PowerShell in the repository root:

```powershell
git fetch origin
git switch era5t-evaluation
git pull --ff-only origin era5t-evaluation

conda activate stormengine

python -u scripts\audit_v7_source_alignment.py `
  --era5t-instant "D:\Documents\py_projects\StormEngine-DL\DownloadDate\era5t_20260801_20260808\era5t_adriatic_20260801_20260808_instant.nc" `
  --era5t-accum "D:\Documents\py_projects\StormEngine-DL\DownloadDate\era5t_20260801_20260808\era5t_adriatic_20260801_20260808_accum.nc"
```

This is a CPU data audit, not training. The result is written to:

```text
artifacts/v7_source_alignment_20260801_20260808/source_alignment.json
```

## Interpretation

The DPC and Open-Meteo errors must not be treated as a controlled head-to-head
ranking because they occupy different physical/coastal and marine locations.
ERA5T is also a gridded reanalysis reference rather than error-free truth. The
audit is used to quantify distribution shift and qualify the V7-A/V7-B model
comparison, not to select a universally superior data provider.
