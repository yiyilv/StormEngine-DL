# V7-B ERA5T proxy-input counterfactual

## Purpose

The operational V7-B RMSE against ERA5T contains both model error and the
effect of replacing ERA5 training-style point values with real DPC and
Open-Meteo inputs. This single controlled counterfactual estimates the combined
operational input-system penalty without introducing a large experiment grid.

```text
Published run: DPC + Open-Meteo -> frozen V7-B -> ERA5T future grid
Counterfactual: ERA5T at the same 390 points -> same V7-B -> same future grid
```

Both runs use the same checkpoint, coordinates, 12-hour histories, +1 through
+6-hour targets, 152 windows, target grids, land-sea mask, and metrics. The
proxy input has complete masks and zero observation age, representing the
idealized in-distribution input available during ERA5-based training.

The primary quantity is:

```text
excess_mse = operational_real_input_mse - era5t_proxy_input_mse
```

Positive excess MSE indicates a combined deployment-input penalty. It includes
source mismatch, missingness, age, and point-versus-grid representativeness. It
is a controlled counterfactual effect, not an additive causal decomposition of
DPC and Open-Meteo separately.

## Windows run

```powershell
git fetch origin
git switch era5t-evaluation
git pull --ff-only origin era5t-evaluation

conda activate stormengine

python -u scripts\evaluate_v7_b_era5t_proxy.py `
  --era5t-instant "D:\Documents\py_projects\StormEngine-DL\DownloadDate\era5t_20260801_20260808\era5t_adriatic_20260801_20260808_instant.nc" `
  --era5t-accum "D:\Documents\py_projects\StormEngine-DL\DownloadDate\era5t_20260801_20260808\era5t_adriatic_20260801_20260808_accum.nc" `
  --device cuda
```

The output is:

```text
artifacts/v7_b_era5t_proxy_20260801_20260808/metrics.json
```

The script compares it directly with the already published operational V7-B
metrics under `results/v7_operational_era5t_20260801_20260808/metrics.json` and
reports full/land/sea aggregate and +1 through +6-hour penalties.
