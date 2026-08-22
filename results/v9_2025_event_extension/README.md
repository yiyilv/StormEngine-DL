# V9-A frozen 2025 event-metric extension

This is a post-freeze metric extension of the existing one-time 2025 test. It reuses the unchanged checkpoint, windows, predictions, and thresholds derived only from 2010--2015. It does not permit model or threshold changes.

| model | event | POD | FAR | CSI | event RMSE | peak bias |
|---|---|---:|---:|---:|---:|---:|
| v7_b | strong_wind_q95 | 0.4672 | 0.2826 | 0.3946 | 2.9060 | -1.5423 |
| v7_b | heavy_precipitation_wet_q95 | 0.1426 | 0.5508 | 0.1214 | 2.9149 | -2.4986 |
| v7_b | heavy_precipitation_fixed_5mm | 0.0615 | 0.5570 | 0.0571 | 5.2646 | -4.2495 |
| v7_b | low_msl_q05 | 0.3155 | 0.8145 | 0.1322 | 7.1601 | 1.8226 |
| v9_a | strong_wind_q95 | 0.4341 | 0.2462 | 0.3802 | 2.9588 | -1.6990 |
| v9_a | heavy_precipitation_wet_q95 | 0.1337 | 0.4763 | 0.1192 | 2.9150 | -2.6348 |
| v9_a | heavy_precipitation_fixed_5mm | 0.0413 | 0.5351 | 0.0394 | 5.2785 | -4.4303 |
| v9_a | low_msl_q05 | 0.2791 | 0.6654 | 0.1795 | 7.0558 | 2.8603 |
| sparse_reconstruction_persistence | strong_wind_q95 | 0.0000 | NA | 0.0000 | 10.4130 | -8.9127 |
| sparse_reconstruction_persistence | heavy_precipitation_wet_q95 | 0.0000 | NA | 0.0000 | 3.6402 | -3.8115 |
| sparse_reconstruction_persistence | heavy_precipitation_fixed_5mm | 0.0000 | NA | 0.0000 | 6.5525 | -6.3812 |
| sparse_reconstruction_persistence | low_msl_q05 | 0.0000 | 1.0000 | 0.0000 | 13.4581 | 11.4478 |
| dense_era5_persistence | strong_wind_q95 | 0.6879 | 0.3139 | 0.5233 | 1.9352 | -0.1978 |
| dense_era5_persistence | heavy_precipitation_wet_q95 | 0.2906 | 0.7065 | 0.1710 | 2.6968 | -0.4501 |
| dense_era5_persistence | heavy_precipitation_fixed_5mm | 0.2112 | 0.7912 | 0.1173 | 4.7668 | -1.1965 |
| dense_era5_persistence | low_msl_q05 | 0.7536 | 0.2412 | 0.6079 | 1.6849 | 0.3434 |
