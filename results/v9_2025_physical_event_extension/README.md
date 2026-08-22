# Frozen 2025 original physical-event extension

This is a post-freeze metric extension of the existing one-time 2025 test. It reuses the unchanged checkpoints, windows, prediction pipeline, and baselines. It does not permit model or threshold changes.

## Original six-hour physical event definitions

Hourly precipitation is clipped at zero and summed over +1--+6 h. Wind speed is derived from u10/v10 and maximized over the same window. Both grid-cell localization and whole-forecast-case detection are reported.

| model | event | grid POD | grid FAR | grid CSI | case POD | case FAR | case CSI | tp6h RMSE | wind-max RMSE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v7_b | rain_6h_10mm | 0.2508 | 0.5026 | 0.2001 | 0.5684 | 0.0854 | 0.5398 | 10.5168 | NA |
| v7_b | heavy_rain_6h_30mm | 0.0342 | 0.7797 | 0.0305 | 0.0879 | 0.3333 | 0.0842 | 28.2727 | NA |
| v7_b | extreme_rain_6h_50mm | 0.0000 | NA | 0.0000 | 0.0000 | NA | 0.0000 | 43.9393 | NA |
| v7_b | strong_wind_6h_15ms | 0.1383 | 0.3978 | 0.1267 | 0.2614 | 0.1481 | 0.2500 | NA | 4.5129 |
| v7_b | extreme_wind_6h_20ms | 0.0000 | NA | 0.0000 | 0.0000 | NA | 0.0000 | NA | 11.3578 |
| v7_b | storm_any_6h | 0.1275 | 0.4257 | 0.1165 | 0.2305 | 0.1385 | 0.2222 | 10.5384 | 4.3529 |
| v7_b | compound_storm_6h | 0.0000 | NA | 0.0000 | 0.0000 | NA | 0.0000 | 11.0346 | 6.2228 |
| v7_b | extreme_weather_6h | 0.0000 | NA | 0.0000 | 0.0000 | NA | 0.0000 | 38.2850 | 6.0986 |
| v9_a | rain_6h_10mm | 0.2448 | 0.4319 | 0.2064 | 0.5126 | 0.0848 | 0.4893 | 10.4150 | NA |
| v9_a | heavy_rain_6h_30mm | 0.0316 | 0.7143 | 0.0293 | 0.0879 | 0.2000 | 0.0860 | 27.8929 | NA |
| v9_a | extreme_rain_6h_50mm | 0.0000 | NA | 0.0000 | 0.0000 | NA | 0.0000 | 44.2803 | NA |
| v9_a | strong_wind_6h_15ms | 0.1236 | 0.3327 | 0.1164 | 0.2955 | 0.0877 | 0.2873 | NA | 4.9023 |
| v9_a | extreme_wind_6h_20ms | 0.0000 | NA | 0.0000 | 0.0000 | NA | 0.0000 | NA | 11.9517 |
| v9_a | storm_any_6h | 0.1138 | 0.3578 | 0.1070 | 0.2510 | 0.0896 | 0.2450 | 10.3746 | 4.6928 |
| v9_a | compound_storm_6h | 0.0000 | NA | 0.0000 | 0.0000 | NA | 0.0000 | 13.8919 | 6.2120 |
| v9_a | extreme_weather_6h | 0.0000 | NA | 0.0000 | 0.0000 | NA | 0.0000 | 38.6571 | 6.1336 |
| sparse_reconstruction_persistence | rain_6h_10mm | 0.0000 | 1.0000 | 0.0000 | 0.0112 | 0.1111 | 0.0112 | 14.1290 | NA |
| sparse_reconstruction_persistence | heavy_rain_6h_30mm | 0.0000 | NA | 0.0000 | 0.0000 | NA | 0.0000 | 37.6039 | NA |
| sparse_reconstruction_persistence | extreme_rain_6h_50mm | 0.0000 | NA | 0.0000 | 0.0000 | NA | 0.0000 | 56.5435 | NA |
| sparse_reconstruction_persistence | strong_wind_6h_15ms | 0.0000 | NA | 0.0000 | 0.0000 | NA | 0.0000 | NA | 14.3049 |
| sparse_reconstruction_persistence | extreme_wind_6h_20ms | 0.0000 | NA | 0.0000 | 0.0000 | NA | 0.0000 | NA | 18.5156 |
| sparse_reconstruction_persistence | storm_any_6h | 0.0000 | NA | 0.0000 | 0.0000 | NA | 0.0000 | 14.2516 | 13.7019 |
| sparse_reconstruction_persistence | compound_storm_6h | 0.0000 | NA | 0.0000 | 0.0000 | NA | 0.0000 | 31.8769 | 15.0765 |
| sparse_reconstruction_persistence | extreme_weather_6h | 0.0000 | NA | 0.0000 | 0.0000 | NA | 0.0000 | 49.5371 | 11.3148 |
| dense_era5_persistence | rain_6h_10mm | 0.4477 | 0.6419 | 0.2484 | 0.8925 | 0.2796 | 0.6629 | 10.6300 | NA |
| dense_era5_persistence | heavy_rain_6h_30mm | 0.3474 | 0.8680 | 0.1058 | 0.7912 | 0.6453 | 0.3243 | 23.2274 | NA |
| dense_era5_persistence | extreme_rain_6h_50mm | 0.2308 | 0.9084 | 0.0702 | 0.6429 | 0.7805 | 0.1957 | 31.8016 | NA |
| dense_era5_persistence | strong_wind_6h_15ms | 0.4144 | 0.1748 | 0.3810 | 0.6307 | 0.0826 | 0.5968 | NA | 3.2067 |
| dense_era5_persistence | extreme_wind_6h_20ms | 0.0588 | 0.6667 | 0.0526 | 0.3333 | 0.5000 | 0.2500 | NA | 11.8994 |
| dense_era5_persistence | storm_any_6h | 0.4131 | 0.4555 | 0.3070 | 0.7078 | 0.4189 | 0.4687 | 9.1529 | 3.1092 |
| dense_era5_persistence | compound_storm_6h | 0.0000 | 1.0000 | 0.0000 | 0.1667 | 0.6667 | 0.1250 | 17.7639 | 6.5751 |
| dense_era5_persistence | extreme_weather_6h | 0.1884 | 0.9030 | 0.0684 | 0.5882 | 0.7561 | 0.2083 | 28.3804 | 6.0593 |
