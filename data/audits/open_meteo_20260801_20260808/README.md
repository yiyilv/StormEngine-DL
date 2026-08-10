# Open-Meteo ICON-2I marine audit

- Coverage: 151/151 registered virtual sea points
- Time: 2026-08-01T00:00:00.000000000 through 2026-08-08T00:00:00.000000000 (169 hourly valid times)
- Contract tensor: `(169, 151, 5)` in `u10,v10,i10fg,t2m,tp` order
- Coordinate distance: max 3.336 km; 0 points over 10 km
- Returned elevation over 10 m: 14 points

Open-Meteo is a model-derived marine support source, not a physical observation network. TP is hourly precipitation and is never interpolated or forward-filled. Model run time is unavailable in this stitched historical product and is not fabricated. The processed tensor and raw JSON remain outside Git.

## Variable coverage

| variable | valid/total | coverage |
|---|---:|---:|
| u10 | 25519/25519 | 100.00% |
| v10 | 25519/25519 | 100.00% |
| i10fg | 25519/25519 | 100.00% |
| t2m | 25519/25519 | 100.00% |
| tp | 25519/25519 | 100.00% |
