# MeteoHub legacy official-observation audit

This audit was generated from the three original MeteoHub JSON Lines exports retained
in the teammate repository. The source files span approximately 18--21 June 2026 and
contain `dpcn-marche`, `dpcn-puglia`, and `dpcn-veneto` observations.

The raw exports are deliberately not copied into this repository. Run
`scripts/audit_official_observations.py` with the original files to reproduce these
tables.

## Main findings

- 853,256 raw variable measurements become 409,605 unique measurements after removing
  overlapping extractions.
- Five overlapping observations have revised values. The later extraction is retained,
  and the revision count is recorded rather than silently ignored.
- The sample contains 397 official stations. It observes 85 of the current 239 enabled
  project physical stations: all 54 Marche stations, all 22 Veneto stations, and 9 of
  61 Puglia stations.
- Absence from this three-day sample does not prove that a registered station does not
  exist or has no sensor. It means only that no matching observation was present in the
  retained extraction files.
- Precipitation (`B13011`) uses several accumulation windows: 60, 300, 600, 900, and
  1,800 seconds. The accumulation period must be retained and normalized before an
  hourly ERA5 comparison.
- Maximum wind gust (`B11041`) is reported over a 3,600-second window. Wind direction
  and wind speed (`B11001`, `B11002`) are instantaneous values at a 10 m level in these
  samples.
- `B13013` and `B13215` remain intentionally unmapped until their meaning and units are
  verified against authoritative MeteoHub/BUFR metadata.
- The JSON value objects retained here contain only `v`; no source quality flag is
  embedded in these particular exports. Future downloads should request reliable-only
  data and preserve extraction metadata.

## Outputs

- `audit_summary.json`: overall counts and network coverage.
- `variable_summary.csv`: variable-level station and observation counts.
- `timerange_summary.csv`: original BUFR time-range combinations.
- `station_variable_summary.csv`: per-station availability and median sampling interval.
- `station_summary.csv`: all stations present in the source files.
- `project_station_coverage.csv`: all 239 physical project stations, including stations
  absent from this sample.
- `verified_stations_candidate.csv`: the 85 project stations observed in this sample.

The candidate file must not replace `data/stations_verified.csv` yet. A new official
download covering all target networks and a longer interval is required first.
