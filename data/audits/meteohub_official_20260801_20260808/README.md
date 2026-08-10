# Official MeteoHub audit: 1--8 August 2026

This directory documents the current official-observation snapshot used to
design the StormEngine operational input adapter. The raw JSON Lines files are
external to Git. Their filenames, request settings, sizes, line counts, and
SHA-256 hashes are frozen in
`data/manifests/meteohub_20260801_20260808.json`.

## Scope

- exact range: 2026-08-01 00:00 to 2026-08-08 00:00 UTC;
- JSON output;
- all available product, level, and timerange semantics retained;
- quality-controlled observations used by default;
- a second unfiltered FVG file retained only for QC comparison.

## Integrity and coverage

| Extract | JSONL records | Parsed variable measurements | Project coverage |
|---|---:|---:|---:|
| FVG QC | 25,597 | 77,729 | 22/22 |
| Central/ER QC | 598,798 | 1,144,966 | all except BOA 2/6 |
| Puglia QC | 1,366,544 | 3,726,712 | 61/61 |
| Veneto QC | 415,304 | 826,637 | 22/22 |

The Central/ER project coverage is Marche 54/54, Molise 4/4, AGRMET
3/3, SIMNBO 14/14, SPDSRA 21/21, URBANE 1/1, MAREFE 9/9, and BOA 2/6.
Across the MeteoHub-backed portion of the registry, 213 of 217 selected coastal
coordinates are observed. The four absent coordinates are BOA Cervia/Cattolica
port or radar stations. Keep them in the coordinate registry and represent
their current absence with masks.

Abruzzo is separate: 22 Polaris-linked project coordinates exist in the
registry but no matching observation extract is available here.

## Quality-control comparison

The FVG QC file retained 77,729 of 77,734 variable measurements (99.9936%).
It removed five simultaneous zero values at `Monte Lussari sm` on
2026-08-03 08:00 UTC: station pressure, relative humidity, wind direction,
wind speed, and gust. No selected FVG coastal observation was removed.

## Important limitations

- Puglia has no pressure or gust descriptor in this interval.
- No selected Veneto coastal coordinate has pressure in this interval.
- `B10004` is station pressure and must not be equated to ERA5 `msl` without a
  documented sea-level correction.
- Station-variable coverage is heterogeneous, so a `[T,N,C]` value mask is
  required. A station-only mask is insufficient.
- Precipitation must retain its BUFR timerange semantics before hourly
  aggregation.
- This one-week snapshot validates the operational schema; it does not replace
  multi-year ERA5 training data.

Detailed execution requirements and acceptance criteria are in
`docs/OFFICIAL_DATA_HANDOFF.md`.
