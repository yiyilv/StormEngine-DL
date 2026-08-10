# Official MeteoHub audit: 15--22 June 2026

This audit uses the quality-controlled JSON Lines extraction named
`stormengine_official_audit_20260615_20260622`.

The raw file is stored outside Git under
`data_external/meteohub/raw/20260615_20260622/`. Its SHA-256 is
`409318f2b4ef650924c88c08bcc2173d355495d73b69b7617a5fcb666a9a92fb`.

## Integrity

- Download size: 475,495,068 bytes.
- JSON Lines records: 937,534.
- Invalid JSON lines: 0.
- Parsed variable measurements: 1,861,850.
- Observation range: exactly 2026-06-15 00:00 to 2026-06-22 00:00 UTC.

## Network coverage

The request selected 11 networks, but the downloaded file contains 10. `arpafvg`
has no records in this interval. Later QC/no-QC comparisons and portal probes showed
that FVG is also absent from 1--8 July and begins supplying data around 27 July 2026.
The June absence is therefore a historical-availability limitation, not evidence that
the quality-control filter removed the network.

Of the 239 current project physical stations, 134 are observed in this extraction:

- Marche: 54/54
- Molise: 4/4
- Veneto: 22/22
- Puglia: 9/61
- AGRMET: 3/3
- BOA: 5/6
- SIMNBO: 14/14
- SPDSRA: 21/21
- URBANE: 1/1
- MAREFE: 1/9
- ARPAFVG: 0/22
- Abruzzo Polaris: 0/22 (not a MeteoHub dataset in this request)

Absence from this extraction means only that no quality-controlled observation from
the registered coordinate was present in this time window. It is not proof that the
station or sensor does not exist.

## Core variable coverage among project stations

- precipitation amount: 89 stations
- air temperature: 63 stations
- relative humidity: 47 stations
- wind speed and direction: 17 stations
- maximum wind gust speed: 6 stations
- pressure: 6 stations
- downward global visible irradiance: 2 stations

The results reinforce the need for a per-time, per-station, per-variable mask. Pressure,
gust and radiation cannot be mandatory inputs for every physical station.

## Required follow-up

1. Retain the August ARPAFVG QC/no-QC control comparison as provenance for the later
   period in which the network is available.
2. Acquire Abruzzo Polaris observations separately.
3. Compare the overlapping 18--21 June records with the teammate extraction.
4. Normalize precipitation by its original accumulation window before hourly resampling.

## Comparison with the teammate extraction

The overlapping interval is 18 June 11:23 to 21 June 16:10 UTC for Marche,
Puglia and Veneto.

- teammate unique measurements: 409,605
- new quality-controlled measurements in the same interval: 435,434
- measurements common to both: 409,605
- teammate-only measurements: 0
- newly available measurements: 25,829
- common measurements with a revised value: 7

The new official extraction therefore contains every retained teammate observation plus
25,829 additional late or newly included measurements. The seven revisions affect four
wind-direction values, two air-temperature values and one wind-speed value. This confirms
that the raw official extraction should be versioned and that later downloads can contain
both additions and corrections.
