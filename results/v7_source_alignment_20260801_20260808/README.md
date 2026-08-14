# V7 source alignment audit, 2026-08-01 to 2026-08-08

This audit compares operational DPC observations and model-derived Open-Meteo
marine inputs with ERA5T at identical valid times and coordinates. It measures
source alignment and distribution shift; it is not a forecast evaluation and
must not be interpreted as a controlled provider ranking because DPC and
Open-Meteo occupy different coastal and marine locations.

## Scope

- Period: 2026-08-01 00:00 through 2026-08-08 00:00, 169 hourly timestamps.
- DPC: 239 physical stations.
- Open-Meteo: 151 marine model coordinates.
- Variables: `u10,v10,i10fg,t2m,tp`.
- DPC comparison: observation minus bilinearly sampled ERA5T.
- Open-Meteo comparison: ICON-2I-derived value minus bilinearly sampled ERA5T.
- DPC is reported for all valid cells and for observations no more than five
  minutes old.
- Open-Meteo is checked at both requested model coordinates and
  provider-returned coordinates.

`msl` is excluded because DPC station pressure is not directly comparable with
ERA5T mean-sea-level pressure.

## Main statistics

| source | variable | count | bias | RMSE | correlation |
|---|---|---:|---:|---:|---:|
| DPC all valid | u10 | 6,832 | +0.094 | 1.722 | 0.522 |
| DPC all valid | v10 | 6,832 | +0.317 | 1.826 | 0.575 |
| DPC all valid | i10fg | 3,543 | -0.307 | 2.136 | 0.627 |
| DPC all valid | t2m | 22,175 | +0.107 | 2.240 | 0.850 |
| DPC all valid | tp | 24,215 | +0.00044 | 0.227 | 0.041 |
| Open-Meteo requested coordinates | u10 | 25,519 | -0.053 | 1.621 | 0.688 |
| Open-Meteo requested coordinates | v10 | 25,519 | +0.165 | 1.512 | 0.710 |
| Open-Meteo requested coordinates | i10fg | 25,519 | -0.091 | 1.811 | 0.573 |
| Open-Meteo requested coordinates | t2m | 25,519 | -0.013 | 1.333 | 0.728 |
| Open-Meteo requested coordinates | tp | 25,519 | +0.00023 | 0.171 | 0.001 |

## Interpretation

- No large constant offset indicates an obvious unit conversion failure in
  wind or temperature.
- DPC temperature aligns strongly with ERA5T and has small mean bias.
- DPC wind and gust show moderate same-time association with ERA5T.
- Restricting DPC to observations at most five minutes old changes the metrics
  negligibly, so observation age is not the principal source discrepancy.
- Open-Meteo wind and temperature show moderate-to-strong association with
  ERA5T and small mean bias.
- Open-Meteo results at requested and returned coordinates are nearly
  identical, so provider grid-cell displacement is not a material issue here.
- Precipitation correlations are near zero. Most cells are dry and reporting
  windows may differ between sources, so precipitation timing semantics require
  further audit before drawing a model conclusion.

The complete count, bias, MAE, RMSE, correlation, moments, and percentiles are
published in `source_alignment.json`. Raw ERA5T, DPC, and Open-Meteo files stay
outside ordinary Git.
