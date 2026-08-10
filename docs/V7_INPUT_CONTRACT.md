# V7 operational input contract

V7 is trained from historical ERA5 fields and deployed with official DPC point
observations. Both sources must produce the same model-facing tensor contract.

## Model-facing variables

| V7 channel | ERA5 training source | DPC inference source | Unit/semantics |
|---|---|---|---|
| `u10` | `u10` sampled at the fixed station | 10 m speed/direction converted to u | m s-1, instantaneous |
| `v10` | `v10` sampled at the fixed station | 10 m speed/direction converted to v | m s-1, instantaneous |
| `i10fg` | `i10fg` sampled at the fixed station | `wind_gust_max` | m s-1; proxy mapping retained explicitly |
| `t2m` | `t2m` sampled at the fixed station | 2 m air temperature | degrees Celsius |
| `tp` | hourly `tp` sampled at the fixed station | accepted complete preceding-hour accumulation | mm per preceding hour |

The existing cache already stores normalized `u10`, `v10`, `i10fg`, and `t2m`
point values. It stores normalized `tp` in `target_grids`; V7 bilinearly samples
that channel at the 239 fixed physical-station coordinates without rebuilding
the multi-year cache.

## Tensor shapes

```text
values           [T,239,5]
value_mask       [T,239,5]
observation_age  [T,239,5]
station_present  [T,239]
coordinates      [239,2]
station_static   [239,F]
```

Values use ERA5 2010-2015 training-only normalization. Missing values and ages
are zero after normalization, while `value_mask` remains false. Valid DPC ages
are expressed as a fraction of the 60-minute latest-at-or-before limit.

ERA5 training replays contiguous 12-hour mask and age windows from the pinned
official DPC tensor. Its byte size, SHA-256, station count, hour count, and
template count are recorded in
`data/manifests/v7_dpc_mask_profile_20260801_20260808.json`. Non-zero ages are
represented by linear interpolation between the preceding and current ERA5
hour for instantaneous wind and temperature. A stale gust uses the preceding
hour rather than interpolating a maximum. TP must be a complete accumulation
ending at the current valid time and is rejected if stale.

## Deliberately excluded from this contract

- DPC station pressure is retained by the operational tensor but is not renamed
  to ERA5 mean-sea-level pressure.
- Relative humidity is retained by the operational tensor. Adding it requires
  an ERA5 humidity/dew-point source and a separately versioned input contract.
- Radiation remains optional because official coverage and measurement
  semantics are not yet aligned with ERA5 `ssrd`.

The output task remains the frozen five-field, six-hour ERA5 grid forecast:
`msl`, `u10`, `v10`, `t2m`, and `tp` on the 31 by 33 domain.
