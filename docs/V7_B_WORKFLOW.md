# V7-B workflow

V7-B retains the V7-A mask-aware SetConv/ConvGRU/decoder architecture and adds
151 fixed Open-Meteo ICON-2I marine support coordinates to the 239 physical
DPC/MeteoHub stations. Both sources use the common variable order
`u10,v10,i10fg,t2m,tp`, ERA5 2010-2015 normalization, variable masks, ages, and
fixed registry order.

Training continues to use the existing 2010-2017 normalized ERA5 memory-mapped
cache. V7-B selects all 390 cached coordinates, so no additional multi-gigabyte
cache is required. The split remains 2010-2015 training, 2016 validation/model
selection, and frozen 2017 testing.

The real one-week replay combines `[169,239,5]` DPC data with `[169,151,5]`
Open-Meteo data into `[169,390,5]`. Only the past 12 valid times up to each
forecast origin are model inputs; Open-Meteo future valid times are not used as
predictors. Open-Meteo is labelled model-derived marine support rather than a
physical observation source.

Use `notebooks/StormEngine_V7B_Workflow.ipynb` and enable long-running stages
one at a time after inspecting the preceding output.
