# V8 Stage 2 — Processor-only sparse forecasting

Two converged Stage-1 spatial candidates were each combined with a randomly initialized ConvGRU L3K3. Encoder and Decoder were frozen; only Processor parameters were optimized. Each candidate was repeated with seeds 42 and 43 on 2013--2015 training and 2016 validation. The 2017 test set was not read.

Provisional spatial recommendation: **PH64-LAT96**. This is a validation-stage decision; Stage 3 joint fine-tuning remains required.
