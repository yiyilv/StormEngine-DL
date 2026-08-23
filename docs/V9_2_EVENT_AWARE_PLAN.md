# V9.2 event-aware correction

## Why this experiment exists

The frozen V9/V9.1 models improve ordinary continuous-field metrics but smooth
physical peaks and can produce zero detections for the original extreme-event
thresholds. The original StormEngine architecture document requires event labels
to be derived from physical thresholds and trained alongside continuous fields.
V9.2 implements that missing requirement without overwriting any frozen model.

## Frozen physical definitions

- extreme rain: six-hour accumulated precipitation `> 50 mm`;
- extreme wind: maximum six-hour wind speed `> 20 m/s`;
- compound storm: six-hour precipitation `> 30 mm` **and** maximum wind `> 15 m/s`.

The thresholds are fixed before training and are not tuned on 2019 or 2025.
Hourly precipitation is converted to a non-negative six-hour sum; wind speed is
derived from `u10` and `v10` and maximized over leads +1 through +6.

## Training correction

V9.2 starts strictly from the frozen six-variable V9.1 pressure checkpoint. Its
loss is the existing sea-weighted field MSE plus:

1. balanced focal classification loss for the three physical events;
2. event-conditioned rain/wind intensity loss to reduce peak underprediction;
3. train-only oversampling of windows containing a physical event.

The same continuous forecast produces the event logits, so the event objective
cannot improve merely through a detached classifier while leaving the weather
fields unchanged.

The first capped pilot is preserved as the `base` profile. It improved ordinary
storm detection but still missed the rare 50 mm/20 m/s extremes. The `strong`
profile is a controlled second pilot with 4x classification weight, 5x intensity
weight, and a 20x train-only event-window sampling weight. It must be compared on
the same 2018 validation set before either profile can advance.

## Leakage boundary

- development training: 2015–2017;
- model selection: 2018 only;
- 2019 is empty in this experiment because V9.1 already exposed it;
- 2025 remains a previously exposed diagnostic, not a new final test.

Consequently V9.2 can be accepted as a development improvement only after its
2018 pilot comparison (`evaluate-pilot` compares the candidate, frozen V9.1,
and dense ERA5 persistence using both field and physical-event metrics). A new untouched period (or prospectively collected data)
is required for a new claim of independent generalization.

## Current scope versus the original final architecture

This correction keeps the current six-hour 31×33 SetConv/ConvGRU system so the
effect of event supervision is isolated. Radar, lightning, CAPE/TCWV, the
Vision-Transformer processor, U-Net decoder, 1 km output, and 48–72 hour range
remain later stages that require new aligned caches and separate controlled
experiments.
