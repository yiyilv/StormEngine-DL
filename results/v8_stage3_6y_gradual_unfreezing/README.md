# V8 Stage 3 — six-year gradual unfreezing

Stage 3A trains Processor and Decoder while keeping Encoder frozen. Stage 3B then jointly fine-tunes Encoder, Processor, and Decoder with discriminative learning rates. Both seeds use 2010--2015 training and 2016 validation; 2017 is not read. The fixed simultaneous reconstruction diagnostic must remain within the preregistered 3% degradation gate.
