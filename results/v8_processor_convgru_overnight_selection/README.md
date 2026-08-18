# V8 ConvGRU overnight selection

This experiment completes a five-candidate local ConvGRU search on a fixed
Processor-only development contract, then repeats the best two candidates with
a second random seed.

- training years: 2013--2015
- validation year: 2016
- locked test year: 2017 (not read)
- history/forecast: 12 h / 6 h
- candidates: L1K3, L2K3, L3K3, L2K5, L3K5
- primary selection metric: normalized sea-weighted validation loss
- replication seeds: 42 and 43

The provisional recommendation is **L3K3**.
This is a Processor-development result, not a final end-to-end test result.
Per-variable land/sea validation metrics must be reviewed before the Processor
configuration is frozen for joint-model training.

`seed42_five_candidate_ranking.json` records the first screen;
`two_seed_comparison.json` records the replication decision; and `runs/`
contains checkpoint-free summaries and histories suitable for Git.
