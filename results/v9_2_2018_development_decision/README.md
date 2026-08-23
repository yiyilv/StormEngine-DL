# V9.2 2018 development decision

V9.2 strong is approved for a one-epoch 2010--2025 production refit. This is a
development decision, not a new independent test claim.

Compared with frozen V9.1 on the complete 2018 validation set:

| Metric | V9.1 | V9.2 strong | Decision signal |
|---|---:|---:|---|
| storm-any case CSI | 0.3516 | 0.4960 | improved |
| storm-any case POD | 0.3631 | 0.5536 | improved |
| storm-any case FAR | 0.0827 | 0.1733 | worsened but bounded |
| extreme-wind case hits | 0 | 1 | first non-zero detection |
| extreme-rain case hits | 0 | 0 | unresolved risk |
| compound-storm case hits | 0 | 0 | unresolved risk |

Sea RMSE changed by +0.8% to +2.6% across the five continuous targets. Peak
underprediction improved for rain and wind, but remained substantial. The
production refit therefore retains both the event-aware checkpoint and frozen
V9.1 baseline; it does not claim that the extreme-rain problem is solved.

The complete machine-readable evaluation is published as
`development_evaluation_2018.json`, with the formal training history in
`formal_train_summary.json`. Large checkpoints are intentionally excluded from Git.
