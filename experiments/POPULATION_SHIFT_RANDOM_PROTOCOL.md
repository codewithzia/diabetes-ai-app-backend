# BRFSS State-Held-Out Adaptation: Random Feedback Follow-Up

## Question

Does adding a random sample of verified target-state labels improve the
Track A logistic-regression model on an untouched target test set? This
follow-up uses the same BRFSS file, held-out states, source-only preprocessing
and feature selection, validation-based model selection, and metric definitions
as `POPULATION_SHIFT_PROTOCOL.md`. The earlier run and its files are preserved.

## Feedback selection

- Hold out states 24, 36, 48, and 53 as the target domain.
- Randomly divide target respondents 50/50 into a feedback pool and an
  untouched target test set **without using diabetes labels** to assign either
  row. Target test prevalence is allowed to differ from feedback-pool
  prevalence.
- Within the feedback pool, sample feedback rows uniformly and without
  replacement at sizes 500, 1,000, 2,500, 5,000, 10,000, 20,000, and 30,000.
  Sampling uses row indices and a seed only; labels are not inputs to the
  selection function. Labels are read only after a sampled row is selected,
  as the simulated verified label used for retraining.
- Run five deterministic feedback samples at each nonzero size. The frozen
  source-only baseline is compared with each retrained model on the same
  target-test rows. Source-test metrics track source-domain regression.

## Model fitting and evaluation

Preprocessing, one-hot encoding, missingness indicators, and fixed top-49
mutual-information feature selection are fitted on source training only.
Source validation selects `C` by PR-AUC and the threshold by F1. These choices
are fixed for every feedback size. The primary inferential comparison is the
10,000-label sample with replicate 0: paired stratified bootstrap for PR-AUC
and exact two-sided McNemar for thresholded errors. All feedback-size metrics
and replicate means/standard deviations are descriptive; no selection of a
best size based on target-test results is allowed.

The source and target results are unweighted. BRFSS 2021 is cross-sectional,
and these retrospective labels simulate verified feedback; they are not
prospective clinician-confirmed outcomes. The result should not be described
as clinical effectiveness or design-weighted population performance.

## Run

```powershell
python -m unittest -v experiments.test_population_shift_random_design
python experiments/run_population_shift_random.py
```

The follow-up writes to a new timestamped folder under
`experiments/results/population_shift_random_feedback/` and does not overwrite
the prior experiment or historical thesis artifacts.
