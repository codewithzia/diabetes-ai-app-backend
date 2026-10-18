# BRFSS State-Held-Out Supervised Adaptation Protocol

## Objective

Measure whether retraining the Track A logistic-regression approach with
verified labels from a held-out BRFSS state group improves discrimination in
that target group, and whether it reduces performance on the source group.
This is a retrospective supervised-adaptation study, not reinforcement
learning and not a prospective clinical deployment evaluation.

## Data and split rules

- Input: `D:\MSc\BRFSS_Diabetes_Thesis\data\processed\brfss_2021_diabetes_research_base.csv`.
- Label: binary `diabetes`; predictors are the same 18 raw BRFSS fields used by
  the Track A backend. `_STATE`, `_LLCPWT`, and every other metadata field are
  excluded from model input.
- Target states are fixed before modeling to BRFSS codes **24, 36, 48, 53**.
  These correspond to Maryland, New York, Texas, and Washington. All other
  included states form the source domain. This fixed geographic holdout is not
  selected by comparing observed model performance.
- Within the source rows, a seeded, outcome-stratified split assigns 70% to
  source training, 15% to source validation, and 15% to a source test set.
  Source validation is for C/threshold selection only; source test measures
  regression after adaptation.
- Target rows are stratified 50/50 into a target feedback pool and target test
  set. Target-test labels are never used for preprocessing, feature selection,
  model selection, threshold selection, feedback sampling, or training.
- Exact split assignments, row counts, positive counts, input SHA-256, and
  state membership are written for each run.

## Modeling and adaptation

- Build the 18 raw predictor columns plus the 17 notebook-style missingness
  flags. Fit most-frequent imputation and one-hot encoding on source training
  only. Unknown categories in held-out states are ignored by the encoder.
- Fit a fixed top-49 mutual-information feature selector on source training
  only. This is a new, explicitly defined selection rule for this experiment;
  it does not reuse the historical selected-feature artifact, which was fit
  before the state holdout was designed.
- Select logistic-regression `C` from `{0.1, 1, 10}` by source-validation
  PR-AUC. Select the classification threshold by source-validation F1. Both
  choices are frozen for all feedback-size comparisons.
- The frozen baseline trains only on source training. Adaptive models are
  retrained from scratch on source training plus a deterministic stratified
  sample of verified labels from the target feedback pool at sizes
  500, 1,000, 2,500, 5,000, 10,000, 20,000, and 30,000 (sizes are run only if
  available). Five feedback-sampling seeds are used per nonzero size.
- Feedback labels are the true retrospective BRFSS outcomes and stand in for
  verified feedback. They are not user agreement, reward, or prospective
  clinician-confirmed labels.

## Outcomes and inference

- Primary outcome: unweighted target-test PR-AUC difference at 10,000 feedback
  labels, using feedback replicate 0. Its 95% interval is a paired,
  outcome-stratified bootstrap over the same target-test respondents.
- The paired exact two-sided McNemar test compares thresholded errors for the
  same primary baseline/adaptive predictions on the target test set.
- The same paired bootstrap and McNemar comparison on the source test set
  measures source-domain regression.
- Accuracy, precision, recall, F1, ROC-AUC, PR-AUC, and confusion counts are
  reported for source and target tests at every feedback size. Mean and sample
  standard deviation across the five feedback samples show sensitivity to
  which labels were reviewed.
- Survey weights and BRFSS strata/PSU variance estimation are not used for the
  primary metrics; results describe these held-out respondents and should not
  be presented as design-weighted population estimates.

## Reproducibility and historical artifacts

Run from the backend checkout with the thesis data available:

```powershell
python experiments/run_population_shift.py
```

The script creates a fresh timestamped directory under
`experiments/results/population_shift/`. It does not modify notebooks, thesis
data, historical results, model artifacts, backend runtime configuration, or
the adaptive registry. The dependency-free split checks are:

```powershell
python -m unittest -v experiments/test_population_shift_design.py
```

No metric improvement is asserted in advance. Report measured values and
uncertainty, including null or negative effects.
