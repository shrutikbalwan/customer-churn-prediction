# Churn improvement study: findings

**Decision: retain the current application model.** The candidate is promising but
did not meet the rule fixed before this study. No model was promoted, no historical
test metrics were used for tuning, and nothing was pushed to GitHub.

## What was compared

The original 5,634 training rows were evaluated with five stratified outer folds
and three inner folds. The 1,409 historical-test rows were excluded throughout.
Each outer training partition was split into 60% fitting, 20% calibration and 20%
threshold validation. Both procedures used the same partitions and sigmoid
calibration. F1 thresholds and business thresholds were chosen on validation rows,
then evaluated on the outer fold.

- Baseline: freshly trained logistic regression with raw features and C tuning.
- Challenger: raw/engineered logistic regression and histogram gradient boosting,
  selected by inner validation log loss. Engineered logistic regression won in all
  five outer folds.

Features tested included tenure bands, service/support counts, historical monthly
spend, spending changes, charge per service, spending-tenure interaction,
contract/internet combinations, and automatic-payment flags. Transformations are
row-local. Imputation, scaling and encoding are fitted inside the search folds.
Availability of historical charges at the intended decision time is an assumption
that must be verified before operational use.

## Results

These are means across outer folds, not the old historical-test scores.

| Metric | Baseline procedure | Challenger procedure |
|---|---:|---:|
| F1 | 0.6248 | 0.6319 |
| Precision | 0.5561 | 0.5646 |
| Recall | 0.7224 | 0.7184 |
| ROC-AUC | 0.8432 | 0.8461 |
| Log loss (lower is better) | 0.4218 | 0.4178 |
| Brier score (lower is better) | 0.1367 | 0.1354 |

Promotion required mean F1 improvement of at least 0.01, positive F1 gains on at
least four of five folds, and no increase in mean log loss. Actual gain was
**0.00714**, with wins on **three** folds. Log loss improved, but the other two
requirements were not met. This conservative engineering rule is not a statistical
significance test; outer folds overlap in their training data.

The baseline is a newly fitted procedure using the same partitions as the
challenger, not a direct out-of-fold score for the saved application model.
Keeping the current model means evidence for replacement was insufficient; it
does not establish that the saved model is universally better.

## Error investigation

For the baseline out-of-fold predictions:

| Contract | Rows | Actual churners | False positives | Missed churners |
|---|---:|---:|---:|---:|
| Month-to-month | 3,102 | 1,326 | 829 | 258 |
| One year | 1,173 | 130 | 43 | 119 |
| Two year | 1,359 | 39 | 1 | 38 |

The global F1 threshold detects many month-to-month churners but generates many
false alarms there. Annual-contract churn is uncommon, and most such churners are
missed. These figures motivate further investigation, not an automatic decision
to target customers by contract type. Other segment tables and local row indexes
are saved in outputs/churn_study/error_groups.csv and out_of_fold_errors.csv.
Segments overlap and must not be summed across attributes.

## Retention-cost scenarios

Assumptions: retained customer value 500 currency units; intervention success 20%;
per-contact costs 10, 30 or 60. Success is hypothetical, not a measured treatment
effect. Costs apply to all selected contacts. Values below are mean assumed net
value per 1,000 evaluated customers, using validation-selected thresholds.

| Contact cost | Baseline | Challenger |
|---|---:|---:|
| 10 | 18,682.9 | 18,675.8 |
| 30 | 8,769.9 | 8,968.7 |
| 60 | 1,689.7 | 1,565.4 |

The challenger is slightly better only in the middle-cost scenario. There is no
consistent business gain. Full tables also compare contacting everyone and
contacting nobody. These are scenario calculations, not realized revenue or
causal estimates of retention impact.

## Verification and reproduction

**20 tests passed**, including seven study-specific tests. They check row-local
features, zero-tenure behavior, threshold choices, promotion rules, partition
separation, complete out-of-fold coverage, metric recomputation, business values,
and unchanged application model hashes. A second full study reproduced identical
mean metrics and the same decision. The two suite warnings intentionally exercise
unknown-category fallback in the existing inference tests.

Run from this project folder:

```powershell
python -m ml_projects.churn_study
python -m ml_projects.advanced_verify
```

Read CHURN_IMPROVEMENT_REPORT.html for the detailed report. Evidence is in
verification/churn_study_checks.json and verification/advanced_pytest.txt.
The exact protocol, candidate searches, row partitions and validation predictions
are included under outputs/churn_study/.

## Limits and next evidence needed

This dataset was used in earlier development. Nested CV improves separation within
this experiment; it cannot turn previously studied data into new external data.
Do not repeatedly revise features against these same outer-fold results and then
claim independent validation. A new temporal cohort, actual intervention-cost
information, and support/renewal history available before the prediction time
would be more useful than increasing model complexity without new evidence.

Method reference: [scikit-learn nested cross-validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).
