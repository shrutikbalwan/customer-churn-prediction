# Advanced model card

## Intended use

Local educational exploration, reproducible evaluation and batch inference.

## Training and selection

Candidate families: linear model and histogram gradient boosting. Preprocessing
is fitted inside each three-fold search. Selection uses neg_log_loss.
The selected model is logistic. Configuration is in config.json.
Partition row IDs are in outputs/advanced/partitions.json. Calibration and
threshold selection use separate rows from model fitting. The original assignment
models and their reports remain in outputs/ and analysis.ipynb.

## Evaluation history

The original test set was inspected during earlier development. Results here are a historical benchmark, not new independent external validation.
No new external data was acquired. Bootstrap intervals resample rows and do not
account for selection uncertainty or geographic dependence. Subgroup comparisons
are descriptive; small samples and single-class groups are flagged.

## Limitations

Churn labels are observational. Calibration is measured, not guaranteed. Retention scenario values and success rates are assumptions, not treatment-effect estimates. No intervention effectiveness or future cohort performance has been demonstrated.

## Operational scope

The app binds to loopback only, stores no submitted records, limits batch size,
and validates numeric inputs. Range warnings are heuristics, not a fitted drift
detector. Joblib models must only be loaded from trusted sources. This local demo
does not include authentication, production hosting or clinical approval.

## Evidence

- outputs/advanced/summary.json: metrics, source/model hashes and configuration
- outputs/advanced/calibration_validation.json: calibration comparison (classification)
- advanced_report.html: readable model evidence
- verification/advanced_checks.json: final automated checks
