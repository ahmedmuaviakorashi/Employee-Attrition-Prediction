# Employee attrition model results

## Data used

- Supplied rows: **14,999**
- Exact repeated rows after the first occurrence: **3,008**
- Distinct rows used for modeling: **11,991**
- Stayed: **10,000**
- Left: **1,991**

The dataset contains no employee identifier. Exact repeated rows could be duplicate records or different employees with identical de-identified values. The primary analysis removes repeated rows to prevent identical records from appearing in both training and test sets; this is a modeling safeguard, not a claim that the people are duplicates.

## Held-out test performance

| Model | ROC-AUC | Average precision | Precision | Recall | F1 | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic regression | 0.848 | 0.395 | 0.430 | 0.847 | 0.570 | 0.788 |
| Random forest | 0.981 | 0.966 | 0.989 | 0.920 | 0.953 | 0.985 |

ROC-AUC and average precision use predicted probabilities, not hard class labels. Cross-validation summaries are available in `metrics.json` and were calculated on the training partition only.

## Interpretation and responsible use

Permutation importance describes which inputs the fitted random forest relies on; it does not establish why employees leave. Satisfaction, evaluation, workload, and tenure may reflect current organizational conditions, may be unavailable at the intended prediction time, and may be influenced by earlier management decisions.

This model should support aggregate investigation, voluntary retention programs, and workload review. It should not be used as the sole basis for decisions about an individual employee, compensation, promotion, discipline, or termination. Deployment would require a defined prediction time, protected-attribute and subgroup audits, monitoring, employee-data governance, and human review.
