# Experiment protocol

## Required comparisons

1. Random ranking (fixed seed).
2. Amount descending ranking.
3. Logistic Regression.
4. LightGBM when dependency/environment permits.
5. Optional GraphSAGE and an ablation without relational information.

Use the validation period to compare feature configurations and select settings. Freeze the models, settings, features, review budgets, and ablation plan before evaluating the fixed test period. Report average precision (AP, stored as `pr_auc` in the JSON), Precision@K, Recall@K, fraud cases found, positive counts, split ranges, seed, and runtime. K is the total number of transactions reviewed across the selected period. A daily budget would require ranking each day separately.

Run the baseline and the historical-feature comparison on validation first. Keep the input files, split, seed, budgets, and categorical encoding the same for both runs.

```bash
python scripts/run_experiment.py --transaction data/ieee-cis/train_transaction.csv --categorical-codes card1 addr1 --k 100 500 --out artifacts/validation_base.json
python scripts/run_experiment.py --transaction data/ieee-cis/train_transaction.csv --categorical-codes card1 addr1 --entity-cols card1 addr1 --models lightgbm --k 100 500 --out artifacts/validation_history.json
```

After freezing the protocol, run the planned configurations with `--split test` and separate output names such as `artifacts/final_metrics.json` and `artifacts/final_history_metrics.json`. The script writes metrics to the JSON file and trained models to a sibling directory named after its stem, such as `artifacts/final_history_metrics_models/`. It rejects existing report and model paths. The default model list includes LightGBM. Install the `lightgbm` extra or pass `--models random amount logistic` for a smoke test without it.

## Final results (holdout test period, evaluated once, seed 42)

Test period: 88,581 transactions, 3,083 fraud cases, fraud rate 3.48%, TransactionDT 13,151,880–15,811,131. Train period: 413,378 transactions (TransactionDT 86,400–10,437,996). Validation period was used only for configuration; test was scored once after freezing.

### K = 500 (primary review budget)

| Model | AP | Precision@500 | Recall@500 | fraud found | Notes |
|---|---:|---:|---:|---:|---|
| Random ranking | 0.0353 | 0.026 | 0.0042 | 13 | close to fraud rate × 500 ≈ 17.4 |
| Amount descending | 0.0371 | 0.020 | 0.0032 | 10 | weaker than the single seeded random ranking at K=500 |
| Logistic Regression | 0.1769 | 0.012 | 0.0019 | 6 | scores saturate at 1.0 under temporal covariate shift; see error analysis |
| LightGBM | 0.4996 | 0.930 | 0.1508 | 465 | 46.5× the fraud found by amount ranking |
| LightGBM + history | 0.4974 | 0.912 | 0.1479 | 456 | 9 fewer hits at K=500 and 2 more at K=100 |

### K = 100 (sensitivity)

| Model | AP | Precision@100 | Recall@100 | fraud found |
|---|---:|---:|---:|---:|
| Random ranking | 0.0353 | 0.010 | 0.0003 | 1 |
| Amount descending | 0.0371 | 0.020 | 0.0006 | 2 |
| Logistic Regression | 0.1769 | 0.060 | 0.0019 | 6 |
| LightGBM | 0.4996 | 0.950 | 0.0308 | 95 |
| LightGBM + history | 0.4974 | 0.970 | 0.0315 | 97 |

### Validation period (configuration selection; not final claims)

| Model | AP | Precision@500 | Recall@500 | fraud found |
|---|---:|---:|---:|---:|
| Random ranking | 0.0338 | 0.036 | 0.0059 | 18 |
| Amount descending | 0.0404 | 0.030 | 0.0049 | 15 |
| Logistic Regression | 0.3998 | 0.864 | 0.1420 | 432 |
| LightGBM | 0.5231 | 0.936 | 0.1538 | 468 |
| LightGBM + history | 0.5237 | 0.946 | 0.1555 | 473 |

## Error analysis: why logistic regression collapses on the test period

- 918 test transactions receive probability exactly 1.0; their fraud rate is 2.18%.
- 196 near-saturated rows (0.999999–1.0) have a 47.5% fraud rate but sort below the exact-1.0 group, so the Top-500 queue misses them entirely (6 frauds in Top-500).
- Reported feature diagnostics suggest temporal covariate shift and sigmoid saturation. These per-row diagnostics are exploratory observations and are not contained in the four committed aggregate metric reports. A follow-up should save the diagnostics and compare ranking by LR's unsquashed decision function. Stable ties follow input order.
- LightGBM retained similar ranking performance on this split, with test AP 0.4996 versus validation AP 0.5231. Trees can also deteriorate under changing distributions.
- A monotone transformation of already tied probabilities cannot restore their lost ordering. The original six-hit result remains the reported LR outcome.

## Historical-feature ablation

Adding strictly-past card1/addr1 statistics changes LightGBM's AP by about -0.0021 on test (+0.0006 on validation), finds 9 fewer frauds at K=500, and finds 2 more at K=100. These features did not provide consistent gains across the measured budgets and periods. Both runs are recorded in `artifacts/final_metrics.json` and `artifacts/final_history_metrics.json`.

Source files for every number above: `artifacts/validation_base.json`, `artifacts/validation_history.json`, `artifacts/final_metrics.json`, `artifacts/final_history_metrics.json`.

The metric values and arithmetic can be checked after cloning with `python scripts/verify_results.py`. See `docs/reproducibility.md` for retraining commands and environment limitations. The final analysis is in `docs/final_report.md`.

This study uses a single seed and one holdout, with default baseline model settings, no systematic tuning and no confidence intervals. Queue evaluation assumes a known complete batch. The operational demonstration uses `triage --review-budget K` for the same selection rule, with policy actions recorded separately. Balanced-class outputs are uncalibrated scores. The example policy thresholds have no validated probability interpretation.
