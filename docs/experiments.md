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
| Amount descending | 0.0371 | 0.020 | 0.0032 | 10 | no better than random; fraud amounts are not large |
| Logistic Regression | 0.1769 | 0.012 | 0.0019 | 6 | scores saturate at 1.0 under temporal covariate shift; see error analysis |
| LightGBM | 0.4996 | 0.930 | 0.1508 | 465 | 46.5× the fraud found by amount ranking |
| LightGBM + history | 0.4974 | 0.912 | 0.1479 | 456 | strictly-past card1/addr1 statistics add nothing |

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
- Cause: temporal covariate shift. Anonymized features V322, V101, V324, V95, V177, V323, V293, V103 reach max |z| of 820–1,829 on the test period versus 19–67 during training. Linear extrapolation saturates the sigmoid, and ties in the queue are then broken by transaction order.
- LightGBM is unaffected: test AP 0.4996 versus validation 0.5231. Tree splits are invariant to monotone feature drift.
- Practical takeaway: a linear model used for capacity-constrained ranking needs drift monitoring and recalibration; the tree model transfers across time.

## Historical-feature ablation

Adding strictly-past card1/addr1 statistics (transaction count and mean amount, computed only from strictly earlier timestamps) changes LightGBM's AP by −0.0022 on the test period (+0.0006 on validation) and finds 9 fewer frauds at K = 500. Conclusion: once card1/addr1 are used as categorical features, strictly-past aggregates provide no additional ranking value. Reported as a negative result; both runs are documented in `artifacts/final_metrics.json` and `artifacts/final_history_metrics.json`.

Source files for every number above: `artifacts/validation_base.json`, `artifacts/validation_history.json`, `artifacts/final_metrics.json`, `artifacts/final_history_metrics.json`.
