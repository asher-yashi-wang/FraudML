# FraudML Evaluation of Transaction Ranking Under a Fixed Review Budget

DSC5001 course project

Authors: 王雅仕, 李熠辉, 王时弘, 吴嘉琦, 陈斯图

## Abstract

Fraud review teams need to prioritize transactions when investigation capacity is limited. We evaluated whether model scores rank labeled fraud more effectively than descending transaction amount on a future period of the IEEE-CIS dataset. A chronological split reserved 413,378 transactions for training, 88,581 for validation, and 88,581 for testing. On the test period, LightGBM identified 465 labeled fraud cases among its 500 highest-scored transactions, compared with 10 for amount ranking and 13 for one seeded random ranking. Its average precision was 0.4996. Logistic Regression deteriorated substantially between validation and test, illustrating why overall ranking metrics and queue-specific metrics should be examined together. Additional historical card and address statistics produced mixed effects across review budgets. These are baseline results from one seed and one time window. A capacity-based review prototype connects the evaluated ranking rule to an investigator queue.

## Research question and scope

Our primary question was whether model-based ranking finds more labeled fraud than amount-based ranking when both methods select exactly 500 transactions from the same future period. We also examined the effect of adding strictly historical entity statistics and evaluated a smaller budget of 100 transactions.

The review budget applies to the complete evaluation batch. A budget of 500 across this period differs from 500 reviews per day. Measuring a daily operation would require separate rankings and budgets for each day. The labels provide retrospective evaluation outcomes. The experiment measures ranking quality and does not establish how many losses a deployed bank system would prevent.

## Data and time split

We used the labeled IEEE-CIS transaction file provided for the Kaggle competition. The recorded runs did not merge the optional identity table. The data contains anonymized attributes, precomputed features, an outcome column called `isFraud`, and the relative time field `TransactionDT`. Entity-like fields such as `card1` and `addr1` are proxies whose full business meaning is unavailable.

| Period | Transactions | First TransactionDT | Last TransactionDT | Labeled fraud |
|---|---:|---:|---:|---:|
| Training | 413,378 | 86,400 | 10,437,996 | Not recorded in metric files |
| Validation | 88,581 | 10,438,003 | 13,151,840 | 3,042 |
| Test | 88,581 | 13,151,880 | 15,811,131 | 3,083 |

Nominal split fractions were 70%, 15%, and 15%. Equal timestamps were kept within a single period. Validation fraud prevalence was 3.4341% and test prevalence was 3.4804%. We excluded the outcome, transaction identifier, and raw split timestamp from the model inputs. Training-only fitting of preprocessing controls leakage introduced by our pipeline. Because upstream anonymous features lack complete definitions, their original construction and decision-time availability cannot be independently audited here.

## Models and feature engineering

The baselines were one random ordering with seed 42, descending transaction amount, Logistic Regression, and LightGBM. Both trained classifiers used `class_weight="balanced"` and seed 42. Logistic Regression used `max_iter=1000`. LightGBM used 300 estimators with learning rate 0.05. Other estimator settings used the installed library defaults. We did not run a systematic hyperparameter search.

Numeric features were median-imputed and standardized using training-period parameters. We explicitly treated `card1` and `addr1` as category codes. Categorical columns with at most 32 distinct training values were imputed and one-hot encoded. Higher-cardinality columns were frequency-encoded from training counts. Unknown or missing values in the frequency branch mapped to zero. The same fitted pipeline was applied to the evaluation periods.

The historical-feature variant added four columns, the prior transaction count and prior mean amount for each of `card1` and `addr1`. History included only strictly earlier timestamps, including earlier unlabeled observations within an evaluation period. Transactions sharing an entity and timestamp therefore received identical history. Mean amounts excluded missing historical amounts from their denominator. This protocol models scoring sequential transactions as their input attributes become available. It does not use past fraud labels. GraphSAGE remained an optional extension and has no measured result in this study.

## Metrics and comparison protocol

Average precision (AP) summarizes precision across recall increments and is computed with `average_precision_score`. The JSON field is named `pr_auc`; its value should be read as AP rather than trapezoidal area under an interpolated precision-recall curve. For a review budget K, Precision@K is labeled fraud selected divided by K, and Recall@K is labeled fraud selected divided by all labeled fraud in the period. Stable descending sorting breaks score ties by input order.

The same split, seed, categorical treatment, and budgets were used for the base and historical-feature comparisons. Training remained confined to the training period. The saved metric files provide the recorded validation and final test results. They do not independently certify when the protocol was frozen or how often an investigator inspected the holdout.

## Test results

| Method | AP | Precision@500 | Recall@500 | Fraud in Top 500 | Fraud in Top 100 |
|---|---:|---:|---:|---:|---:|
| Random ranking | 0.035292 | 0.026 | 0.004217 | 13 | 1 |
| Amount descending | 0.037122 | 0.020 | 0.003244 | 10 | 2 |
| Logistic Regression | 0.176851 | 0.012 | 0.001946 | 6 | 6 |
| LightGBM | 0.499573 | 0.930 | 0.150827 | 465 | 95 |
| LightGBM with history | 0.497424 | 0.912 | 0.147908 | 456 | 97 |

At K=100, Precision@100 was 0.010, 0.020, 0.060, 0.950, and 0.970 respectively. Recall@100 was 0.000324, 0.000649, 0.001946, 0.030814, and 0.031463. At the primary budget, base LightGBM found 455 more labeled fraud cases than amount ranking. Its 465 hits represent 93% precision, while capturing 15.08% of the 3,083 labeled fraud cases in the test period. High queue precision therefore coexists with substantial fraud remaining outside the queue.

| Model | Validation AP | Test AP | Validation Top 500 hits | Test Top 500 hits |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.399773 | 0.176851 | 432 | 6 |
| LightGBM | 0.523112 | 0.499573 | 468 | 465 |
| LightGBM with history | 0.523723 | 0.497424 | 473 | 456 |

LightGBM retained similar queue quality across these two periods. That observation is specific to this split. Tree models can also suffer under changing feature distributions. The recorded base LightGBM test fit-and-score time was 20.724 seconds, compared with 117.588 seconds for Logistic Regression. These measurements are run timings, not controlled serving-latency benchmarks.

## Error analysis and historical-feature ablation

Logistic Regression had better test AP than the simple baselines but worse Top-500 precision. The distinction matters operationally because only the queue's first 500 transactions receive attention in this experiment. Exploratory analysis reported 918 test predictions exactly equal to 1.0, with a fraud rate of about 2.18%, and extreme standardized values for several anonymous features. Such saturation would produce more tied predictions than the complete review budget, making selected cases sensitive to tie order.

These reported diagnostics suggest a combination of temporal distribution shift and numerical probability saturation. The four metric JSON files do not contain per-row scores or feature diagnostics, so they establish the performance difference but do not by themselves prove this mechanism. A reproducible follow-up would compare score histograms, training and test feature distributions, and ranking by the fitted LR's unsquashed `decision_function`. It should be labeled a post-hoc investigation and retain the original six-hit result as the primary measured outcome. A monotone calibration applied to already tied probabilities cannot recover the lost ordering.

Adding history increased validation Top-500 hits from 468 to 473. In test it reduced hits from 465 to 456 and reduced AP by 0.002149. At K=100 it increased test hits from 95 to 97. The evidence therefore indicates no consistent improvement from these particular historical statistics under the measured configuration. It does not show that all relational or historical features lack value.

## Review workflow and model scores

The capacity-based prototype selects the K highest scores into a review queue using the same stable ranking as evaluation. The queue's `review_selected` and `review_rank` fields are independent of policy action. This allows high-score `CHALLENGE` cases to appear in the queue rather than being excluded by the legacy middle-score-only queue. An explicit `--review-budget 500` produces this evaluated mode. The legacy threshold mode remains available for comparison and has a different queue definition.

The balanced class weights alter the training objective. We use classifier outputs as ranking scores and have not established probability calibration. The example policy's thresholds of 0.45 and 0.85 illustrate routing and cannot be interpreted as validated fraud probabilities. Fixed review capacity is expressed directly through K. Any future score-threshold operating point should be selected on validation data under a stated workload or recall target. Investigator outcomes are recorded separately from model scores and policy actions.

## Model service and Java integration

A local HTTP scoring service loads one trusted model artifact at startup and exposes its fitted input schema through GET /health. POST /score validates request size, row count and feature types, then returns a model SHA-256 version and uncalibrated scores. History-based artifacts require previously prepared causal history columns. A Java 11 HttpClient example supplies connection and request timeouts. The synthetic integration smoke test compiled the client and called a live local server. Three scores matched direct Python predictions to an absolute tolerance of 1e-12. This establishes local interoperability. Production deployment, load testing, authentication and operational monitoring remain future work.

## Reproducibility and limitations

The numerical sources are `artifacts/validation_base.json`, `artifacts/validation_history.json`, `artifacts/final_metrics.json`, and `artifacts/final_history_metrics.json`. They record input paths, split ranges, budgets, seed, category and history settings, results, and model artifact locations. Keeping these small files in version control allows readers to trace tables without obtaining the CSV. Retraining additionally requires downloading the IEEE-CIS data under its access terms, installing compatible dependencies, and running the documented experiment commands. Trained joblib files and source CSVs remain local artifacts.

This baseline study uses one seed and one chronological holdout, provides no confidence intervals, and includes no systematic tuning. The single random realization has sampling variation. Anonymous precomputed features limit interpretation and independent feature-availability checks. Proxy entities can merge unrelated behavior. Historical statistics assume prior transaction inputs are available at scoring time. The human-review demonstration uses synthetic inputs and does not measure investigator accuracy, financial savings, or deployed bank performance.

The results support model-based prioritization on this dataset. They also show that a strong full-ranking metric can coexist with a poor capacity-limited queue, and that added feature complexity can fail to improve a later period. Future evaluation should preserve an untouched time window, assess multiple periods and seeds, and separately measure review operations and any calibrated probability use.

## References

1. Kaggle. IEEE-CIS Fraud Detection. 2019. https://www.kaggle.com/competitions/ieee-fraud-detection
2. Ke, G., et al. LightGBM: A Highly Efficient Gradient Boosting Decision Tree. Advances in Neural Information Processing Systems 30, 2017. https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html
3. Saito, T., and Rehmsmeier, M. The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets. PLOS ONE 10(3), e0118432, 2015. https://doi.org/10.1371/journal.pone.0118432
