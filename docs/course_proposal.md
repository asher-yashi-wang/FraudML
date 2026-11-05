# DSC5001 Project Proposal

Team FraudML · Submitted 5 November 2026

**Project title** Can Machine Learning Find More Payment Fraud Under a Fixed Review Budget?

**Team name** FraudML

| Member | UID | Main responsibility |
|---|---|---|
| Wang Yashi | [UID] | System integration and project coordination |
| Li Yihui | [UID] | Data preparation and exploratory analysis |
| Wang Shihong | [UID] | Baselines and Logistic Regression |
| Wu Jiaqi | [UID] | LightGBM and feature comparison |
| Chen Situ | [UID] | Interpretation and presentation |

## 1. Background and Motivation

Payment reviewers have limited capacity. Payment-system development experience motivates comparing model scores with amount-based prioritization. We measure labeled fraud selected within a fixed review budget.

## 2. Research Questions

- **RQ1** Does model-based ranking find more labeled fraud than amount ranking among 500 selected transactions in a future period?
- **RQ2** Do strictly historical card/address statistics improve LightGBM beyond categorical inputs?
- **RQ3** How does ranking quality change between validation and test, particularly when extreme inputs produce saturated scores?

## 3. Data

- **Source** IEEE-CIS Fraud Detection, provided by Vesta through Kaggle [1], accessed 5 November 2026. `TransactionDT` provides relative time.
- **Unit and size** Each row is a transaction. The labeled transaction file has 590,540 rows and 394 columns. The comparison excludes the identity table.
- **Response** `isFraud` is a binary historical label.
- **Predictors** Amount, product, card/address proxies and anonymous attributes. Additional features summarize prior count and mean amount by `card1` and `addr1`.
- **Known issues** Imbalance, missingness, outliers and unclear upstream definitions. Proxy entities may combine unrelated activity. Anonymous feature availability cannot be audited.

## 4. Proposed Methods

- **Baseline** Seeded random ranking and descending amount ranking address RQ1.
- **Interpretable model** Balanced Logistic Regression addresses RQ1 and RQ3.
- **Flexible model** LightGBM [2] captures nonlinear relationships. An identical configuration with additional historical statistics addresses RQ2.
- **Preprocessing** Fit imputation, scaling and encodings on training data. Frequency-encode high-cardinality categories. Exclude transaction ID, outcome and raw split time.

## 5. Validation and Evaluation

- **Split** Use chronological 70/15/15 training, validation and test periods, retaining equal timestamps together. Historical features use strictly earlier timestamps and exclude labels.
- **Tuning** Use fixed estimator settings and seed 42, with configuration comparison on validation. Holdout comparisons are complete. Further changes use validation, and test diagnostics are identified as post-hoc.
- **Metrics** Report average precision [3], Precision@K, Recall@K and labeled fraud found at K=500, with K=100 as sensitivity analysis. K applies to the period rather than each day.
- **Interpretation and robustness** Compare period outcomes, errors, standardized-feature extremes and LR probability versus decision-score rankings. Single-seed results have no confidence intervals and support baseline-level conclusions.

## 6. Expected Contribution

We evaluate investigation yield at fixed capacity and the incremental value of historical features. A prototype links evaluated Top-K selection to an auditable queue, including high-score cases. Scores are uncalibrated ranking signals. Dataset anonymity, one holdout and simulated review operations limit conclusions about deployed banking performance.

## 7. Timeline

Final submission is due 29 November 2026 at 11:59 PM. Internal completion is 25 November.

| Week | Dates | Milestone |
|---|---|---|
| 1 | 5–8 Nov | Audit data, preserve existing results and finalize proposal |
| 2 | 9–15 Nov | Verify baseline implementation, features and capacity-based review |
| 3 | 16–22 Nov | Complete interpretation and robustness analysis, freeze report |
| 4 | 23–29 Nov | Complete by 25 Nov, prepare at most 10 slides, 8-minute recording and code appendix, submit by 29 Nov |

## References

[1] Kaggle. IEEE-CIS Fraud Detection. 2019. https://www.kaggle.com/competitions/ieee-fraud-detection. Accessed 5 November 2026.

[2] Ke, G., et al. LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NeurIPS, 2017.

[3] Saito, T., and Rehmsmeier, M. The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets. PLOS ONE, 10(3), e0118432, 2015.
