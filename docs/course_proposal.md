# Project Proposal — FraudML: Leakage-Aware Fraud Detection and Investigation Prioritization

**Course:** DSC5001<br>
**Team:** [Add each team member's name and SID]<br>
**Due date:** 25 November 2026

## Motivation and question

Fraud teams investigate a limited number of transactions. On a future holdout, can model-based ranking find more labeled fraud than amount-based ranking when both methods can flag only 500 transactions? We will also test whether adding strictly earlier entity statistics improves LightGBM's ranking.

## Data

We will use the Kaggle IEEE-CIS Fraud Detection training data after complying with its terms. It is a labeled, anonymized historical transaction dataset with a relative time field (`TransactionDT`). Its entity-like fields are proxies rather than verified identities. We will not claim evaluation on Kaggle's unlabeled test set.

## Method

We will hold out consecutive future time periods for validation and testing. Preprocessing is fitted on the training period only. Baselines are random ordering, descending amount, Logistic Regression, and LightGBM. The primary outcomes are PR-AUC, Precision@K, and Recall@K. If feasible, we will add a transaction–entity GraphSAGE extension with graph snapshots that exclude future edges/features. The GNN is optional and will be dropped if it cannot meet leakage and reproducibility checks.

## Expected contribution and risks

The deliverable is a reproducible Python package and documented experimental protocol, rather than a claim that a complex model necessarily wins. Main risks include anonymized entity semantics, temporal leakage, class imbalance, and GNN resource limits. Mitigations are time splits, train-only transforms, explicit baselines, and sampled mini-batch GNN training.

## References

1. Kaggle. [IEEE-CIS Fraud Detection competition data](https://www.kaggle.com/competitions/ieee-fraud-detection/data).
2. Ke, G., et al. (2017). [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html). Advances in Neural Information Processing Systems, 30.
3. Saito, T., & Rehmsmeier, M. (2015). [The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0118432). PLOS ONE, 10(3), e0118432.
