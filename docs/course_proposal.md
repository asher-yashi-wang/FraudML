# Project Proposal — FraudML: Leakage-Aware Fraud Detection and Investigation Prioritization

**Course:** [Course name]  
**Team:** [Names]  
**Due date:** 25 November 2026

## Motivation and question

Fraud teams investigate a limited number of transactions, so a useful system must both estimate fraud risk and prioritize a finite queue. Our question is: **do transaction-level and cautiously constructed relational features improve time-forward fraud detection and Top-K investigation yield?**

## Data

We will use the Kaggle IEEE-CIS Fraud Detection training data after complying with its terms. It is a labeled, anonymized historical transaction dataset with a relative time field (`TransactionDT`). Its entity-like fields are proxies rather than verified identities. We will not claim evaluation on Kaggle's unlabeled test set.

## Method

We will hold out consecutive future time periods for validation and testing. Preprocessing is fitted on the training period only. Baselines are random ordering, descending amount, Logistic Regression, and LightGBM. The primary outcomes are PR-AUC, Precision@K, and Recall@K. If feasible, we will add a transaction–entity GraphSAGE extension with graph snapshots that exclude future edges/features. The GNN is optional and will be dropped if it cannot meet leakage and reproducibility checks.

## Expected contribution and risks

The deliverable is a reproducible Python package and documented experimental protocol, rather than a claim that a complex model necessarily wins. Main risks include anonymized entity semantics, temporal leakage, class imbalance, and GNN resource limits. Mitigations are time splits, train-only transforms, explicit baselines, and sampled mini-batch GNN training.
