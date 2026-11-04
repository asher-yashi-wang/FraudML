# FraudML

Leakage-aware fraud detection framework for an IEEE-CIS course project. It provides a runnable tabular core (Logistic Regression and optional LightGBM), an optional PyTorch Geometric GraphSAGE extension, chronological evaluation, investigation Top-K selection, **reviewer-maintained policies and a human-in-the-loop review workflow**, plus course-document templates.

> **Research integrity:** no claims or pre-filled IEEE-CIS results are included. Run experiments after obtaining the data and record the actual metrics.

## Quick start (synthetic smoke test)

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e '.[dev]'
fraudml make-demo --out data/demo.csv --rows 1500
fraudml train --data data/demo.csv --model logistic --out artifacts/logistic.joblib --k 50
fraudml predict --model artifacts/logistic.joblib --data data/demo.csv --out artifacts/scored.csv
pytest -q
```

## Policy and human review workflow

Score transactions, then apply the versioned policy managed by risk operations:

```bash
fraudml predict --model artifacts/logistic.joblib --data data/demo.csv --out artifacts/scored.csv
fraudml triage --data artifacts/scored.csv --policy configs/risk_policy.json --out artifacts/decisions.csv --review-queue artifacts/review_queue.csv
fraudml review --case-id case-PASTE_ID_HERE --outcome confirmed_fraud --reviewer alice --note "Verified after investigation"
```

The policy produces only `APPROVE`, `REVIEW`, or `CHALLENGE` decisions. `CHALLENGE` is a hold/step-up-verification recommendation, **not a fraud confirmation**. Only an investigator's `confirmed_fraud` review outcome is a confirmed label. Triage and review events are appended to `artifacts/audit.jsonl`; keep raw PII out of audit notes.

For LightGBM: `pip install -e '.[lightgbm,dev]'` and use `--model lightgbm`. The optional GNN requires a PyTorch/PyG build compatible with your CUDA driver; see [docs/usage.md](docs/usage.md).

## Scope and layout

| Path | Purpose |
|---|---|
| `fraudml/data.py` | IEEE-CIS loading and contiguous chronological split |
| `fraudml/features.py` | train-only preprocessing and strictly-past entity statistics |
| `fraudml/models/` | train/save/load/predict_proba tabular models |
| `fraudml/graph/` | optional causal transaction–entity GraphSAGE building blocks |
| `fraudml/decision.py` | risk ranking and Top-K investigation policy |
| `fraudml/policy.py`, `fraudml/workflow.py` | versioned policy, triage, review queue, audit events |
| `scripts/run_experiment.py` | one-shot final evaluation on the holdout test split |
| `docs/` | data, architecture, experiments, usage, proposal and report templates |

## Data availability and restrictions

IEEE-CIS Fraud Detection is distributed through Kaggle. Download it yourself in accordance with Kaggle and competition terms; do **not** commit its CSV files, credentials, or derived identifiers. The dataset is historical and its fields are partially anonymized; entity-like columns are proxies, not verified account/device identity. It contains labels for the competition training set, while Kaggle test labels are unavailable. This repository evaluates only holdout portions of labeled training data.

## Safety against leakage

The baseline uses time-contiguous splits, fits preprocessing only on training data, and derives entity aggregates only from earlier records. For GNN work, build/serve each graph snapshot using only transactions and entity edges known at that prediction time. Do not let test-time nodes or future edges message into earlier transactions.

## Reproducibility

Set a seed, save the exact config/environment and model artifact per run. Report PR-AUC and capacity-constrained Recall@K/Precision@K along with random and amount-ranking baselines. See `docs/experiments.md`.
