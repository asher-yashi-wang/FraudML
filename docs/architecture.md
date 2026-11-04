# Architecture

```text
CSV / synthetic data → chronological split → train-only preprocessing → model → probabilities
                                  │                                      │
                         safe past aggregates                    PR-AUC / Top-K
                                  │                                      ↓
                          optional causal graph → GraphSAGE → policy triage
                                                                  │
                                             APPROVE / REVIEW queue / CHALLENGE
                                                                  │
                                                human outcome → append-only audit log
```

The package deliberately separates data operations, preprocessing, models, metrics, and decision policy. `TabularFraudModel` is a simple Python API: `fit(X,y)`, `predict_proba(X)`, `save(path)`, `load(path)`.

### Policy and reviewer loop

`configs/risk_policy.json` is an editable, versioned policy artifact containing review/challenge score thresholds, high-amount markers, and explicit blocked values. It is intended to be maintained by an authorized risk-operations owner, with approval outside this demo repository. `triage` emits explainable reasons and a review queue. The only valid confirmation comes from `review` via an investigator outcome; score-based decisions must never be stored as confirmed fraud labels. `audit.jsonl` records policy version and workflow events without raw personally identifying details.

### Graph extension

Represent each transaction and each namespaced entity value (for example `DeviceInfo:abc`) as nodes; connect a transaction to entity values observed for that transaction. A production-quality evaluation must create temporal snapshots: when scoring time *t*, features and edges from after *t* may not be available. Start with two GraphSAGE layers, hidden size 64, neighbor sampling `[10,5]`. On an RTX 4070 Ti 12GB, retain the graph in CPU RAM, sample mini-batches (256, reduce to 128 on OOM), and use mixed precision only after correctness is verified.
