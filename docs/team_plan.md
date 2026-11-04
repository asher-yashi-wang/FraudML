# Five-person plan to 25 November 2026

| Owner | Responsibility | Deliverable | Date |
|---|---|---|---|
| Member 1 | Data governance + split | data audit, leakage checklist | Nov 3–7 |
| Member 2 | Tabular baseline | LR/LightGBM runs and artifacts | Nov 3–10 |
| Member 3 | Features + decisions | past-only aggregates, Top-K analysis | Nov 6–13 |
| Member 4 | GNN extension | causal graph proof-of-concept / go-no-go | Nov 8–17 |
| Member 5 | Evaluation + writing | metric tables, figures, report integration | Nov 10–25 |

## Milestones

- **Nov 3–7:** download/check data terms, establish chronological holdout, run smoke test and LR.
- **Nov 8–12:** LightGBM and decision baselines; code review for feature leakage.
- **Nov 12:** GNN go/no-go. Stop the GNN if causal snapshots or stable sampled training cannot be demonstrated.
- **Nov 13–17:** GNN or relational-feature ablation; freeze test protocol.
- **Nov 18–20:** final actual runs, figures, error analysis.
- **Nov 21–25:** report, reproducibility audit, presentation, package release.

Everyone reviews the held-out test evaluation before it is run. Member roles indicate ownership, not sole authorship.
