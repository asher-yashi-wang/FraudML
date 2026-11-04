# Experiment protocol

## Required comparisons

1. Random ranking (fixed seed).
2. Amount descending ranking.
3. Logistic Regression.
4. LightGBM when dependency/environment permits.
5. Optional GraphSAGE and an ablation without relational information.

Use one fixed chronological test period that is never tuned on. Tune on the validation period; retrain only under a documented protocol. For every model report PR-AUC, Precision@K, Recall@K, K, positive counts, split date/range, seed, and runtime. K should reflect a plausible daily investigation capacity (e.g., 100/500); it is not a universal constant.

Once tuning is finished and the protocol is frozen, produce the final numbers with `python scripts/run_experiment.py --transaction <csv> --identity <csv> --k 100 500 --entity-cols <cols> ...`; run it with and without `--entity-cols` for the feature ablation. It writes `artifacts/final_metrics.json`, saves model artifacts, and prints the comparison table.

| Model | PR-AUC | Precision@K | Recall@K | K | Notes |
|---|---:|---:|---:|---:|---|
| Random | pending | pending | pending | pending | actual run required |
| Amount | pending | pending | pending | pending | actual run required |
| Logistic | pending | pending | pending | pending | actual run required |
| LightGBM | pending | pending | pending | pending | actual run required |
| GraphSAGE | pending | pending | pending | pending | actual run required |

Do not replace `pending` with estimates. Include confidence intervals or repeated temporal windows if time permits, and report negative results.
