"""Rolling time-window robustness evaluation (task B).

Evaluates the base configuration (random / amount / LightGBM, K = 500) on
several chronological test windows. The first window reproduces the frozen
70/15/15 holdout, so it MUST match artifacts/final_metrics.json exactly
(LightGBM 465 hits, amount 10, random 13) — treat any mismatch as a bug.

These windows are post-hoc robustness evidence, not model selection:
no setting may be changed after seeing them.

Run from the repo root:
  python scripts/rolling_windows.py \
      --transaction data/ieee-cis/train_transaction.csv \
      --categorical-codes card1 addr1 \
      --k 500 --seed 42 --out artifacts/rolling_windows.json
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from fraudml.data import chronological_split, feature_columns, load_ieee_cis
from fraudml.decision import amount_baseline, random_baseline
from fraudml.features import as_categorical
from fraudml.metrics import evaluate
from fraudml.models import TabularFraudModel

DEFAULT_WINDOWS = [(0.70, 0.15), (0.60, 0.15), (0.80, 0.10)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--transaction", required=True)
    ap.add_argument("--categorical-codes", nargs="*", default=[])
    ap.add_argument("--windows", nargs="+", default=["0.70/0.15", "0.60/0.15", "0.80/0.10"],
                    help="train/valid fractions; the first should be 0.70/0.15")
    ap.add_argument("--k", nargs="+", type=int, default=[500])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-rows", type=int, default=None,
                    help="smoke-test only: read the first N rows")
    ap.add_argument("--out", default="artifacts/rolling_windows.json")
    args = ap.parse_args()

    windows = [tuple(float(x) for x in w.split("/")) for w in args.windows]
    for tf, vf in windows:
        if not 0 < tf < 1 or not 0 < vf < 1 or tf + vf >= 1:
            ap.error(f"bad window {tf}/{vf}")

    frame = load_ieee_cis(args.transaction, None)
    if args.max_rows:
        import pandas as pd
        frame = pd.read_csv(args.transaction, nrows=args.max_rows)
    if args.categorical_codes:
        frame = as_categorical(frame, args.categorical_codes)
    cols = feature_columns(frame)

    results = {}
    print("| window | n_test | fraud rate | amount@500 | random@500 | LGBM hits@500 | LGBM P@500 | LGBM AP |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for tf, vf in windows:
        split = chronological_split(frame, train_fraction=tf, valid_fraction=vf)
        y = split.test["isFraud"].to_numpy()
        X = split.test[cols]
        model = TabularFraudModel("lightgbm", random_state=args.seed)
        model.fit(split.train[cols], split.train["isFraud"])
        scores = model.predict_proba(X)
        entry = {
            "train_fraction": tf,
            "valid_fraction": vf,
            "n_test": int(len(split.test)),
            "fraud_rate": round(float(y.mean()), 5),
            "amount": {f"k={k}": evaluate(y, amount_baseline(X), k) for k in args.k},
            "random": {f"k={k}": evaluate(y, random_baseline(len(X), seed=args.seed), k) for k in args.k},
            "lightgbm": {f"k={k}": evaluate(y, scores, k) for k in args.k},
        }
        results[f"{tf:g}/{vf:g}"] = entry
        k = args.k[0]
        row = entry["lightgbm"][f"k={k}"]
        print(f"| {tf:g}/{vf:g} | {entry['n_test']} | {entry['fraud_rate']*100:.2f}% "
              f"| {entry['amount'][f'k={k}']['hits_at_k']} | {entry['random'][f'k={k}']['hits_at_k']} "
              f"| {row['hits_at_k']} | {row['precision_at_k']:.3f} | {row['pr_auc']:.4f} |")

    payload = {
        "note": "post-hoc rolling-window robustness evidence; the frozen test protocol is "
                "untouched and no setting was changed after viewing these windows",
        "seed": args.seed,
        "budgets": args.k,
        "categorical_codes": args.categorical_codes,
        "windows": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    print("check: the first row (0.7/0.15) must match artifacts/final_metrics.json "
          "(465 / 10 / 13 at K=500). Any mismatch is a bug, not a finding.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
