"""One-shot final evaluation on the untouched holdout test period.

Run this exactly once, after all tuning on the validation split is done and
the protocol is frozen (see docs/experiments.md). It fits on the training
period, ranks the test period, and writes a metrics JSON plus model
artifacts. Nothing here is tuned against the test split.

Examples:
    python scripts/run_experiment.py \\
        --transaction data/ieee-cis/train_transaction.csv \\
        --identity data/ieee-cis/train_identity.csv \\
        --entity-cols card1 card4 DeviceInfo \\
        --k 100 500 --seed 42 \\
        --out artifacts/final_metrics.json

Run the same command with and without --entity-cols for the feature
ablation comparison (LightGBM with vs. without strictly-past statistics).
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from fraudml.data import chronological_split, feature_columns, load_ieee_cis
from fraudml.decision import amount_baseline, random_baseline
from fraudml.features import add_past_entity_stats, as_categorical
from fraudml.metrics import evaluate
from fraudml.models import TabularFraudModel


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--transaction", required=True,
                    help="path to train_transaction.csv")
    ap.add_argument("--identity", default=None,
                    help="optional path to train_identity.csv")
    ap.add_argument("--entity-cols", nargs="*", default=[],
                    help="entity proxy columns for strictly-past stats (ablation: rerun without)")
    ap.add_argument("--categorical-codes", nargs="*", default=[],
                    help="integer-coded entity columns to treat as categorical, e.g. "
                         "card1 card2 card5 card6 addr1 addr2 D1 D4 D6 D8 D10 D11 D12 D15 id_30 id_31 id_33")
    ap.add_argument("--k", nargs="+", type=int, default=[100, 500],
                    help="review-budget sizes to evaluate")
    ap.add_argument("--models", nargs="+", default=["random", "amount", "logistic", "lightgbm"],
                    choices=["random", "amount", "logistic", "lightgbm"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-fraction", type=float, default=0.70)
    ap.add_argument("--valid-fraction", type=float, default=0.15)
    ap.add_argument("--out", default="artifacts/final_metrics.json")
    args = ap.parse_args()

    df = load_ieee_cis(args.transaction, args.identity)
    if args.categorical_codes:
        df = as_categorical(df, args.categorical_codes)
    # Strictly-past aggregates are safe to compute before splitting: because
    # the splits are chronological, every row only ever sees earlier rows,
    # which is exactly what would be known at its prediction time.
    if args.entity_cols:
        df = add_past_entity_stats(df, args.entity_cols)
    split = chronological_split(df, train_fraction=args.train_fraction,
                                valid_fraction=args.valid_fraction)
    cols = feature_columns(df)
    y = split.test["isFraud"]
    X = split.test[cols]

    n_test = len(X)
    too_large = [k for k in args.k if k > n_test]
    if too_large:
        print(f"[warning] K values {too_large} exceed test size {n_test}; skipped", file=sys.stderr)
    args.k = [k for k in args.k if 1 <= k <= n_test]
    if not args.k:
        print(f"[error] no valid K remains; test split has only {n_test} rows", file=sys.stderr)
        return 2

    results: dict[str, dict[str, dict]] = {}
    def record(name: str, scores) -> None:
        results[name] = {f"k={k}": evaluate(y, scores, k) for k in args.k}

    if "random" in args.models:
        record("random", random_baseline(len(X), seed=args.seed))
    if "amount" in args.models:
        record("amount_descending", amount_baseline(X))

    for kind in ("logistic", "lightgbm"):
        if kind not in args.models:
            continue
        t0 = time.time()
        try:
            model = TabularFraudModel(kind, random_state=args.seed)
            model.fit(split.train[cols], split.train["isFraud"])
            scores = model.predict_proba(X)
        except ImportError:
            print(f"[skip] {kind}: optional dependency not installed", file=sys.stderr)
            continue
        results[kind] = {f"k={k}": evaluate(y, scores, k) for k in args.k}
        results[kind]["runtime_seconds"] = round(time.time() - t0, 1)
        out_dir = Path(args.out).parent / "models"
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save(out_dir / f"{kind}_seed{args.seed}.joblib")

    payload = {
        "seed": args.seed,
        "transaction_file": args.transaction,
        "identity_file": args.identity,
        "entity_cols": args.entity_cols,
        "categorical_codes": args.categorical_codes,
        "split": {
            "train_fraction": args.train_fraction,
            "valid_fraction": args.valid_fraction,
            "n_train": int(len(split.train)),
            "n_valid": int(len(split.valid)),
            "n_test": int(len(split.test)),
            "train_end_dt": int(split.train["TransactionDT"].max()),
            "valid_end_dt": int(split.valid["TransactionDT"].max()),
            "test_end_dt": int(split.test["TransactionDT"].max()),
            "test_fraud_rate": round(float(y.mean()), 5),
        },
        "results": results,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")

    for k in args.k:
        print(f"\nK = {k} (n_test = {payload['split']['n_test']}, "
              f"fraud rate = {payload['split']['test_fraud_rate']:.4f})")
        print("| model | PR-AUC | Precision@K | Recall@K | fraud found |")
        print("|---|---:|---:|---:|---:|")
        for name, metrics in results.items():
            row = metrics[f"k={k}"]
            print(f"| {name} | {row['pr_auc']:.4f} | {row['precision_at_k']:.4f} "
                  f"| {row['recall_at_k']:.4f} | {row['hits_at_k']} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
