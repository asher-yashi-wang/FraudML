"""Data-layer temporal drift audit (task A).

Evidence that the fraud distribution and feature magnitudes drift across
periods, upstream of any model. Outputs:
  artifacts/drift_audit.json         period + weekly fraud rates, drift table
  artifacts/drift_audit_features.csv top drifted numeric features
  artifacts/drift_audit_weekly.png   weekly fraud rate with split boundaries

Run from the repo root:
  python scripts/drift_audit.py \
      --transaction data/ieee-cis/train_transaction.csv \
      --out-dir artifacts
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import tempfile
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from fraudml.data import chronological_split, validate_timestamps

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplcache"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--transaction", required=True)
    ap.add_argument("--train-fraction", type=float, default=0.70)
    ap.add_argument("--valid-fraction", type=float, default=0.15)
    ap.add_argument("--out-dir", default="artifacts")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="smoke-test only: read the first N rows")
    ap.add_argument("--top-n", type=int, default=10)
    args = ap.parse_args()

    df = pd.read_csv(args.transaction, nrows=args.max_rows)
    validate_timestamps(df)
    split = chronological_split(df, train_fraction=args.train_fraction,
                                valid_fraction=args.valid_fraction)

    # --- 1. fraud prevalence per period and per week -------------------------
    periods = {}
    weekly_rows = []
    for name, part in [("train", split.train), ("validation", split.valid), ("test", split.test)]:
        rate = float(part["isFraud"].mean())
        periods[name] = {
            "rows": int(len(part)),
            "fraud": int(part["isFraud"].sum()),
            "fraud_rate": round(rate, 5),
            "start_dt": int(part["TransactionDT"].min()),
            "end_dt": int(part["TransactionDT"].max()),
        }
        week = (part["TransactionDT"] // 604800).astype(int)
        tmp = pd.DataFrame({"period": name, "week": week, "y": part["isFraud"].to_numpy()})
        weekly_rows.append(tmp)
    weekly = pd.concat(weekly_rows, ignore_index=True)
    weekly_rate = (weekly.groupby(["period", "week"])["y"]
                   .agg(["sum", "size"]).reset_index())
    weekly_rate["rate"] = weekly_rate["sum"] / weekly_rate["size"]
    weekly_rate = weekly_rate.rename(columns={"sum": "fraud", "size": "rows"})

    # --- 2. numeric feature drift: max |z| shift train -> test ---------------
    numeric = [c for c in df.select_dtypes(include="number").columns
               if c not in {"TransactionID", "TransactionDT", "isFraud"}]
    train_num = split.train[numeric].to_numpy(dtype=float)
    mean = np.nanmean(train_num, axis=0)
    std = np.nanstd(train_num, axis=0)
    std[std == 0] = np.nan  # constant features have no drift

    def zmax(part):
        raw = part[numeric].to_numpy(dtype=float)
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            z = np.abs((raw - mean) / std)
            return np.nanmax(z, axis=0)

    z_train, z_valid, z_test = zmax(split.train), zmax(split.valid), zmax(split.test)
    drift = pd.DataFrame({
        "feature": numeric,
        "max_abs_z_train": np.round(z_train, 1),
        "max_abs_z_validation": np.round(z_valid, 1),
        "max_abs_z_test": np.round(z_test, 1),
        "test_minus_train": np.round(z_test - z_train, 1),
    }).sort_values("test_minus_train", ascending=False)
    top = drift.head(args.top_n)

    # --- 3. outputs ----------------------------------------------------------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "note": "post-hoc data audit; the frozen test protocol is untouched",
        "split": {"train_fraction": args.train_fraction, "valid_fraction": args.valid_fraction},
        "periods": periods,
        "weekly_fraud_rates": weekly_rate.to_dict(orient="records"),
        "top_drifted_features": top.to_dict(orient="records"),
    }
    (out_dir / "drift_audit.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    top.to_csv(out_dir / "drift_audit_features.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4))
    boundaries = [periods["train"]["end_dt"] // 604800, periods["validation"]["end_dt"] // 604800]
    for period_name, group in weekly_rate.groupby("period"):
        ax.plot(group["week"], group["rate"] * 100, marker=".", label=period_name)
    for b in boundaries:
        ax.axvline(b, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("week index (TransactionDT // 604800)")
    ax.set_ylabel("labeled fraud rate (%)")
    ax.set_title("Weekly labeled fraud rate across periods (post-hoc data audit)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "drift_audit_weekly.png", dpi=150)
    print(f"wrote {out_dir / 'drift_audit.json'}, {out_dir / 'drift_audit_features.csv'}, {out_dir / 'drift_audit_weekly.png'}")

    print("\nFraud prevalence by period:")
    for name, info in periods.items():
        print(f"  {name:10s} rows={info['rows']:7d} fraud={info['fraud']:5d} "
              f"rate={info['fraud_rate']*100:.2f}%")
    print("\nTop drifted features (max |z| shift train -> test):")
    print(top[["feature", "max_abs_z_train", "max_abs_z_test", "test_minus_train"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
