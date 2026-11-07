"""Compare fraud rankings on validation, or on the explicitly requested test holdout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

# 直接运行这个脚本时，也能从项目根目录导入 fraudml。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fraudml.data import (
    chronological_split,
    feature_columns,
    load_ieee_cis,
    validate_feature_config,
)
from fraudml.decision import amount_baseline, random_baseline
from fraudml.features import add_past_entity_stats, as_categorical
from fraudml.metrics import evaluate
from fraudml.models import TabularFraudModel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transaction", required=True, help="path to train_transaction.csv")
    parser.add_argument("--identity", help="optional path to train_identity.csv")
    parser.add_argument("--entity-cols", nargs="*", default=[],
                        help="entity proxy columns for strictly-past statistics")
    parser.add_argument("--categorical-codes", nargs="*", default=[],
                        help="numeric category codes to encode, e.g. card1 addr1")
    parser.add_argument("--k", nargs="+", type=int, default=[100, 500],
                        help="review budgets across the complete evaluation period")
    kinds = ("random", "amount", "logistic", "lightgbm")
    parser.add_argument("--models", nargs="+", default=kinds, choices=kinds)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--valid-fraction", type=float, default=0.15)
    # 先用验证期比较配置，固定实验方案后再显式评估测试期。
    parser.add_argument("--split", choices=("validation", "test"), default="validation",
                        help="use test after freezing the experiment settings")
    parser.add_argument("--out", type=Path, help="use a distinct JSON filename for each run")
    args = parser.parse_args()

    default_report = "validation_metrics.json" if args.split == "validation" else "final_metrics.json"
    report_path = args.out or Path("artifacts") / default_report
    models_path = report_path.with_name(f"{report_path.stem}_models")
    # 每次实验使用单独的报告和模型目录，方便追溯配置和结果。
    if report_path.exists() or models_path.exists():
        parser.error(f"report or model directory already exists; choose another --out: {report_path}")

    frame = load_ieee_cis(args.transaction, args.identity)
    try:
        validate_feature_config(frame, args.entity_cols, args.categorical_codes)
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    if args.categorical_codes:
        frame = as_categorical(frame, args.categorical_codes)
    if args.entity_cols:
        # 模拟逐笔评分，只汇总严格早于当前交易的输入，包括评估期内较早的记录。
        frame = add_past_entity_stats(frame, args.entity_cols)
    split = chronological_split(frame, train_fraction=args.train_fraction,
                                valid_fraction=args.valid_fraction)
    periods = {"train": split.train, "validation": split.valid, "test": split.test}
    if any(period.empty for period in periods.values()):
        parser.error("chronological split has an empty period; check timestamps and fractions")

    cols = feature_columns(frame)
    scored_period = periods[args.split]
    y = scored_period["isFraud"]
    X = scored_period[cols]
    # K 是整个评估批次的审核名额。保留首次出现的有效预算。
    budgets = list(dict.fromkeys(k for k in args.k if 1 <= k <= len(X)))
    invalid_budgets = [k for k in args.k if not 1 <= k <= len(X)]
    if invalid_budgets:
        print(f"[warning] K values {invalid_budgets} outside 1..{len(X)}; skipped", file=sys.stderr)
    if not budgets:
        parser.error(f"no valid K for {args.split} period with {len(X)} rows")

    results = {}
    artifacts = {}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{report_path.stem}-", dir=report_path.parent) as scratch:
        staged_models = Path(scratch) / "models"
        for kind in dict.fromkeys(args.models):
            started = perf_counter()
            name = kind
            if kind == "random":
                scores = random_baseline(len(X), seed=args.seed)
            elif kind == "amount":
                name = "amount_descending"
                scores = amount_baseline(X)
            else:
                model = TabularFraudModel(kind, random_state=args.seed)
                try:
                    # 预处理和模型都只在训练期拟合，选定的评估期只用于评分和算指标。
                    model.fit(split.train[cols], split.train["isFraud"])
                except ImportError as error:
                    parser.error(f"could not train {kind}: {error}")
                scores = model.predict_proba(X)
                # 保存特征配置，批量预测时可按训练口径处理类别与历史统计。
                model.entity_cols = tuple(args.entity_cols)
                model.categorical_codes = tuple(args.categorical_codes)
                staged_models.mkdir(exist_ok=True)
                filename = f"{kind}_seed{args.seed}.joblib"
                model.save(staged_models / filename)
                artifacts[kind] = (Path(models_path.name) / filename).as_posix()
            results[name] = {
                **{f"k={k}": evaluate(y, scores, k) for k in budgets},
                "runtime_seconds": round(perf_counter() - started, 3),
            }

        # 报告记录输入、时间边界、种子、指标与模型路径，便于复查和复跑。
        payload = {
            "evaluation_split": args.split,
            "review_budgets": budgets,
            "seed": args.seed,
            "transaction_file": args.transaction,
            "identity_file": args.identity,
            "entity_cols": args.entity_cols,
            "categorical_codes": args.categorical_codes,
            "split": {
                "train_fraction": args.train_fraction,
                "valid_fraction": args.valid_fraction,
                "periods": {
                    name: {
                        "rows": len(period),
                        "start_dt": int(period["TransactionDT"].min()),
                        "end_dt": int(period["TransactionDT"].max()),
                    }
                    for name, period in periods.items()
                },
            },
            "evaluation_fraud_count": int(y.sum()),
            "evaluation_fraud_rate": float(y.mean()),
            "model_artifacts": artifacts,
            "results": results,
        }
        report = json.dumps(payload, indent=2) + "\n"
        with report_path.open("x", encoding="utf-8") as output:
            if artifacts:
                staged_models.rename(models_path)
            output.write(report)

    print(f"wrote {report_path}")
    for k in budgets:
        print(f"\nK = {k} ({args.split}: {len(X)} rows, fraud rate = {y.mean():.4f})")
        print("| model | AP | Precision@K | Recall@K | fraud found |")
        print("|---|---:|---:|---:|---:|")
        for name, metrics in results.items():
            row = metrics[f"k={k}"]
            print(f"| {name} | {row['pr_auc']:.4f} | {row['precision_at_k']:.4f} "
                  f"| {row['recall_at_k']:.4f} | {row['hits_at_k']} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
