from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from .data import chronological_split, feature_columns, load_ieee_cis, validate_feature_config, validate_timestamps
from .features import add_past_entity_stats, as_categorical
from .metrics import evaluate
from .models import TabularFraudModel
from .policy import RiskPolicy
from .synthetic import make_synthetic_transactions
from .workflow import append_audit, record_review, review_queue, triage

def main():
 p=argparse.ArgumentParser("fraudml"); sub=p.add_subparsers(dest="command",required=True)
 d=sub.add_parser("make-demo"); d.add_argument("--out",default="data/demo.csv"); d.add_argument("--rows",type=int,default=1000)
 t=sub.add_parser("train"); t.add_argument("--data",required=True); t.add_argument("--model",choices=["logistic","lightgbm"],default="logistic"); t.add_argument("--out",default="artifacts/model.joblib"); t.add_argument("--k",type=int,default=50)
 q=sub.add_parser("predict"); q.add_argument("--model",required=True); q.add_argument("--data",required=True); q.add_argument("--out",required=True)
 q.add_argument("--identity", help="optional identity CSV for scoring transactions")
 q.add_argument("--history", help="optional earlier transaction CSV for history features")
 q.add_argument("--history-identity", help="optional identity CSV for earlier transactions")
 p0=sub.add_parser("policy-init", help="write an editable starter policy"); p0.add_argument("--out",default="configs/risk_policy.json"); p0.add_argument("--reviewer",default="risk-operations-owner")
 x=sub.add_parser("triage", help="apply approved policy to scored transactions"); x.add_argument("--data",required=True); x.add_argument("--policy",default="configs/risk_policy.json"); x.add_argument("--out",required=True); x.add_argument("--review-queue",required=True); x.add_argument("--audit",default="artifacts/audit.jsonl")
 x.add_argument("--review-budget", type=int, help="select exactly K highest-scored rows across all policy decisions")
 r=sub.add_parser("review", help="record an investigator's outcome in the audit log"); r.add_argument("--case-id",required=True); r.add_argument("--outcome",choices=["confirmed_fraud","not_fraud","needs_more_information"],required=True); r.add_argument("--reviewer",required=True); r.add_argument("--note",default=""); r.add_argument("--audit",default="artifacts/audit.jsonl")
 a=p.parse_args()
 if a.command=="make-demo":
  Path(a.out).parent.mkdir(parents=True,exist_ok=True); make_synthetic_transactions(a.rows).to_csv(a.out,index=False); return
 if a.command=="train":
  # 日常训练只打印验证期指标。最终测试由 run_experiment.py 按冻结后的配置执行。
  df=pd.read_csv(a.data); split=chronological_split(df); cols=feature_columns(df); m=TabularFraudModel(a.model).fit(split.train[cols],split.train.isFraud); m.save(Path(a.out)); print(json.dumps({"split":"validation","note":"holdout test split is reserved; tune on validation and run scripts/run_experiment.py once for final numbers","metrics":evaluate(split.valid.isFraud,m.predict_proba(split.valid[cols]),min(a.k,len(split.valid)))},indent=2)); return
 if a.command=="predict":
  model = TabularFraudModel.load(a.model)
  df = load_ieee_cis(a.data, a.identity)
  entity_cols = getattr(model, "entity_cols", ())
  categorical_codes = getattr(model, "categorical_codes", ())
  validate_feature_config(df, entity_cols, categorical_codes)
  features = df
  score_order = None
  if a.history_identity and not a.history:
   p.error("--history-identity requires --history")
  if a.history and not entity_cols:
   p.error("--history requires a model trained with history features")
  if entity_cols:
   validate_timestamps(df)
   if a.history:
    history = load_ieee_cis(a.history, a.history_identity)
    validate_feature_config(history, entity_cols, categorical_codes)
    validate_timestamps(history)
    # 单独提供的历史批次必须早于所有待评分交易，才能在聚合后只取末尾的待评分行。
    if history["TransactionDT"].max() >= df["TransactionDT"].min():
     p.error("--history must contain only transactions strictly earlier than the scoring batch")
    features = pd.concat([history, df], ignore_index=True, sort=False)
   score_order = df["TransactionDT"].sort_values(kind="stable").index
  if categorical_codes:
   features = as_categorical(features, categorical_codes)
  if entity_cols:
   features = add_past_entity_stats(features, entity_cols).tail(len(df))
  expected = getattr(model.pipeline, "feature_names_in_", feature_columns(features))
  missing = set(expected) - set(features.columns)
  if missing:
   p.error(f"model requires missing columns: {sorted(missing)}")
  scores = model.predict_proba(features[list(expected)])
  # 历史聚合按时间重新排序。写回 CSV 时按原行号对齐，确保分数对应正确的交易。
  df["risk_score"] = scores if score_order is None else pd.Series(scores, index=score_order).reindex(df.index)
  Path(a.out).parent.mkdir(parents=True, exist_ok=True)
  df.to_csv(a.out, index=False)
 if a.command=="policy-init": RiskPolicy(reviewer=a.reviewer).save(a.out)
 if a.command=="triage":
  policy = RiskPolicy.load(a.policy)
  # 处置表记录每笔交易的策略动作，审核队列只保留选中调查的交易。
  decisions = triage(pd.read_csv(a.data), policy, review_budget=a.review_budget)
  queue = review_queue(decisions)
  Path(a.out).parent.mkdir(parents=True, exist_ok=True)
  Path(a.review_queue).parent.mkdir(parents=True, exist_ok=True)
  decisions.to_csv(a.out, index=False)
  queue.to_csv(a.review_queue, index=False)
  append_audit(a.audit, {
   "event": "triage_completed",
   "policy_version": policy.version,
   "reviewer": policy.reviewer,
   "total": len(decisions),
   "review_cases": int((decisions.decision == "REVIEW").sum()),
   "challenge_cases": int((decisions.decision == "CHALLENGE").sum()),
   "queue_mode": "top_k" if a.review_budget is not None else "threshold",
   "review_budget": a.review_budget,
   "selected_for_review": len(queue),
  })
 if a.command=="review": record_review(a.audit,a.case_id,a.outcome,a.reviewer,a.note)
if __name__ == "__main__": main()
