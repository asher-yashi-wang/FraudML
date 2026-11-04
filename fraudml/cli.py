from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from .data import chronological_split, feature_columns
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
 p0=sub.add_parser("policy-init", help="write an editable starter policy"); p0.add_argument("--out",default="configs/risk_policy.json"); p0.add_argument("--reviewer",default="risk-operations-owner")
 x=sub.add_parser("triage", help="apply approved policy to scored transactions"); x.add_argument("--data",required=True); x.add_argument("--policy",default="configs/risk_policy.json"); x.add_argument("--out",required=True); x.add_argument("--review-queue",required=True); x.add_argument("--audit",default="artifacts/audit.jsonl")
 r=sub.add_parser("review", help="record an investigator's outcome in the audit log"); r.add_argument("--case-id",required=True); r.add_argument("--outcome",choices=["confirmed_fraud","not_fraud","needs_more_information"],required=True); r.add_argument("--reviewer",required=True); r.add_argument("--note",default=""); r.add_argument("--audit",default="artifacts/audit.jsonl")
 a=p.parse_args()
 if a.command=="make-demo":
  Path(a.out).parent.mkdir(parents=True,exist_ok=True); make_synthetic_transactions(a.rows).to_csv(a.out,index=False); return
 if a.command=="train":
  df=pd.read_csv(a.data); split=chronological_split(df); cols=feature_columns(df); m=TabularFraudModel(a.model).fit(split.train[cols],split.train.isFraud); m.save(Path(a.out)); print(json.dumps({"split":"validation","note":"holdout test split is reserved; tune on validation and run scripts/run_experiment.py once for final numbers","metrics":evaluate(split.valid.isFraud,m.predict_proba(split.valid[cols]),min(a.k,len(split.valid)))},indent=2)); return
 if a.command=="predict":
  df=pd.read_csv(a.data); cols=feature_columns(df); df["risk_score"]=TabularFraudModel.load(a.model).predict_proba(df[cols]); df.to_csv(a.out,index=False)
 if a.command=="policy-init": RiskPolicy(reviewer=a.reviewer).save(a.out)
 if a.command=="triage":
  policy=RiskPolicy.load(a.policy); decisions=triage(pd.read_csv(a.data), policy); Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.review_queue).parent.mkdir(parents=True,exist_ok=True); decisions.to_csv(a.out,index=False); review_queue(decisions).to_csv(a.review_queue,index=False); append_audit(a.audit,{"event":"triage_completed","policy_version":policy.version,"reviewer":policy.reviewer,"total":len(decisions),"review_cases":int((decisions.decision=="REVIEW").sum()),"challenge_cases":int((decisions.decision=="CHALLENGE").sum())})
 if a.command=="review": record_review(a.audit,a.case_id,a.outcome,a.reviewer,a.note)
if __name__ == "__main__": main()
