"""Human-in-the-loop triage, review evidence, and append-only audit events."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any
import pandas as pd
from .policy import RiskPolicy

DECISIONS = {"APPROVE", "REVIEW", "CHALLENGE"}
OUTCOMES = {"confirmed_fraud", "not_fraud", "needs_more_information"}

def _case_ids(frame: pd.DataFrame, policy: RiskPolicy, transaction_id_col="TransactionID") -> pd.Series:
    """Versioned, deterministic case ids: sha256(policy_version:value)[:12]."""
    if transaction_id_col in frame:
        values = frame[transaction_id_col].astype(str)
    else:
        values = pd.Series(frame.index.astype(str), index=frame.index)
    joined = policy.version + ":" + values
    return pd.Series("case-" + joined.map(lambda s: hashlib.sha256(s.encode()).hexdigest()[:12]),
                     index=frame.index)

def triage(frame: pd.DataFrame, policy: RiskPolicy, score_col="risk_score", amount_col="TransactionAmt") -> pd.DataFrame:
    """Assign an operational decision and reasons; never assign a fraud label.

    Vectorized: decision logic is evaluated column-wise instead of row-by-row,
    so it scales to the full dataset. Blocked values and score thresholds
    produce the same decisions and reason text as the earlier row loop.
    """
    if score_col not in frame: raise KeyError(f"Missing model score column: {score_col}")
    policy.validate(); out = frame.copy()
    scores = pd.to_numeric(out[score_col], errors="coerce")
    if scores.isna().any(): raise ValueError("risk_score contains missing values")

    high_score = scores >= policy.challenge_threshold
    review_score = ~high_score & (scores >= policy.review_threshold)
    approve = ~high_score & ~review_score

    blocked = pd.Series(False, index=out.index)
    parts: list[tuple[pd.Series, str]] = []
    for column, values in policy.blocked_values.items():
        if column not in out or not values: continue
        hit = out[column].notna() & out[column].astype(str).isin([str(x) for x in values])
        blocked |= hit
        parts.append((hit, f"policy blocked value: {column}"))

    decisions = pd.Series("", index=out.index, dtype="object")
    decisions[blocked] = "CHALLENGE"
    decisions[~blocked & high_score] = "CHALLENGE"
    decisions[~blocked & review_score] = "REVIEW"
    decisions[~blocked & approve] = "APPROVE"

    parts.append((~blocked & high_score,
                  f"model score >= challenge threshold ({policy.challenge_threshold:.2f})"))
    parts.append((~blocked & review_score,
                  f"model score >= review threshold ({policy.review_threshold:.2f})"))
    parts.append((~blocked & approve, "score below review threshold"))
    if amount_col in out:
        amounts = pd.to_numeric(out[amount_col], errors="coerce")
        parts.append((amounts >= policy.high_amount,
                      f"amount >= policy high-amount marker ({policy.high_amount:g})"))

    reasons = pd.Series("", index=out.index, dtype="object")
    for mask, text in parts:
        reasons = reasons.mask(mask, reasons[mask] + text + "; ")
    reasons = reasons.str.rstrip("; ")

    out["decision"] = decisions
    out["decision_reason"] = reasons
    out["policy_version"] = policy.version
    out["review_case_id"] = _case_ids(out, policy)
    return out

def review_queue(triaged: pd.DataFrame) -> pd.DataFrame:
    """Only uncertain cases enter the manual review queue."""
    return triaged.loc[triaged["decision"] == "REVIEW"].sort_values("risk_score", ascending=False, kind="stable").copy()

def append_audit(path: str | Path, event: dict[str, Any]) -> None:
    """Append-only JSONL audit record; do not write raw personally identifying fields."""
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"at_utc": datetime.now(timezone.utc).isoformat(), **event}
    with target.open("a", encoding="utf-8") as fp: fp.write(json.dumps(payload, ensure_ascii=False) + "\n")

def record_review(audit_path: str | Path, case_id: str, outcome: str, reviewer: str, note="") -> None:
    if outcome not in OUTCOMES: raise ValueError(f"outcome must be one of {sorted(OUTCOMES)}")
    if not reviewer.strip(): raise ValueError("reviewer is required")
    append_audit(audit_path, {"event": "review_completed", "review_case_id": case_id, "outcome": outcome, "reviewer": reviewer, "note": note})
