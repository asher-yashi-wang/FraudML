from __future__ import annotations
import numpy as np
from sklearn.metrics import average_precision_score

def pr_auc(y_true, scores) -> float:
    return float(average_precision_score(y_true, scores))

def top_k_metrics(y_true, scores, k: int) -> dict[str, float]:
    y, s = np.asarray(y_true), np.asarray(scores)
    if len(y) != len(s) or not 1 <= k <= len(y): raise ValueError("k must be between 1 and sample count")
    top = np.argsort(-s, kind="stable")[:k]
    hits = int(y[top].sum()); positives = int(y.sum())
    return {"k": k, "precision_at_k": hits / k, "recall_at_k": hits / positives if positives else 0.0, "hits_at_k": hits}

def evaluate(y_true, scores, k: int) -> dict[str, float]:
    return {"pr_auc": pr_auc(y_true, scores), **top_k_metrics(y_true, scores, k)}
