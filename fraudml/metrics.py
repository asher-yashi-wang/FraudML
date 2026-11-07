from __future__ import annotations
import numpy as np
from sklearn.metrics import average_precision_score

def pr_auc(y_true, scores) -> float:
    # 沿用 pr_auc 字段名，实际计算 AP，即用各阈值的召回增量加权精确率。
    return float(average_precision_score(y_true, scores))

def top_k_metrics(y_true, scores, k: int) -> dict[str, float]:
    y, s = np.asarray(y_true), np.asarray(scores)
    if len(y) != len(s) or not 1 <= k <= len(y): raise ValueError("k must be between 1 and sample count")
    # 降序取前 K 个位置，分数相同时保留输入顺序。
    top = np.argsort(-s, kind="stable")[:k]
    # 标签是 0/1。精确率分母为 K，召回率分母为评估期全部欺诈笔数。
    hits = int(y[top].sum()); positives = int(y.sum())
    return {"k": k, "precision_at_k": hits / k, "recall_at_k": hits / positives if positives else 0.0, "hits_at_k": hits}

def evaluate(y_true, scores, k: int) -> dict[str, float]:
    # AP 使用全部分数，Top-K 指标只看给定预算。
    return {"pr_auc": pr_auc(y_true, scores), **top_k_metrics(y_true, scores, k)}
