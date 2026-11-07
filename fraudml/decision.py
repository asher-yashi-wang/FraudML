from __future__ import annotations
import numpy as np
import pandas as pd

def select_top_k(frame: pd.DataFrame, scores, k: int, score_name="risk_score") -> pd.DataFrame:
    if not 1 <= k <= len(frame): raise ValueError("k must be between 1 and sample count")
    # 按位置把分数附到副本，稳定排序后取前 K 笔。
    result = frame.copy(); result[score_name] = np.asarray(scores)
    return result.sort_values(score_name, ascending=False, kind="stable").head(k)

def random_baseline(n: int, seed=42) -> np.ndarray:
    # 固定种子生成随机分数，提供无需训练的排序参照。
    return np.random.default_rng(seed).random(n)

def amount_baseline(frame: pd.DataFrame, amount_col="TransactionAmt") -> np.ndarray:
    # 金额基线直接把交易金额当排序分数，缺失金额按零处理。
    return frame[amount_col].fillna(0).to_numpy(float)
