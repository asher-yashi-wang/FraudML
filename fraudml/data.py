"""Data loading and chronological splitting. Never use target labels in transforms."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class DataSplit:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame

def load_ieee_cis(transaction_csv: str | Path, identity_csv: str | Path | None = None) -> pd.DataFrame:
    """Load Kaggle IEEE-CIS train files; merge identity only by TransactionID."""
    tx = pd.read_csv(transaction_csv)
    if identity_csv:
        identity = pd.read_csv(identity_csv)
        # 左连接保留缺少身份信息的交易，一对一校验防止重复键把交易行扩增。
        tx = tx.merge(identity, on="TransactionID", how="left", validate="one_to_one")
    return tx

def validate_timestamps(df: pd.DataFrame, time_col: str = "TransactionDT") -> None:
    if time_col not in df:
        raise KeyError(f"Missing time column: {time_col}")
    timestamps = df[time_col]
    # 缺失或无穷时间无法判断先后，不能用于时间切分和历史聚合。
    if not pd.api.types.is_numeric_dtype(timestamps) or not np.isfinite(timestamps.to_numpy(dtype=float, na_value=np.nan)).all():
        raise ValueError(f"{time_col} must contain finite numeric timestamps")


def chronological_split(df: pd.DataFrame, time_col: str = "TransactionDT",
                        train_fraction: float = .70, valid_fraction: float = .15) -> DataSplit:
    """Contiguous time split without dividing any equal-timestamp group."""
    validate_timestamps(df, time_col)
    if not 0 < train_fraction < 1 or not 0 < valid_fraction < 1 or train_fraction + valid_fraction >= 1:
        raise ValueError("fractions must be positive and sum to less than 1")
    # 用较早的交易训练，用较晚的交易模拟未来表现。稳定排序保留同一时间戳内的原行顺序。
    ordered = df.sort_values(time_col, kind="stable").reset_index(drop=True)
    n = len(ordered); a, b = int(n * train_fraction), int(n * (train_fraction + valid_fraction))
    # 若切点落在同一时间戳中间，右移到整组末尾，所以实际比例可略有偏差。
    while a < n and a > 0 and ordered.loc[a, time_col] == ordered.loc[a-1, time_col]: a += 1
    while b < n and b > 0 and ordered.loc[b, time_col] == ordered.loc[b-1, time_col]: b += 1
    if not 0 < a < b < n:
        raise ValueError("Time split produced an empty partition; adjust the fractions or use more timestamps")
    return DataSplit(ordered.iloc[:a].copy(), ordered.iloc[a:b].copy(), ordered.iloc[b:].copy())

def feature_columns(df: pd.DataFrame, target: str = "isFraud",
                    drop: tuple[str, ...] = ("TransactionID", "TransactionDT")) -> list[str]:
    """Feature columns excluding the target, identifiers, and the split time column.

    TransactionDT is dropped by default: it is the column used for chronological
    splitting, and its absolute values differ systematically between periods,
    so using it as a raw feature can hurt extrapolation to the holdout.
    """
    # 标签只用于监督和评估。原始时间戳仍用于切分与历史统计，默认不直接送入模型。
    return [c for c in df if c not in {target, *drop}]


def validate_feature_config(df, entity_cols, categorical_codes):
    """Reject unavailable inputs and identifiers or outcomes used as features."""
    # 禁止用当笔标签或审核结果作为分组键，否则派生的历史特征也可能暴露结果。
    configured = set(entity_cols) | set(categorical_codes)
    reserved = {"TransactionID", "TransactionDT", "isFraud", "risk_score",
                "decision", "decision_reason", "policy_version", "review_case_id", "outcome"}
    if configured & reserved:
        raise ValueError(f"Reserved feature configuration columns: {sorted(configured & reserved)}")
    if "TransactionAmt" in categorical_codes:
        raise ValueError("TransactionAmt must stay numeric")
    missing = configured - set(df.columns)
    if missing:
        raise ValueError(f"Missing feature configuration columns: {sorted(missing)}")
