"""Data loading and chronological splitting. Never use target labels in transforms."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
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
        tx = tx.merge(identity, on="TransactionID", how="left", validate="one_to_one")
    return tx

def chronological_split(df: pd.DataFrame, time_col: str = "TransactionDT",
                        train_fraction: float = .70, valid_fraction: float = .15) -> DataSplit:
    """Contiguous time split. Equal timestamps remain together at the later boundary."""
    if time_col not in df: raise KeyError(f"Missing time column: {time_col}")
    if not 0 < train_fraction < 1 or not 0 < valid_fraction < 1 or train_fraction + valid_fraction >= 1:
        raise ValueError("fractions must be positive and sum to less than 1")
    ordered = df.sort_values(time_col, kind="stable").reset_index(drop=True)
    n = len(ordered); a, b = int(n * train_fraction), int(n * (train_fraction + valid_fraction))
    # Move boundaries right to avoid the same timestamp in two splits.
    while a < n and a > 0 and ordered.loc[a, time_col] == ordered.loc[a-1, time_col]: a += 1
    while b < n and b > 0 and ordered.loc[b, time_col] == ordered.loc[b-1, time_col]: b += 1
    return DataSplit(ordered.iloc[:a].copy(), ordered.iloc[a:b].copy(), ordered.iloc[b:].copy())

def feature_columns(df: pd.DataFrame, target: str = "isFraud",
                    drop: tuple[str, ...] = ("TransactionID", "TransactionDT")) -> list[str]:
    """Feature columns excluding the target, identifiers, and the split time column.

    TransactionDT is dropped by default: it is the column used for chronological
    splitting, and its absolute values differ systematically between periods,
    so using it as a raw feature leaks temporal structure and hurts
    extrapolation to the holdout.
    """
    return [c for c in df if c not in {target, *drop}]
