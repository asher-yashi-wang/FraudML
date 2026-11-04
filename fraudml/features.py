"""Train-only preprocessing and safe historical entity aggregates.

Categorical columns are split by cardinality: low-cardinality columns are
one-hot encoded, high-cardinality columns (e.g. IEEE-CIS ``card1``/``addr1``)
are frequency encoded so the design matrix stays tractable. Frequencies are
fitted on training data only, so no target or future information leaks into
the encoding.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DEFAULT_MAX_CARDINALITY = 32


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Target-agnostic frequency encoding fitted on train only.

    Each value is replaced by its relative frequency in the training data.
    Unseen and missing values map to 0.0. One output column per input column,
    so high-cardinality columns do not blow up the feature matrix.
    """

    def __init__(self):
        self.maps_: dict[str, pd.Series] = {}

    def fit(self, X, y=None):
        frame = pd.DataFrame(X)
        self.maps_ = {}
        for col in frame.columns:
            self.maps_[col] = frame[col].astype("string").value_counts(dropna=False, normalize=True)
        return self

    def transform(self, X):
        frame = pd.DataFrame(X).copy()
        for col in frame.columns:
            encoded = frame[col].astype("string").map(self.maps_[col])
            frame[col] = encoded.fillna(0.0).astype(np.float64)
        return frame

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = list(self.maps_)
        return np.asarray(list(input_features), dtype=object)


def build_preprocessor(frame: pd.DataFrame,
                       max_cardinality: int = DEFAULT_MAX_CARDINALITY) -> ColumnTransformer:
    """Median-impute/scale numerics, one-hot low-cardinality categories,
    frequency-encode high-cardinality categories. Fitted on train data only;
    ``sparse_threshold=1.0`` keeps the output sparse so logistic regression
    stays memory-feasible on the full IEEE-CIS dataset.
    """
    numeric = frame.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in frame.columns if c not in numeric]
    low_card = [c for c in categorical if frame[c].nunique(dropna=False) <= max_cardinality]
    high_card = [c for c in categorical if c not in low_card]

    transformers: list = []
    if numeric:
        transformers.append(("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric))
    if low_card:
        transformers.append(("cat_low", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), low_card))
    if high_card:
        transformers.append(("cat_freq", FrequencyEncoder(), high_card))
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=1.0)


def add_past_entity_stats(df: pd.DataFrame, entity_cols: list[str],
                          time_col="TransactionDT", amount_col="TransactionAmt") -> pd.DataFrame:
    """Features for each row use strictly earlier transactions only.

    Vectorized: past count is a group cumcount, past amount mean is a shifted
    group cumsum divided by that count, which scales to the full dataset
    (the previous per-group expanding mean was quadratic-ish in practice).
    """
    out = df.sort_values(time_col, kind="stable").reset_index(drop=True).copy()
    for col in entity_cols:
        if col not in out:
            continue
        group_amounts = out.groupby(col, dropna=False, sort=False)[amount_col]
        past_count = group_amounts.cumcount()
        past_sum = group_amounts.shift().fillna(0.0).groupby(out[col], dropna=False, sort=False).cumsum()
        out[f"{col}__past_count"] = past_count
        out[f"{col}__past_amount_mean"] = past_sum / past_count.replace(0, np.nan)
    return out
