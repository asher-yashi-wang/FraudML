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
from .data import validate_timestamps

DEFAULT_MAX_CARDINALITY = 32


def as_categorical(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Treat integer-coded entity proxies as categories, not magnitudes.

    IEEE-CIS stores fields like ``card1`` and ``addr1`` as numeric codes
    whose order is meaningless. Converting them to strings
    routes them to the categorical branch of ``build_preprocessor`` (one-hot
    for low cardinality, frequency encoding for high), which is usually much
    stronger for logistic regression than standardizing raw codes. Missing
    values are preserved.
    """
    out = frame.copy()
    for col in cols:
        if col in out:
            # Plain str + np.nan in an object column. pandas StringDtype keeps
            # pd.NA through .astype(object), and sklearn's SimpleImputer
            # raises on pd.NA comparisons, so mask with np.nan after str()
            # instead of relying on a string dtype.
            s = out[col].astype(str)
            if pd.api.types.is_float_dtype(out[col]):
                # CSV inference may switch an integer code to float when nulls appear.
                s = s.str.replace(r"^(-?\d+)\.0$", r"\1", regex=True)
            out[col] = s.where(out[col].notna(), np.nan).astype(object)
    return out


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
            counts = frame[col].astype("string").value_counts(dropna=True)
            self.maps_[col] = counts / len(frame)
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

    Rows with equal timestamps share the same history. The mean uses only
    historical rows with a known amount, while the count includes every
    historical transaction for the entity.
    """
    validate_timestamps(df, time_col)
    out = df.sort_values(time_col, kind="stable").reset_index(drop=True).copy()
    for col in entity_cols:
        if col not in out:
            continue
        grouped = out.groupby([col, time_col], dropna=False, sort=False, observed=True)
        # The sorted input makes buckets appear in chronological order within
        # each entity. Using bucket IDs avoids a full-frame merge per feature.
        buckets = grouped[amount_col].agg(["size", "sum", "count"])
        totals = buckets.groupby(level=0, dropna=False, sort=False, observed=True).cumsum()
        previous = totals.groupby(level=0, dropna=False, sort=False, observed=True).shift(fill_value=0)
        bucket_ids = grouped.ngroup().to_numpy()
        out[f"{col}__past_count"] = previous["size"].to_numpy()[bucket_ids]
        past_mean = previous["sum"] / previous["count"].replace(0, np.nan)
        out[f"{col}__past_amount_mean"] = past_mean.to_numpy()[bucket_ids]
    return out
