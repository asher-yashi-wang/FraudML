from __future__ import annotations
from pathlib import Path
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from fraudml.features import build_preprocessor

class TabularFraudModel:
    def __init__(self, kind="logistic", random_state=42, **kwargs):
        self.kind, self.random_state, self.kwargs = kind, random_state, kwargs
        self.pipeline = None
    def fit(self, X, y):
        if self.kind == "logistic": estimator = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=self.random_state, **self.kwargs)
        elif self.kind == "lightgbm":
            try:
                from lightgbm import LGBMClassifier
            except ImportError as e: raise ImportError("Install optional dependency: pip install -e '.[lightgbm]'") from e
            estimator = LGBMClassifier(n_estimators=300, learning_rate=.05, class_weight="balanced", random_state=self.random_state, **self.kwargs)
        else: raise ValueError("kind must be 'logistic' or 'lightgbm'")
        self.pipeline = Pipeline([("preprocess", build_preprocessor(X)), ("model", estimator)])
        self.pipeline.fit(X, y); return self
    def predict_proba(self, X):
        if self.pipeline is None: raise RuntimeError("Model is not fitted")
        return self.pipeline.predict_proba(X)[:, 1]
    def save(self, path): joblib.dump(self, Path(path))
    @classmethod
    def load(cls, path): return joblib.load(Path(path))
