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
        # 类别加权提高少数类样本在训练损失中的权重。
        if self.kind == "logistic": estimator = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=self.random_state, **self.kwargs)
        elif self.kind == "lightgbm":
            try:
                from lightgbm import LGBMClassifier
            except ImportError as e: raise ImportError("Install optional dependency: pip install -e '.[lightgbm]'") from e
            estimator = LGBMClassifier(n_estimators=300, learning_rate=.05, class_weight="balanced", random_state=self.random_state, **self.kwargs)
        else: raise ValueError("kind must be 'logistic' or 'lightgbm'")
        # 把预处理和分类器装进同一条管线，预测时复用训练中拟合出的转换规则。
        self.pipeline = Pipeline([("preprocess", build_preprocessor(X)), ("model", estimator)])
        self.pipeline.fit(X, y); return self
    def predict_proba(self, X):
        if self.pipeline is None: raise RuntimeError("Model is not fitted")
        # 取欺诈类（标签为 1）的输出。当前项目将未校准输出用于风险排序。
        return self.pipeline.predict_proba(X)[:, 1]
    def save(self, path):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        # 保存整条管线和模型配置。加载 joblib 时应使用可信来源的文件。
        joblib.dump(self, target)
    @classmethod
    def load(cls, path): return joblib.load(Path(path))
