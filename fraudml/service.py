"""Local HTTP scoring service for trusted FraudML model artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import pandas as pd

from .features import as_categorical
from .models import TabularFraudModel


class RequestError(ValueError):
    """Input error that can be returned without exposing model internals."""


def _finite_number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


class ScoringApplication:
    """Load a model once and validate incoming features against its fitted schema."""

    def __init__(self, model_path, max_rows=256):
        if max_rows < 1:
            raise ValueError("max_rows must be positive")
        path = Path(model_path)
        # 用模型文件的 SHA-256 标识此次推理所用的产物，便于排查和复现。
        digest = hashlib.sha256()
        with path.open("rb") as artifact:
            for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
                digest.update(chunk)
        self.model_version = f"sha256:{digest.hexdigest()}"
        self.model = TabularFraudModel.load(path)
        self.max_rows = max_rows
        # 从已拟合的模型读取字段名和类型，避免在线服务另写一份字段清单。
        self.columns = list(self.model.pipeline.feature_names_in_)
        preprocessor = self.model.pipeline.named_steps["preprocess"]
        self.numeric_columns = []
        for name, _, columns in preprocessor.transformers_:
            if name == "num":
                self.numeric_columns.extend(columns)
        self.categorical_codes = tuple(getattr(self.model, "categorical_codes", ()))
        self.history_required = bool(getattr(self.model, "entity_cols", ()))

    def health(self):
        return {
            "status": "ready",
            "model_version": self.model_version,
            "score_type": "uncalibrated_model_score",
            "required_columns": self.columns,
            "numeric_columns": self.numeric_columns,
            "categorical_code_columns": list(self.categorical_codes),
            "requires_prepared_history": self.history_required,
            "max_rows": self.max_rows,
        }

    def score(self, payload):
        if not isinstance(payload, dict):
            raise RequestError("request must be a JSON object")
        if set(payload) - {"transactions", "features_prepared"}:
            raise RequestError("unsupported request fields")
        transactions = payload.get("transactions")
        if not isinstance(transactions, list) or not 1 <= len(transactions) <= self.max_rows:
            raise RequestError(f"transactions must contain 1..{self.max_rows} objects")
        if "features_prepared" in payload and not isinstance(payload["features_prepared"], bool):
            raise RequestError("features_prepared must be a boolean")
        # 历史字段由上游按交易时点计算。这里能检查字段和类型，无法验证统计窗口是否严格早于该交易。
        if self.history_required and payload.get("features_prepared") is not True:
            raise RequestError("history model requires prepared causal features and features_prepared=true")

        numeric = set(self.numeric_columns)
        categorical_codes = set(self.categorical_codes)
        for row in transactions:
            if not isinstance(row, dict):
                raise RequestError("each transaction must be an object")
            missing = set(self.columns) - set(row)
            if missing:
                raise RequestError("missing required model features; inspect GET /health")
            for column in self.columns:
                value = row[column]
                if value is None:
                    continue
                if isinstance(value, bool):
                    raise RequestError("boolean feature values are unsupported")
                if column in numeric:
                    if not _finite_number(value):
                        raise RequestError("numeric model features must be finite numbers or null")
                elif column in categorical_codes:
                    if not isinstance(value, (str, int, float)):
                        raise RequestError("categorical codes must be strings, numbers, or null")
                    if isinstance(value, (int, float)) and not _finite_number(value):
                        raise RequestError("categorical codes must be finite")
                elif not isinstance(value, str):
                    raise RequestError("categorical model features must be strings or null")

        # 按训练列对齐输入，忽略额外字段，保留请求中的交易顺序。
        frame = pd.DataFrame(transactions).reindex(columns=self.columns)
        # Null numerics must stay numeric even if every row is missing.
        for column in self.numeric_columns:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        if self.categorical_codes:
            frame = as_categorical(frame, list(self.categorical_codes))
        frame = frame.where(pd.notna(frame), np.nan)
        # 复用已拟合的转换和分类器。调用方仍需按训练口径准备特征值。
        scores = np.asarray(self.model.predict_proba(frame), dtype=float)
        if scores.shape != (len(frame),) or not np.all(np.isfinite(scores)):
            raise RuntimeError("model returned invalid scores")
        if np.any((scores < 0) | (scores > 1)):
            raise RuntimeError("model scores outside supported range")
        # 分数用于排序。策略动作由调用方决定，调查结论由审核员记录。
        return {
            "model_version": self.model_version,
            "score_type": "uncalibrated_model_score",
            "results": [
                {"row_index": index, "risk_score": float(score)}
                for index, score in enumerate(scores)
            ],
        }


def make_server(application, host="127.0.0.1", port=8080, max_body_bytes=1024 * 1024):
    """Create a local, single-worker server. The caller manages its lifecycle."""
    if max_body_bytes < 1:
        raise ValueError("max_body_bytes must be positive")

    class Server(HTTPServer):
        def handle_error(self, request, client_address):
            # A client disconnect can interrupt the response write.
            # Keep traceback and connection details out of request logs.
            pass

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(5)

        def log_message(self, *_):
            # Avoid recording request paths, feature values, and payment data.
            pass

        def send_json(self, status, payload):
            body = json.dumps(payload, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self.send_json(200, application.health())
            else:
                self.send_json(404, {"error": "unknown endpoint"})

        def do_POST(self):
            if self.path != "/score":
                self.send_json(404, {"error": "unknown endpoint"})
                return
            if self.headers.get_content_type() != "application/json":
                self.send_json(415, {"error": "Content-Type must be application/json"})
                return
            if self.headers.get("Transfer-Encoding"):
                self.send_json(400, {"error": "transfer encoding is unsupported"})
                return
            length_header = self.headers.get("Content-Length")
            if length_header is None:
                self.send_json(411, {"error": "Content-Length is required"})
                return
            try:
                length = int(length_header)
            except ValueError:
                self.send_json(400, {"error": "invalid Content-Length"})
                return
            if length < 1 or length > max_body_bytes:
                self.send_json(413, {"error": "body length outside configured limits"})
                return
            try:
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise RequestError("incomplete request body")
                payload = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant)
                result = application.score(payload)
            except (UnicodeDecodeError, json.JSONDecodeError):
                self.send_json(400, {"error": "invalid JSON"})
            except RequestError as error:
                self.send_json(422, {"error": str(error)})
            except TimeoutError:
                self.send_json(408, {"error": "request timed out"})
            except Exception:
                self.send_json(500, {"error": "scoring failed"})
            else:
                self.send_json(200, result)

    return Server((host, port), Handler)


def _reject_constant(_):
    raise RequestError("JSON must not contain NaN or Infinity")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--max-rows", type=int, default=256)
    parser.add_argument("--max-body-bytes", type=int, default=1024 * 1024)
    args = parser.parse_args()
    try:
        application = ScoringApplication(args.model, max_rows=args.max_rows)
        server = make_server(application, args.host, args.port, args.max_body_bytes)
    except Exception:
        parser.error("could not initialize scorer; check trusted model path, schema, and listen address")
    print(f"FraudML serving at http://{args.host}:{server.server_port} ({application.model_version})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
