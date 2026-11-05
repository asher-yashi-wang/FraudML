import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

from fraudml.data import feature_columns
from fraudml.features import add_past_entity_stats, as_categorical
from fraudml.models import TabularFraudModel
from fraudml.service import RequestError, ScoringApplication, make_server
from fraudml.synthetic import make_synthetic_transactions


@pytest.fixture
def scorer(tmp_path):
    frame = make_synthetic_transactions(200)
    cols = feature_columns(frame)
    model = TabularFraudModel().fit(frame[cols], frame.isFraud)
    path = tmp_path / "model.joblib"
    model.save(path)
    return ScoringApplication(path), frame[cols]


def test_service_scores_match_saved_model(scorer):
    application, frame = scorer
    rows = frame.head(3)
    response = application.score({"transactions": rows.to_dict("records")})
    assert response["score_type"] == "uncalibrated_model_score"
    assert response["model_version"].startswith("sha256:")
    assert np.allclose(
        [row["risk_score"] for row in response["results"]],
        application.model.predict_proba(rows),
    )


def test_service_requires_typed_complete_bounded_features(scorer):
    application, frame = scorer
    row = frame.iloc[0].to_dict()
    with pytest.raises(RequestError, match="missing required"):
        application.score({"transactions": [{}]})
    with pytest.raises(RequestError, match="finite numbers"):
        application.score({"transactions": [{**row, "TransactionAmt": "100"}]})
    with pytest.raises(RequestError, match="finite numbers"):
        application.score({"transactions": [{**row, "TransactionAmt": float("nan")}]})
    with pytest.raises(RequestError, match="finite numbers"):
        application.score({"transactions": [{**row, "TransactionAmt": 10 ** 500}]})
    with pytest.raises(RequestError, match="1..256"):
        application.score({"transactions": [row] * 257})


def test_service_normalizes_categorical_codes(tmp_path):
    frame = make_synthetic_transactions(200)
    frame["card1"] = frame["card1"].astype(int)
    codes = ["card1"]
    cols = feature_columns(frame)
    features = as_categorical(frame, codes)
    model = TabularFraudModel().fit(features[cols], frame.isFraud)
    model.categorical_codes = tuple(codes)
    path = tmp_path / "model.joblib"
    model.save(path)
    application = ScoringApplication(path)
    response = application.score({"transactions": frame[cols].head(2).to_dict("records")})
    assert np.allclose(
        [row["risk_score"] for row in response["results"]],
        model.predict_proba(features[cols].head(2)),
    )


def test_service_history_artifact_requires_prepared_columns(tmp_path):
    frame = make_synthetic_transactions(200)
    features = add_past_entity_stats(frame, ["DeviceInfo"])
    cols = feature_columns(features)
    model = TabularFraudModel().fit(features[cols], features.isFraud)
    model.entity_cols = ("DeviceInfo",)
    path = tmp_path / "history.joblib"
    model.save(path)
    application = ScoringApplication(path)
    rows = frame[feature_columns(frame)].tail(1).to_dict("records")
    with pytest.raises(RequestError, match="prepared causal features"):
        application.score({"transactions": rows})
    with pytest.raises(RequestError, match="missing required"):
        application.score({"transactions": rows, "features_prepared": True})
    response = application.score({
        "transactions": features[cols].tail(1).to_dict("records"),
        "features_prepared": True,
    })
    assert response["results"][0]["risk_score"] == pytest.approx(
        model.predict_proba(features[cols].tail(1))[0])


def test_service_http_contract_and_body_limits(scorer):
    application, frame = scorer
    server = make_server(application, port=0, max_body_bytes=1024)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base + "/health", timeout=2) as response:
            assert json.load(response)["status"] == "ready"
        payload = json.dumps({"transactions": frame.head(1).to_dict("records")}).encode()
        request = Request(base + "/score", data=payload,
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=2) as response:
            assert len(json.load(response)["results"]) == 1
        invalid = Request(base + "/score", data=b"NaN",
                          headers={"Content-Type": "application/json"})
        with pytest.raises(HTTPError) as error:
            urlopen(invalid, timeout=2)
        assert error.value.code == 422
        too_large = Request(base + "/score", data=b" " * 1025,
                            headers={"Content-Type": "application/json"})
        with pytest.raises(HTTPError) as error:
            urlopen(too_large, timeout=2)
        assert error.value.code == 413
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)
