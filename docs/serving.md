# Local model scoring and Java integration

The HTTP proof of concept loads one trusted FraudML artifact at startup. A Java payment backend can submit feature values, read the returned risk score, and use its own policy and review workflow. The service marks scores as uncalibrated because the current class-weighted models have no probability calibration step.

## Start the scorer

Install the project's core dependencies. The service uses Python's standard HTTP server and requires no additional server package.

```powershell
python -m fraudml.cli make-demo --out data/demo.csv --rows 1500
python -m fraudml.cli train --data data/demo.csv --model logistic --out artifacts/demo_model.joblib --k 50
python -m fraudml.service --model artifacts/demo_model.joblib --host 127.0.0.1 --port 8080
```

The default address accepts local connections. This proof of concept runs one worker with a five-second read timeout, a 1 MiB request-body limit, and at most 256 rows per request. Select limits with `--max-body-bytes` and `--max-rows`. Request feature values are excluded from server logs.

## Inspect and submit features

`GET /health` returns readiness, the artifact's SHA-256 model version, required model columns, numeric columns, categorical-code columns, and whether historical features are required. Inspect this schema before building a request for a different model.

For the synthetic model above, save the following UTF-8 JSON as `request.json`.

```json
{
  "transactions": [
    {
      "TransactionAmt": 125.50,
      "ProductCD": "W",
      "DeviceInfo": "7",
      "card1": "42"
    }
  ]
}
```

`POST /score` requires `Content-Type: application/json`. Numeric features accept finite JSON numbers or `null`. Text categories accept strings or `null`. Columns explicitly configured as categorical codes at training accept numeric codes and strings. The existing fitted preprocessing handles missing values. Each row must include all required model columns, using `null` where the value is unavailable.

The response contains `model_version`, `score_type: "uncalibrated_model_score"`, and ordered results with `row_index` and `risk_score`. It returns scores only. Connect them to the existing policy workflow and record any investigator conclusions there.

Models trained with historical entity statistics require an upstream feature source that supplies the fitted historical columns using strictly earlier observations. Submit those prepared columns with `"features_prepared": true`. The service checks column presence and numeric types. The source remains responsible for the timestamp and entity-history semantics. A baseline artifact without historical features supports direct transaction scoring.

Validation failures return JSON errors with HTTP 422, malformed JSON returns 400, unsupported content types return 415, and oversized bodies return 413. Unexpected scoring errors return 500 without a traceback or request content.

## Call from Java

Use JDK 11 or newer. Keep the Python scorer running in one terminal. Compile and run the client in another terminal from the project root.

```powershell
javac examples/java/FraudClient.java
java -cp examples/java FraudClient http://127.0.0.1:8080 request.json
```

The client sets a two-second connection timeout and five-second overall timeout. It prints a successful response for this demonstration and exits with a nonzero code on HTTP failure. In an existing backend, parse the response with that backend's JSON library, retain the model version for audit, and route timeout or error outcomes through its existing fallback policy. Timeout, error, or missing model responses provide no score.

The built-in HTTP server provides a local integration demonstration. A deployed backend needs its normal authenticated network boundary, transport protection, worker management, and operational monitoring.
