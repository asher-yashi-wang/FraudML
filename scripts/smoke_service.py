"""Exercise a synthetic model through HTTP and a compiled JDK 11 client."""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fraudml.data import feature_columns
from fraudml.models import TabularFraudModel
from fraudml.service import ScoringApplication, make_server
from fraudml.synthetic import make_synthetic_transactions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--java-home", type=Path, help="JDK directory containing bin/java and bin/javac")
    args = parser.parse_args()
    java = str(args.java_home / "bin/java") if args.java_home else shutil.which("java")
    javac = str(args.java_home / "bin/javac") if args.java_home else shutil.which("javac")
    if not java or not javac:
        parser.error("JDK 11+ is required")
    source = Path(__file__).resolve().parents[1] / "examples/java/FraudClient.java"
    with tempfile.TemporaryDirectory(prefix="fraudml-service-") as directory:
        scratch = Path(directory)
        frame = make_synthetic_transactions(400)
        columns = feature_columns(frame)
        model = TabularFraudModel().fit(frame[columns], frame.isFraud)
        artifact = scratch / "model.joblib"
        model.save(artifact)
        transactions = frame.iloc[:3][columns].to_dict(orient="records")
        request_path = scratch / "request.json"
        request_path.write_text(json.dumps({"transactions": transactions}), encoding="utf-8")
        subprocess.run([javac, "--release", "11", "-d", directory, str(source)], check=True)
        application = ScoringApplication(artifact)
        server = make_server(application, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            response = subprocess.run([
                java, "-cp", directory, "FraudClient",
                f"http://127.0.0.1:{server.server_port}", str(request_path),
            ], check=True, capture_output=True, text=True, timeout=15)
            result = json.loads(response.stdout)
            expected = model.predict_proba(frame.iloc[:3][columns])
            np.testing.assert_allclose(
                [row["risk_score"] for row in result["results"]], expected,
                atol=1e-12, rtol=0,
            )
            assert result["model_version"] == application.model_version
            print("PASS: compiled Java client called the live local scorer; all 3 scores match Python")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    main()
