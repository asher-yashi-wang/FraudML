"""Verify committed metric reports without downloading IEEE-CIS data."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ("validation_base.json", "validation_history.json",
           "final_metrics.json", "final_history_metrics.json")


def main():
    for filename in REPORTS:
        path = ROOT / "artifacts" / filename
        report = json.loads(path.read_text(encoding="utf-8"))
        positives = report["evaluation_fraud_count"]
        for model, result in report["results"].items():
            for key, metric in result.items():
                if not key.startswith("k="):
                    continue
                k, hits = metric["k"], metric["hits_at_k"]
                assert 0 <= hits <= min(k, positives), (filename, model, key)
                assert math.isclose(metric["precision_at_k"], hits / k)
                assert math.isclose(metric["recall_at_k"], hits / positives)
                assert 0 <= metric["pr_auc"] <= 1
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"{filename}  {report['evaluation_split']}  sha256={digest}")
    print("PASS: all four metric reports are present and internally consistent")


if __name__ == "__main__":
    main()
