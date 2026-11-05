import json
import sys

import pandas as pd
import pytest

from fraudml.policy import RiskPolicy
from fraudml.workflow import review_queue, triage


def test_fixed_budget_includes_challenges_and_matches_ranking():
    frame = pd.DataFrame({
        "TransactionID": [1, 2, 3, 4],
        "risk_score": [0.50, 0.95, 0.10, 0.50],
        "TransactionAmt": [10, 20, 30, 40],
    }, index=[9, 4, 6, 2])
    result = triage(frame, RiskPolicy(), review_budget=2)
    queue = review_queue(result)
    assert queue["TransactionID"].tolist() == [2, 1]
    assert queue["decision"].tolist() == ["CHALLENGE", "REVIEW"]
    assert result.index.tolist() == frame.index.tolist()
    assert queue["review_rank"].tolist() == [1, 2]
    assert result["review_selected"].sum() == 2
    assert result.loc[2, "decision"] == "REVIEW"
    assert not result.loc[2, "review_selected"]


def test_budget_can_include_low_risk_rows_and_roundtrip_csv(tmp_path):
    frame = pd.DataFrame({"TransactionID": [1, 2], "risk_score": [0.95, 0.1]})
    result = triage(frame, RiskPolicy(), review_budget=2)
    output = tmp_path / "decisions.csv"
    result.to_csv(output, index=False)
    queue = review_queue(pd.read_csv(output))
    assert queue["decision"].tolist() == ["CHALLENGE", "APPROVE"]
    assert len(queue) == 2


def test_legacy_queue_preserves_threshold_behavior():
    legacy = pd.DataFrame({
        "decision": ["APPROVE", "REVIEW", "CHALLENGE"],
        "risk_score": [0.1, 0.5, 0.95],
    })
    assert review_queue(legacy)["decision"].tolist() == ["REVIEW"]
    current = triage(legacy.drop(columns="decision"), RiskPolicy())
    assert review_queue(current)["decision"].tolist() == ["REVIEW"]


@pytest.mark.parametrize("budget", [0, -1, 4, True, 1.5])
def test_invalid_review_budget_is_rejected(budget):
    with pytest.raises(ValueError, match="review_budget"):
        triage(pd.DataFrame({"risk_score": [0.1, 0.5, 0.95]}),
               RiskPolicy(), review_budget=budget)


@pytest.mark.parametrize("score", [None, float("inf"), -0.1, 1.1, "bad"])
def test_invalid_scores_are_rejected(score):
    with pytest.raises(ValueError, match="finite numeric"):
        triage(pd.DataFrame({"risk_score": [score]}), RiskPolicy())


def test_triage_cli_writes_exact_budget_and_audit(tmp_path, monkeypatch):
    from fraudml.cli import main

    scored = tmp_path / "scored.csv"
    pd.DataFrame({
        "TransactionID": [1, 2, 3], "risk_score": [0.6, 0.99, 0.2],
    }).to_csv(scored, index=False)
    policy = tmp_path / "policy.json"
    RiskPolicy().save(policy)
    decisions = tmp_path / "decisions.csv"
    queue = tmp_path / "queue.csv"
    audit = tmp_path / "audit.jsonl"
    monkeypatch.setattr(sys, "argv", [
        "fraudml", "triage", "--data", str(scored), "--policy", str(policy),
        "--out", str(decisions), "--review-queue", str(queue),
        "--audit", str(audit), "--review-budget", "2",
    ])
    main()
    assert pd.read_csv(queue)["TransactionID"].tolist() == [2, 1]
    event = json.loads(audit.read_text())
    assert event["queue_mode"] == "top_k"
    assert event["selected_for_review"] == event["review_budget"] == 2
