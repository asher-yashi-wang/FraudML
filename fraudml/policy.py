"""Versioned, reviewer-maintained policy used with—not replaced by—model scores."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

@dataclass(frozen=True)
class RiskPolicy:
    """Decision thresholds and explainable hard controls.

    `CHALLENGE` means ask for step-up verification or hold for an approved
    operational action. It is not a claim that the transaction is fraud.
    """
    version: str = "2026-11-04-v1"
    reviewer: str = "unassigned"
    review_threshold: float = 0.45
    challenge_threshold: float = 0.85
    high_amount: float = 10_000.0
    blocked_values: dict[str, list[str]] = field(default_factory=dict)
    notes: str = "Review thresholds must be approved and monitored."

    def validate(self) -> "RiskPolicy":
        if not self.version.strip() or not self.reviewer.strip():
            raise ValueError("policy version and reviewer are required")
        if not 0 <= self.review_threshold <= self.challenge_threshold <= 1:
            raise ValueError("thresholds must satisfy 0 <= review <= challenge <= 1")
        if self.high_amount < 0: raise ValueError("high_amount cannot be negative")
        return self

    @classmethod
    def load(cls, path: str | Path) -> "RiskPolicy":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**raw).validate()

    def save(self, path: str | Path) -> None:
        self.validate(); target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def as_dict(self) -> dict[str, Any]: return asdict(self)
