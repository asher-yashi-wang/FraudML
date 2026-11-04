# Policy-driven, human-in-the-loop review workflow

This extension separates three things which should not be conflated:

| Stage | Decision | Meaning |
|---|---|---|
| Policy/model triage | `APPROVE` | Below current review threshold; normal monitoring continues. |
| Policy/model triage | `REVIEW` | Uncertain risk; enters investigator queue. |
| Policy/model triage | `CHALLENGE` | Strong rule or high score; require step-up verification/hold only as approved by operations. |
| Investigator feedback | `confirmed_fraud` | Human/process-confirmed outcome; eligible as a future training label after governance checks. |

## Operating procedure

1. A policy owner changes thresholds, high-value markers, or explicit blocked entity values in `configs/risk_policy.json`.
2. The owner increments `version`, documents the reason, receives the required approval, and tests the policy on a held-out/sandbox batch.
3. Triage writes decisions, a risk-sorted `REVIEW` queue, and a policy-versioned audit event.
4. An investigator records an outcome with the `review` command. Never infer confirmed fraud simply from `CHALLENGE`.
5. Periodically monitor review volume, confirmed-fraud rate, false positives, latency, and disparate impact where lawful/appropriate. Roll back harmful policy versions.

## Example commands

```powershell
fraudml policy-init --out configs\risk_policy.json --reviewer risk-ops-owner
fraudml triage --data artifacts\scored.csv --policy configs\risk_policy.json --out artifacts\decisions.csv --review-queue artifacts\review_queue.csv
fraudml review --case-id case-abc123 --outcome confirmed_fraud --reviewer reviewer-01 --note "Investigation reference stored in case system"
```

For production, replace local JSON/CSV/JSONL with authenticated policy storage, role-based access, encrypted audit retention, a case-management UI, dual approval, monitoring, and an approved action service. Do not expose raw personal or payment data in prompts, audit notes, or logs.
