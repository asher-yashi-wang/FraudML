# FraudML team responsibilities and implementation plan

This is a proposed allocation for the five-member course project. Each member owns a technical module and the evidence needed to explain it. The allocation records planned work. Final contribution statements should describe the work each person actually completes.

| Member | Main responsibility | Concrete deliverables |
|---|---|---|
| Wang Yashi 王雅仕 | Model development and scoring integration | Own the LR and LightGBM training pipeline and validation configuration choices. Save fitted preprocessing, model artifacts and input schemas. Integrate Python scoring with the Java client and verify prediction consistency, including mixed category-code inputs and request failures. |
| Li Yihui 李熠辉 | Data audit and chronological validation | Audit transaction counts, fraud prevalence, missingness and outliers. Supply documented time partitions and verify timestamp boundaries. Document data access, field limitations and label-availability assumptions. |
| Wang Shihong 王时弘 | Feature engineering and ablation design | Implement training-only categorical encoding and strictly historical statistics. Verify same-timestamp exclusion and missing-value handling. Specify the base and historical-feature variants and check feature availability before model fitting. |
| Wu Jiaqi 吴嘉琦 | Ranking evaluation and temporal robustness | Validate AP and Top-K calculations under the shared configurations. Produce model and ablation comparisons and result figures. Investigate LR score ties, extreme inputs and raw decision scores, and evaluate temporal changes under a predefined protocol. |
| Chen Situ 陈斯图 | Review policy and experiment reproducibility | Own capacity-based queue selection, independent policy recommendations and investigator/audit records. Verify queue/evaluation agreement and archived result provenance. Maintain the reproducible demonstration and release checklist. |

## Shared report and presentation

Each member drafts the method, result or implementation section for their own module and provides its supporting files. All five review the final report and prepare to explain their own technical contribution during questions. Presenter assignments can be agreed separately from module ownership. Wang Yashi integrates the final package.

## Coordination

Li supplies audited data and the time partitions. Wang Shihong supplies fitted feature-transform specifications and ablation configurations to Wang Yashi. Wang Yashi owns model fitting, validation configuration selection, saved artifacts and the scoring interface, and supplies scores and experiment records to Wu. Wu evaluates the ranking and temporal behavior. Chen connects the agreed scores to policy recommendations, the exact-K queue and audit records, and verifies result reproducibility. Wang Yashi and Chen jointly check the Java/Python scoring contract and the scoring-to-review demonstration.

The main comparisons cover tabular ranking and the historical-feature variant. GraphSAGE remains an optional extension with no reported result. Additional experiments require an agreed question, a fixed protocol and an owner.

## Milestones

| Period | Milestone |
|---|---|
| 5–8 November | Finalize the problem statement and proposed ownership. Audit data and preserve recorded experiment results. |
| 9–15 November | Verify the data split and causal features. Complete validation comparisons, LR diagnostics and the scoring/review integration example. |
| 16–22 November | Complete evaluation figures, feature comparisons and temporal error analysis. Draft each module section and review the combined report. |
| 23–25 November | Rehearse the presentation and questions. Check code, configuration, metric provenance and the reproducible package. |
| 26–29 November | Complete submission checks and submit under the deadline recorded in the proposal. |

Internal completion is 25 November. The proposal records final submission as 29 November 2026 at 11:59 PM.

## Evaluation discipline

Use earlier data to fit preprocessing and models. Select configurations using validation data. Preserve the existing final holdout results. Label subsequent holdout diagnostics and experiments as post-hoc extensions, and predefine the windows and comparisons for any additional backtesting. Report investigator feedback separately from dataset labels and model scores.
