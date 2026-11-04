# Final Report Template — FraudML

## Abstract

[State task, data, evaluation period, models, and actual main result. Do not insert unrun results.]

## 1. Introduction

[Why capacity-constrained fraud ranking matters; research question.]

## 2. Data and ethics

[Source/terms, fields used, anonymization, limitations, privacy handling, class balance.]

## 3. Methods

### 3.1 Leakage controls
[Exact chronological cutoffs, train-only fitting, graph snapshot protocol.]

### 3.2 Models and baselines
[Random, amount, LR, LightGBM, and optional GraphSAGE configuration.]

### 3.3 Metrics
[PR-AUC and exact definition of Precision@K and Recall@K; why accuracy is excluded.]

## 4. Results

| Model | PR-AUC | Precision@K | Recall@K | K | Split | Seed |
|---|---:|---:|---:|---:|---|---:|
| [fill after run] | | | | | | |

[Include actual outcomes, uncertainty/sensitivity, compute environment, and failures/negative results.]

## 5. Discussion

[Interpret relative to baselines; operational trade-offs; leakage threats; generalization limits.]

## 6. Conclusion

[Answer the research question narrowly; no claims beyond measured evidence.]

## Reproducibility appendix

[Commit hash, commands, package versions, configs, seeds, hardware, split cutoffs.]
