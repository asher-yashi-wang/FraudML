# 李一辉任务书:数据层漂移审计 + 滚动时间窗验证

**目标:** 把"数据审计与时序验证"职责下两块真实未完成的工作做成正式交付物,用于个人报告、海报与录播口播。全部脚本已写好并冒烟测试通过,你只需运行、核对、理解并撰写。

## 任务 A:数据层时间漂移审计

**为什么做:** 你们报告里 LR 崩塌的诊断缺"数据层证据"。本任务产出三样东西:三期欺诈率、按周欺诈率曲线、特征漂移表(train → test 的 max|z| 变化),证明漂移在数据层面已经存在,不只是模型现象。

**运行:**
```bash
cd /Users/wangyashi/Desktop/FraudML/FraudML-IEEE-CIS
python scripts/drift_audit.py \
    --transaction data/ieee-cis/train_transaction.csv \
    --out-dir artifacts
```

**产出(artifacts/ 下):**
- `drift_audit.json` — 三期行数/欺诈数/欺诈率、按周欺诈率、漂移前 10 特征;
- `drift_audit_features.csv` — 漂移特征表(可直接贴报告);
- `drift_audit_weekly.png` — 周欺诈率曲线(含切分边界竖线)。

**验收标准:**
1. 三期欺诈率约为 3.5% / 3.43% / 3.48%,与 `final_metrics.json` 一致;
2. 漂移前 10 特征里应出现 V322、V101、V324、V95、V177、V323、V293、V103 中的大多数(test 期 max|z| 达数百至上千);
3. 你能解释:max|z| 是什么、为什么它度量漂移、为什么它只描述数据不涉及标签。

**写进个人报告的话术(示例):**
"对 394 列逐一审计后,我计算了三期欺诈率(train 3.5%, validation 3.43%, test 3.48%)与各数值特征在三期的 max|z|。结果显示 V322 等特征在测试期的极端值达到训练期的数十倍(z≈1829 vs 66),说明漂移在数据层已可观测。这为'LR 排序崩塌'提供了上游证据。"

## 任务 B:滚动时间窗鲁棒性验证

**为什么做:** 报告最大的限制是"单一 holdout"。本任务在现有 70/15/15 之外再切两个时间窗(60/15/25、80/10/10),验证"LightGBM >> 金额排序"的结论是否跨窗口成立。

**运行:**
```bash
cd /Users/wangyashi/Desktop/FraudML/FraudML-IEEE-CIS
python scripts/rolling_windows.py \
    --transaction data/ieee-cis/train_transaction.csv \
    --categorical-codes card1 addr1 \
    --k 500 --seed 42 --out artifacts/rolling_windows.json
```
(LightGBM 会打印训练日志,属正常现象。)

**产出:** `artifacts/rolling_windows.json` + 终端上的对比表(三个窗口的 n、欺诈率、金额/随机/LightGBM 命中@500、P@500、AP)。

**验收标准(重要):**
1. **第一行 0.7/0.15 必须精确复现 465 / 10 / 13(LightGBM/金额/随机命中@500)**——这是脚本的正确性自检,对不上就是 bug,先停下报告,不要继续;
2. 三个窗口的结论方向一致:LightGBM 命中数都远大于金额排序;
3. 你能解释:为什么滚动窗口是"额外外层 holdout",以及为什么**不许**因为某个窗口结果好看就改模型配置。

**写进个人报告的话术(示例):**
"我在三个时间窗(70/15/15、60/15/25、80/10/10)上复跑基础配置。首个窗口精确复现冻结协议的结果(LightGBM 465 笔 vs 金额 10 笔),另外两个窗口结论方向一致(LightGBM 命中数均远超金额排序),说明主结论不依赖单一切点。"

## 红线(务必遵守)

1. 两个脚本的输出都标注为 **post-hoc**,不得回头修改任何模型配置、特征或 seed;
2. 冻结的 test 协议不可改动;滚动窗口只是额外证据;
3. 结果好坏都如实报告,不许挑窗口。
4. 运行后请把 JSON 与图留在 `artifacts/`(已被 gitignore,属本地证据),把 `scripts/drift_audit.py`、`scripts/rolling_windows.py` 提交到仓库。

## 时间估算

- 运行:A 约 1–2 分钟,B 约 5–8 分钟;
- 理解 + 撰写个人报告段落:各半天。
