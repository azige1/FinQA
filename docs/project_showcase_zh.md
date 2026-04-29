# 项目展示稿：FinGround-QA

## 1. 项目背景

金融报告问答对 grounded generation 的要求比较高：模型不仅要回答正确，还要说明答案来自哪些证据，并且不能把百分比、小数、百万/千等数值尺度弄错。

FinGround-QA 选择了一个受控问题：

```text
当问题和 evidence contexts 已经给定时，SFT 和 DPO 会如何改变模型的答案正确性、引用质量、数值 grounding 和拒答行为？
```

所以这个项目不是完整 RAG，也不是金融大模型，而是一个大模型后训练与评测项目。

## 2. 方法总览

```text
TAT-QA / FinQA
-> 统一 question-context-evidence schema
-> 数据质量检查：quote hit、numeric grounding、leakage
-> grounded JSON answer protocol
-> Qwen2.5-7B-Instruct Base eval
-> QLoRA SFT
-> SFT badcase mining
-> Error-Type Balanced DPO
-> checkpoint sweep
-> citation / numeric / answerability eval
-> badcase audit 与模型选择
```

核心设计点：

```text
1. 不只看 EM，还拆出 citation、numeric、answerability 指标。
2. DPO preference pair 需要做质量审计，避免 length bias、format bias 和弱 rejected。
3. DPO checkpoint 需要选择，不是 step 越多越好。
4. 负结果同样进入报告，用来解释 tradeoff 和下一步方向。
```

## 3. 数据与任务

```text
数据源: TAT-QA / FinQA
主评测集: 400 held-out samples
SFT train: 5000
SFT val: 500
Unified rows: 24283
FinanceBench: external audit only
Train/eval exact question overlap: 0
Evidence quote hit rate: 90.6%
```

模型输出遵守 grounded JSON answer protocol，要求答案与引用字段同时可解析。这样可以把“答案对了但引用错了”和“引用形式正确但证据不支持”分开评估。

## 4. 实验矩阵

```text
Base: Qwen2.5-7B-Instruct
SFT: QLoRA SFT
DPO v1: balanced preference data, 500 steps
DPO v2: reweighted preference data, checkpoint-50 / checkpoint-100
DPO v3: guarded preference data, checkpoint-100
DPO v4: targeted guardrail data, checkpoint-50 / 75 / 100
```

## 5. 主结果

| model | EM | Numeric EM | Faithfulness | Citation Precision | Wrong Citation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 0.1975 | 0.0504 | 0.1675 | 0.2487 | 0.7600 |
| SFT | 0.3300 | 0.1727 | 0.4200 | 0.8950 | 0.0825 |
| DPO v2 s50 | 0.3275 | 0.1763 | 0.4175 | 0.9575 | 0.0300 |
| DPO v3 s100 | 0.3275 | 0.1727 | 0.4125 | 0.9450 | 0.0300 |
| DPO v4 s50 | 0.3225 | 0.1655 | 0.3425 | 0.6750 | 0.3000 |

![Final metrics comparison](../reports/final_metrics_comparison.png)

## 6. 最终模型选择

```text
正式 baseline: SFT
最佳 DPO candidate: DPO v2 checkpoint-50
不晋升: DPO v3 guarded s100
不晋升: DPO v4 targeted sweep
```

SFT 是最稳 baseline：

```text
EM 最高: 0.3300
faithfulness_rate 最高: 0.4200
fabricated_number_rate 最低: 0.5350
```

DPO v2 checkpoint-50 是最佳 DPO candidate：

```text
EM 接近 SFT: 0.3275 vs 0.3300
numeric EM 略高于 SFT: 0.1763 vs 0.1727
citation precision 高于 SFT: 0.9575 vs 0.8950
wrong-citation rate 低于 SFT: 0.0300 vs 0.0825
```

## 7. v4 负结果

v4 的设计目标是修复 DPO 的副作用。数据包含 680 条 targeted preference pairs：

```text
numeric_scale_guard: 240
protect_correct_guard: 160
unanswerable_refusal_guard: 120
citation_repair_guard: 160
```

但结果显示 v4 没有打过 v2：

```text
v4 s50 EM: 0.3225
v4 s50 citation precision: 0.6750
v4 s50 wrong-citation rate: 0.3000
v4 s75/s100 继续退化
```

合理解释是：targeted pair 的局部分布和主评测分布不完全匹配，citation-repair pair 可能强化了“输出引用形式”，但没有稳定强化证据一致性。DPO 对小规模偏好数据敏感，继续加 step 会放大偏差。

## 8. 面试讲法

推荐一句话：

```text
我做的是金融报告 grounded QA 后训练闭环：SFT 把 Base EM 从 19.75% 提升到 33.00%；DPO v2 checkpoint-50 在 EM 基本不降的情况下，把 citation precision 提升到 95.75%，wrong-citation rate 降到 3.00%。同时我通过 v3/v4 发现 DPO 的 numeric、answerability 和 citation regression 风险，所以最终没有盲目追更长 step，而是基于 held-out eval 和 badcase audit 选择 DPO v2 s50。
```

这个项目最值得强调的不是某个单点指标，而是：

```text
数据质量检查
偏好数据构造与审计
多版本 checkpoint selection
细粒度错误分析
能接受并解释负结果
```

## 9. 不要过度声称

```text
不要说这是金融 foundation model。
不要说这是完整 RAG 系统。
不要说 DPO 全面超过 SFT。
不要说 v4 成功。
不要说 proxy faithfulness 等价于人工事实性判断。
```
