# FinGround-QA 最终实验报告

## 摘要

FinGround-QA 是一个金融报告问答场景下的 grounded generation 后训练项目。项目使用 Qwen2.5-7B-Instruct 作为基座，围绕 TAT-QA / FinQA 构建统一证据 schema，完成 SFT、DPO 多版本训练、checkpoint sweep、细粒度评测和 badcase audit。

最重要的结论不是“DPO 一定优于 SFT”，而是：

```text
SFT 是最稳的正式 baseline。
DPO v2 checkpoint-50 是最佳 DPO candidate。
DPO 可以显著改善 citation 行为，但偏好数据设计不稳时会造成数值和引用回归。
DPO v4 targeted sweep 没有打过 v2，应作为负结果消融保留。
```

## 实验设置

```text
Base model: Qwen2.5-7B-Instruct
SFT method: QLoRA SFT
DPO method: QLoRA DPO
Main eval size: 400 held-out samples
SFT train size: 5000
SFT val size: 500
Unified rows: 24283
Train/eval exact question overlap: 0
```

输出格式采用 grounded JSON answer protocol，要求模型在给定 evidence contexts 的条件下回答，并返回引用信息。评测同时覆盖答案正确性、数值正确性、引用质量和 answerability 行为。

## 指标说明

错误类指标越低越好：

```text
wrong_citation_rate
fabricated_number_rate
calculation_error_rate
over_refusal_rate
forced_answer_rate
format_error_rate
schema_error_rate
```

其余核心指标越高越好：

```text
exact_match
numeric_exact_match
faithfulness_rate
citation_precision
citation_consistency_score
schema_pass_rate
```

注意：`faithfulness_rate` 和 `citation_consistency_score` 是规则与弱语义 proxy，不等于完整事实性判断。

## 主结果

| model | exact_match | numeric_exact_match | faithfulness_rate | citation_precision | citation_consistency_score | wrong_citation_rate | fabricated_number_rate | calculation_error_rate | over_refusal_rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 0.1975 | 0.0504 | 0.1675 | 0.2487 | 0.2202 | 0.7600 | 0.6000 | 0.6300 | 0.0700 |
| SFT | 0.3300 | 0.1727 | 0.4200 | 0.8950 | 0.6331 | 0.0825 | 0.5350 | 0.5750 | 0.0100 |
| DPO v1 s500 | 0.2975 | 0.1475 | 0.4000 | 0.9450 | 0.6225 | 0.0125 | 0.5625 | 0.5925 | 0.0300 |
| DPO v2 s50 | 0.3275 | 0.1763 | 0.4175 | 0.9575 | 0.6425 | 0.0300 | 0.5525 | 0.5725 | 0.0050 |
| DPO v2 s100 | 0.3175 | 0.1691 | 0.4075 | 0.9525 | 0.6356 | 0.0350 | 0.5600 | 0.5775 | 0.0050 |
| DPO v3 guarded s100 | 0.3275 | 0.1727 | 0.4125 | 0.9450 | 0.6369 | 0.0300 | 0.5500 | 0.5750 | 0.0125 |
| DPO v4 targeted s50 | 0.3225 | 0.1655 | 0.3425 | 0.6750 | 0.5681 | 0.3000 | 0.5550 | 0.5800 | 0.0125 |
| DPO v4 targeted s75 | 0.3175 | 0.1583 | 0.3375 | 0.6400 | 0.5600 | 0.3275 | 0.5550 | 0.5850 | 0.0200 |
| DPO v4 targeted s100 | 0.3200 | 0.1655 | 0.3400 | 0.6275 | 0.5581 | 0.3400 | 0.5475 | 0.5800 | 0.0200 |

## 模型选择

最终建议：

```text
正式展示 baseline: SFT
最佳 DPO candidate: DPO v2 checkpoint-50
不推荐晋升: DPO v3 guarded s100
不推荐晋升: DPO v4 targeted s50/s75/s100
```

选择 DPO v2 checkpoint-50 作为最佳 DPO candidate 的原因：

```text
1. EM 为 0.3275，基本接近 SFT 的 0.3300。
2. Numeric EM 为 0.1763，略高于 SFT 的 0.1727。
3. Citation precision 为 0.9575，高于 SFT 的 0.8950。
4. Wrong-citation rate 为 0.0300，明显低于 SFT 的 0.0825。
5. 比 DPO v2 s100 更稳，说明更长 DPO 不一定更好。
```

SFT 仍然是最稳正式 baseline 的原因：

```text
1. EM 最高。
2. Faithfulness rate 最高。
3. Fabricated-number rate 最低。
4. 自动指标与人工 badcase audit 均显示 DPO 存在副作用。
```

## v4 负结果分析

v4 的目标是基于 train split 构建 targeted guardrail preference data，重点覆盖：

```text
numeric scale guard
protect-correct guard
unanswerable refusal guard
citation repair guard
```

v4 数据规模：

```text
total pairs: 680
numeric_scale_guard: 240
protect_correct_guard: 160
unanswerable_refusal_guard: 120
citation_repair_guard: 160
manual/audit sample: 120
gate_pass: true
```

但 v4 评测结果显示，targeted DPO 没有达到预期：

```text
v4 s50 EM: 0.3225，低于 SFT 和 DPO v2 s50
v4 s50 citation precision: 0.6750，显著低于 DPO v2 s50 的 0.9575
v4 s50 wrong-citation rate: 0.3000，显著高于 DPO v2 s50 的 0.0300
v4 s75/s100 继续变差，说明增加 step 没有修复问题
```

合理解释：

```text
1. targeted pair 的局部分布与主 eval 分布不完全一致。
2. citation-repair pair 可能强化了“输出引用形式”，但没有稳定提升证据一致性。
3. DPO 对小规模偏好数据较敏感，680 pairs 训练到 100 step 已经约 2.35 epoch，继续训练可能放大偏差。
4. 自动 pair audit 通过不代表主评测一定提升，仍需要 held-out eval 和 badcase audit 闭环。
```

因此 v4 应写成负结果消融，而不是包装成最终提升。

## 面试叙事

推荐表达：

```text
这个项目不是简单跑通 SFT/DPO，而是围绕 grounded financial QA 构建了一套后训练实验闭环。我先用 SFT 把 Base EM 从 19.75% 提升到 33.00%，再用 DPO v2 在保持 EM 接近的情况下把 citation precision 提升到 95.75%，wrong-citation rate 降到 3.00%。同时，我没有把 DPO 当成万能提升：v3/v4 的 badcase 和指标显示，偏好优化会带来 numeric scale、answerability 和 citation regression，因此最终选择 DPO v2 s50 作为最佳 DPO candidate，并把 v4 作为负结果消融分析。
```

面试中应该强调：

```text
数据质量检查
偏好对构造和 audit
checkpoint selection
指标拆解而不是只看 EM
负结果分析能力
对 DPO tradeoff 的理解
```

不应该声称：

```text
这是完整金融大模型
这是生产级 RAG 系统
DPO 全面超过 SFT
v4 成功修复了问题
proxy faithfulness 等同于人工事实性判断
```

## 关键文件

```text
results/base_metrics.json
results/sft_metrics.json
results/dpo_metrics.json
results/dpo_v2_reweighted_s50_metrics.json
results/dpo_v2_reweighted_s100_metrics.json
results/dpo_v3_guarded_s100_metrics.json
results/dpo_v4_targeted_s50_retry_metrics.json
results/dpo_v4_targeted_s75_retry_metrics.json
results/dpo_v4_targeted_s100_retry_metrics.json
reports/dpo_model_selection_report.md
reports/dpo_v3_posthoc_audit_report.md
reports/final_experiment_report_zh.md
docs/resume_project_brief.md
docs/interview_script.md
```
