# DPO 模型选择报告

生成日期：2026-04-29

## 最终建议

```text
正式 baseline: SFT
最佳 DPO candidate: DPO v2 checkpoint-50
DPO v3 guarded s100: 不晋升
DPO v4 targeted s50/s75/s100: 不晋升，作为负结果消融
```

原因：

```text
SFT 是最稳的正式 baseline，EM 和 faithfulness 最高，fabricated-number rate 最低。
DPO v2 checkpoint-50 是最佳 DPO candidate，在 EM 接近 SFT 的前提下，citation precision 和 wrong-citation rate 明显更好。
DPO v3 没有超过 v2 s50。
DPO v4 targeted sweep 没有修复预期问题，反而显著损伤 citation grounding。
```

## 主指标

主评测集为 400 条 held-out 样本。错误类指标越低越好，其余指标越高越好。

| model | exact_match | numeric_exact_match | faithfulness_rate | citation_precision | citation_consistency_score | wrong_citation_rate | fabricated_number_rate | calculation_error_rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 0.1975 | 0.0504 | 0.1675 | 0.2487 | 0.2202 | 0.7600 | 0.6000 | 0.6300 |
| SFT | 0.3300 | 0.1727 | 0.4200 | 0.8950 | 0.6331 | 0.0825 | 0.5350 | 0.5750 |
| DPO v1 s500 | 0.2975 | 0.1475 | 0.4000 | 0.9450 | 0.6225 | 0.0125 | 0.5625 | 0.5925 |
| DPO v2 s100 | 0.3175 | 0.1691 | 0.4075 | 0.9525 | 0.6356 | 0.0350 | 0.5600 | 0.5775 |
| DPO v2 s50 | 0.3275 | 0.1763 | 0.4175 | 0.9575 | 0.6425 | 0.0300 | 0.5525 | 0.5725 |
| DPO v3 guarded s100 | 0.3275 | 0.1727 | 0.4125 | 0.9450 | 0.6369 | 0.0300 | 0.5500 | 0.5750 |
| DPO v4 targeted s50 | 0.3225 | 0.1655 | 0.3425 | 0.6750 | 0.5681 | 0.3000 | 0.5550 | 0.5800 |
| DPO v4 targeted s75 | 0.3175 | 0.1583 | 0.3375 | 0.6400 | 0.5600 | 0.3275 | 0.5550 | 0.5850 |
| DPO v4 targeted s100 | 0.3200 | 0.1655 | 0.3400 | 0.6275 | 0.5581 | 0.3400 | 0.5475 | 0.5800 |

## DPO Candidate 排序

```text
1. DPO v2 checkpoint-50
2. DPO v3 guarded s100
3. DPO v2 checkpoint-100
4. DPO v1 s500
5. DPO v4 targeted checkpoint-50
6. DPO v4 targeted checkpoint-100
7. DPO v4 targeted checkpoint-75
```

排序依据不是单一 EM，而是综合：

```text
exact_match
numeric_exact_match
faithfulness_rate
citation_precision
citation_consistency_score
wrong_citation_rate
fabricated_number_rate
calculation_error_rate
over_refusal_rate
forced_answer_rate
```

## 与 SFT 的关键差异

| model | EM delta | numeric EM delta | citation precision delta | wrong citation delta | fabricated number delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| DPO v1 s500 | -0.0325 | -0.0252 | +0.0500 | -0.0700 | +0.0275 |
| DPO v2 s50 | -0.0025 | +0.0036 | +0.0625 | -0.0525 | +0.0175 |
| DPO v2 s100 | -0.0125 | -0.0036 | +0.0575 | -0.0475 | +0.0250 |
| DPO v3 guarded s100 | -0.0025 | +0.0000 | +0.0500 | -0.0525 | +0.0150 |
| DPO v4 targeted s50 | -0.0075 | -0.0072 | -0.2200 | +0.2175 | +0.0200 |
| DPO v4 targeted s75 | -0.0125 | -0.0144 | -0.2550 | +0.2450 | +0.0200 |
| DPO v4 targeted s100 | -0.0100 | -0.0072 | -0.2675 | +0.2575 | +0.0125 |

## 为什么不选 v4

v4 的目标是 targeted anti-regression：用 train split 构造 numeric scale、protect-correct、unanswerable refusal 和 citation repair 四类 preference pairs，并做 50/75/100 step sweep。

但 held-out eval 显示：

```text
v4 最好 checkpoint 是 s50。
v4 s50 EM 为 0.3225，低于 SFT 和 DPO v2 s50。
v4 s50 citation precision 为 0.6750，显著低于 DPO v2 s50 的 0.9575。
v4 s50 wrong-citation rate 为 0.3000，显著高于 DPO v2 s50 的 0.0300。
s75/s100 没有改善，说明增加 step 不解决该问题。
```

因此 v4 不应作为最终模型，只应作为负结果消融。它的价值在于说明：DPO 的偏好数据需要严格匹配主任务分布，pair audit 通过不代表 held-out eval 一定提升。

## 最终表述

推荐在项目展示中这样表述：

```text
SFT 是正式 baseline；DPO v2 checkpoint-50 是最佳 DPO candidate。DPO v2 在保持 EM 接近 SFT 的同时显著提升 citation precision 并降低 wrong-citation rate。但 DPO 不是免费提升，v3/v4 显示偏好优化可能带来数值尺度、answerability 和引用回归，因此最终模型选择基于 held-out eval + badcase audit，而不是训练版本号或步数。
```
