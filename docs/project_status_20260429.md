# 项目状态 - 2026-04-29

## 项目定位

FinGround-QA 是一个金融报告问答场景下的 grounded generation 后训练项目。它应该被表述为“大模型后训练与评测项目”，而不是金融大模型、生产级 RAG 系统或检索系统。

核心问题：

```text
当 evidence contexts 已经给定时，SFT 和 DPO 分别如何改变模型的答案正确性、引用质量、数值 grounding 和 answerability 行为？
```

## 当前产物

已纳入项目的核心内容：

```text
数据构建与校验代码
统一 schema
SFT / DPO 数据文件
训练与评测脚本
metrics、reports、badcase audit
A100/A800 运行记录
中文项目报告和面试材料
```

不建议纳入 Git 的内容：

```text
LoRA adapter 权重
optimizer states
大日志文件
wandb 目录
原始大体量 all_unified.jsonl
同步备份目录
```

## 数据快照

```text
SFT train: 5000
SFT val: 500
Main eval: 400
FinanceBench audit: 150
Unified rows: 24283
Train/eval exact question overlap: 0
Evidence quote hit rate: 90.6%
```

## 模型实验

```text
Base: Qwen2.5-7B-Instruct
SFT: grounded JSON answer protocol 上的 QLoRA SFT
DPO v1: 原始 balanced preference training，500 steps
DPO v2: reweighted preference data，checkpoint-50 / checkpoint-100
DPO v3: guarded candidate，加入 numeric / unanswerable guards，100 steps
DPO v4: targeted guardrail data，50/75/100 step sweep
```

## 主指标

主评测集为 400 条 held-out 样本。错误类指标越低越好，其余指标越高越好。

| model | exact_match | numeric_exact_match | faithfulness_rate | citation_precision | citation_consistency_score | wrong_citation_rate | fabricated_number_rate | calculation_error_rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 0.1975 | 0.0504 | 0.1675 | 0.2487 | 0.2202 | 0.7600 | 0.6000 | 0.6300 |
| SFT | 0.3300 | 0.1727 | 0.4200 | 0.8950 | 0.6331 | 0.0825 | 0.5350 | 0.5750 |
| DPO v1 s500 | 0.2975 | 0.1475 | 0.4000 | 0.9450 | 0.6225 | 0.0125 | 0.5625 | 0.5925 |
| DPO v2 s50 | 0.3275 | 0.1763 | 0.4175 | 0.9575 | 0.6425 | 0.0300 | 0.5525 | 0.5725 |
| DPO v2 s100 | 0.3175 | 0.1691 | 0.4075 | 0.9525 | 0.6356 | 0.0350 | 0.5600 | 0.5775 |
| DPO v3 guarded s100 | 0.3275 | 0.1727 | 0.4125 | 0.9450 | 0.6369 | 0.0300 | 0.5500 | 0.5750 |
| DPO v4 targeted s50 | 0.3225 | 0.1655 | 0.3425 | 0.6750 | 0.5681 | 0.3000 | 0.5550 | 0.5800 |
| DPO v4 targeted s75 | 0.3175 | 0.1583 | 0.3375 | 0.6400 | 0.5600 | 0.3275 | 0.5550 | 0.5850 |
| DPO v4 targeted s100 | 0.3200 | 0.1655 | 0.3400 | 0.6275 | 0.5581 | 0.3400 | 0.5475 | 0.5800 |

## 当前模型选择

```text
正式 baseline: SFT
最佳 DPO candidate: DPO v2 checkpoint-50
DPO v3 guarded s100: 不晋升
DPO v4 targeted sweep: 不晋升，作为负结果消融保留
```

判断理由：

```text
SFT 是最稳的正式 baseline，EM、faithfulness、fabricated-number 等指标综合最好。
DPO v2 checkpoint-50 是最佳 DPO candidate：EM 接近 SFT，numeric EM 略高于 SFT，同时 citation precision 与 wrong-citation rate 明显更好。
DPO v3 没有超过 v2 s50，只能作为 guarded DPO 消融。
DPO v4 targeted DPO 没有修复预期问题，反而显著损伤 citation grounding，因此不晋升。
```

## 关键经验

DPO 在这个项目里不是“训练越多越好”。它能强化引用行为，但也可能带来数值尺度错误、answerability 回归和引用退化。

典型问题：

```text
-4.1 -> -0.041
18.75 -> 18750000.0
345.0 -> 0.345
unanswerable sample -> unsupported numeric answer
```

v4 targeted DPO 的负结果说明：即使 pair audit 通过，如果偏好对分布与主 eval 行为不匹配，或者 citation-repair pair 过强，也可能把模型从“谨慎引用”推向“形式上引用但证据不一致”。

## 下一步

```text
1. 保留 DPO v2 checkpoint-50 作为最佳 DPO candidate。
2. 把 v4 写成负结果消融，而不是继续盲目加 step。
3. 简历与面试重点讲完整实验链路、DPO tradeoff、badcase-driven iteration。
4. 后续若继续改进，应先重做 citation-repair pair 质量与分布，而不是直接训练更久。
```
