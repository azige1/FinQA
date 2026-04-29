# 实验卡片

## 项目名

```text
FinGround-QA: Error-Type Balanced DPO for Financial Grounded Generation
```

## 模型与训练

```text
Base: Qwen2.5-7B-Instruct
SFT: QLoRA
DPO: QLoRA
GPU: A100/A800 40GB
主评测集: 400 held-out samples
```

## 实验矩阵

```text
Base
SFT-grounded
DPO v1 s500
DPO v2 reweighted checkpoint-50 / checkpoint-100
DPO v3 guarded s100
DPO v4 targeted checkpoint-50 / checkpoint-75 / checkpoint-100
```

## 核心分析

```text
Citation Consistency Score
Error Delta by error type
Answerability-aware over_refusal / forced_answer analysis
Post-hoc DPO flip analysis
Numeric scale regression audit
Targeted DPO negative-result analysis
```

## 当前模型选择

```text
正式 baseline: SFT
最佳 DPO candidate: DPO v2 checkpoint-50
DPO v3 guarded s100: 不晋升
DPO v4 targeted sweep: 不晋升，作为负结果消融
```

理由：

```text
DPO v2 checkpoint-50 在 numeric exact match、citation precision 和 citation consistency 上取得最佳 DPO 综合平衡。
SFT 仍是最稳正式 baseline，因为 EM、faithfulness 和 fabricated-number 指标更稳。
DPO v4 targeted sweep 没有超过 v2，且 citation grounding 明显退化。
```

## 已保存预测

```text
results/base_predictions.jsonl
results/sft_predictions.jsonl
results/dpo_predictions.jsonl
results/dpo_v2_reweighted_s50_predictions.jsonl
results/dpo_v3_guarded_s100_predictions.jsonl
results/dpo_v4_targeted_s50_retry_predictions.jsonl
results/dpo_v4_targeted_s75_retry_predictions.jsonl
results/dpo_v4_targeted_s100_retry_predictions.jsonl
```

## 已保存指标

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
reports/final_experiment_report_zh.md
reports/dpo_model_selection_report.md
reports/dpo_v3_posthoc_audit_report.md
```
