# Experiment Card

```text
FinGround-QA: Error-Type Balanced DPO for Financial Grounded Generation
```

## Model

```text
Base: Qwen2.5-7B-Instruct
SFT: QLoRA
DPO: QLoRA
GPU: 1 x A100 40GB
```

## Main Matrix

```text
Base
SFT-grounded
DPO v1 s500
DPO v2 reweighted checkpoint-50 / checkpoint-100
DPO v3 guarded s100 candidate
```

## Main Analysis

```text
Citation Consistency Score
Error Delta by error type
Answerability-aware over_refusal / forced_answer analysis
Post-hoc DPO flip analysis
Numeric scale regression audit
```

## Current Model Selection

```text
Formal baseline: SFT
Best DPO candidate: DPO v2 checkpoint-50
DPO v3 guarded s100: no-go for promotion
```

Reason:

```text
DPO v2 checkpoint-50 has the best DPO balance across numeric exact match,
citation precision, and citation consistency. SFT remains the safest final
baseline because post-DPO audit still finds high-severity regressions.
```

## Saved Predictions

```text
results/base_predictions.jsonl
results/sft_predictions.jsonl
results/dpo_predictions.jsonl
results/dpo_v2_reweighted_s50_predictions.jsonl
results/dpo_v3_guarded_s100_predictions.jsonl
```

## Saved Metrics

```text
results/base_metrics.json
results/sft_metrics.json
results/dpo_metrics.json
results/dpo_v2_reweighted_s50_metrics.json
results/dpo_v3_guarded_s100_metrics.json
reports/error_delta_report.json
reports/dpo_model_selection_report.md
reports/dpo_v3_posthoc_audit_report.md
```
