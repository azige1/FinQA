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
Error-Type Balanced DPO
```

## Main Analysis

```text
Citation Consistency Score
Error Delta by error type
Answerability-aware over_refusal / forced_answer analysis
```

## Saved Predictions

```text
results/base_predictions.jsonl
results/sft_predictions.jsonl
results/dpo_predictions.jsonl
```

## Saved Metrics

```text
results/base_metrics.json
results/sft_metrics.json
results/dpo_metrics.json
reports/error_delta_report.json
```
