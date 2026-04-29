# Project Status - 2026-04-29

## Positioning

FinGround-QA is a grounded generation post-training project for financial
report QA. It is best presented as an LLM post-training and evaluation project,
not as a full financial foundation model or a production RAG system.

Core question:

```text
How do SFT and DPO change answer correctness, citation quality, numeric
grounding, and answerability behavior when evidence contexts are provided?
```

## Current Artifacts

Tracked in Git:

```text
source code
data construction scripts
small/medium training and eval JSONL artifacts
metrics, reports, badcase audits
runbooks and project docs
```

Not tracked in Git:

```text
LoRA adapter weights
optimizer states
logs
wandb
sync backups
raw all_unified.jsonl
```

Current Git commits:

```text
69d4e65 Initial FinGround-QA project snapshot
f58d369 Add DPO v3 posthoc audit
```

## Data Snapshot

```text
SFT train: 5000
SFT val: 500
Main eval: 400
FinanceBench audit: 150
Unified rows: 24283
Train/eval exact question overlap: 0
Evidence quote hit rate: 90.6%
```

## Model Runs

```text
Base: Qwen2.5-7B-Instruct
SFT: QLoRA SFT on grounded JSON answers
DPO v1: original balanced preference training, 500 steps
DPO v2: reweighted preference data, checkpoint-50 and checkpoint-100
DPO v3: guarded candidate with numeric and unanswerable guards, 100 steps
```

## Main Metrics

Error metrics are lower-is-better. Other metrics are higher-is-better.

| model | exact_match | numeric_exact_match | faithfulness_rate | citation_precision | citation_consistency_score | wrong_citation_rate | fabricated_number_rate | calculation_error_rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 0.1975 | 0.0504 | 0.1675 | 0.2487 | 0.2202 | 0.7600 | 0.6000 | 0.6300 |
| SFT | 0.3300 | 0.1727 | 0.4200 | 0.8950 | 0.6331 | 0.0825 | 0.5350 | 0.5750 |
| DPO v1 s500 | 0.2975 | 0.1475 | 0.4000 | 0.9450 | 0.6225 | 0.0125 | 0.5625 | 0.5925 |
| DPO v2 s100 | 0.3175 | 0.1691 | 0.4075 | 0.9525 | 0.6356 | 0.0350 | 0.5600 | 0.5775 |
| DPO v2 s50 | 0.3275 | 0.1763 | 0.4175 | 0.9575 | 0.6425 | 0.0300 | 0.5525 | 0.5725 |
| DPO v3 guarded s100 | 0.3275 | 0.1727 | 0.4125 | 0.9450 | 0.6369 | 0.0300 | 0.5500 | 0.5750 |

## Current Decision

```text
Formal baseline: SFT
Best DPO candidate for analysis: DPO v2 checkpoint-50
Do not promote DPO v3 guarded s100
```

Rationale:

```text
SFT remains the safest final model by exact match, faithfulness, unsupported
claim rate, and fabricated-number rate.

DPO v2 checkpoint-50 is the best DPO candidate: it improves citation precision
and citation consistency while keeping exact match close to SFT.

DPO v3 guarded s100 does not beat v2 checkpoint-50 on numeric exact match or
citation consistency, and it remains blocked by pending manual pair audit.
```

## Key Learning

DPO is not a free improvement in this project. It improves citation behavior,
especially wrong-citation rate, but can introduce numeric scale regressions and
answerability regressions. This tradeoff is the core technical story for
resume and interview discussion.

Concrete v3 regression examples:

```text
-4.1 -> -0.041
18.75 -> 18750000.0
345.0 -> 0.345
unanswerable sample -> unsupported numeric answer
```

## Next Technical Step

Build a v4 targeted DPO data pool from train split rows only. The v4 pool should
use eval badcases only as an error taxonomy, not as training examples.

Targeted guards:

```text
numeric scale and decimal protection
unanswerable refusal protection
protect-correct preference pairs
separated citation-repair and numeric-answer pair buckets
```

