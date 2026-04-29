# Data Card

## Sources

```text
TAT-QA: main training and eval source for tabular/textual financial report QA.
FinQA: supplemental numerical financial QA source.
FinanceBench: external high-quality audit sample only.
```

## Unified Schema

Each sample is converted to:

```text
question
contexts with chunk_id and type
answer
evidence with chunk_id and quote
answer_type
reasoning_type
grounding_type
```

## Grounding Types

```text
direct_grounded: answer/evidence directly appears in context.
numeric_grounded: answer number appears in evidence/context.
calculation_hard: derivation is complex or direct numeric grounding is weak.
unanswerable: context is insufficient.
```

## Required Reports

```text
data_quality_report.json
evidence_quote_hit_report.json
numeric_grounding_report.json
table_linearization_report.json
answer_type_distribution.json
reasoning_type_distribution.json
answerability_report.json
train_eval_leakage_report.json
answerability_eval_report.json
data_difficulty_report.json
```

## Non-Toy Diagnostics

`answerability_eval.jsonl` is a balanced auxiliary eval set for answerability
calibration. It is reported separately from the main held-out eval so that
forced-answer and over-refusal behavior are not hidden by direct-grounded
samples.

`data_difficulty_audit_100.jsonl` is an audit worksheet with automatic proxy
flags for trivial lookup, numeric reasoning, distractor contexts, citation
confusability, direct answer copying, and grounded sample quality. These flags
are diagnostics for manual review, not final human labels.
