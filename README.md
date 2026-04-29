# FinGround-QA

FinGround-QA is a small-scale, high-quality grounded generation post-training
experiment for financial report QA.

Project positioning:

```text
FinGround-QA: Error-Type Balanced DPO for Financial Grounded Generation
```

It is not a financial large model, a full RAG system, a retrieval system, or a
table parsing system. The first version studies how SFT and model-mined DPO
change model behavior when the model is given a question and evidence contexts.
The core innovation is:

```text
1. Error-Type Balanced DPO
2. Answerability-aware Hard Negatives
3. Citation Consistency Score + Error Delta Analysis
```

## Main Pipeline

```text
TAT-QA / FinQA data standardization
-> evidence-grounded SFT data
-> Base Qwen2.5-7B-Instruct eval
-> 7B QLoRA SFT
-> SFT model-mined rejected
-> Error-Type Balanced DPO
-> Citation Consistency / Error Delta / Badcase Audit
```

## Current Scope

In scope:

```text
unified question-context-evidence schema
evidence quote checking
numeric grounding proxy
table-to-markdown linearization
SFT grounded JSON format
DPO high-confidence preference pairs
answerability-aware hard negatives
Error-Type Balanced DPO reports
Citation Consistency Score proxy
rule/weak-semantic checker
manual preference/output audit files
```

Out of scope:

```text
GRPO / PPO / IPO
multi-model comparison
complete RAG retrieval stack
PDF parsing / OCR
complex table reasoning
financial professional advice
```

## Local Phase 0

```bash
python -m src.finground_qa.pipeline prepare-data --output-dir .
python -m src.finground_qa.pipeline validate-sft \
  --file data/sft/sft_train.jsonl \
  --output reports/validate_sft_train.json
python -m src.finground_qa.pipeline build-rule-dpo \
  --unified-train data/unified/train_unified.jsonl \
  --target 600 \
  --output data/dpo/rule_dpo_pairs.jsonl \
  --report reports/preference_pair_quality_report.json
python -m src.finground_qa.pipeline audit-pairs \
  --pairs data/dpo/rule_dpo_pairs.jsonl \
  --output results/preference_pair_audit_100.jsonl
python -m src.finground_qa.pipeline summarize-audit \
  --audit results/preference_pair_audit_100.jsonl \
  --output reports/preference_pair_audit_report.json
python -m src.finground_qa.pipeline build-answerability-eval
python -m src.finground_qa.pipeline data-difficulty-audit
```

Current generated v1 data summary:

```text
SFT train: 5000
SFT val: 500
Eval: 400
FinanceBench audit: 150
Unified rows: 24283
Train/eval exact question overlap: 0
Evidence quote hit rate: 90.6%
Unanswerable rows in train/val/eval reports: 111
```

This produces:

```text
data/unified/train_unified.jsonl
data/unified/val_unified.jsonl
data/unified/eval_unified.jsonl
data/sft/sft_train.jsonl
data/sft/sft_val.jsonl
data/eval/eval.jsonl
data/eval/answerability_eval.jsonl
data/eval/financebench_audit.jsonl
results/data_difficulty_audit_100.jsonl
reports/*.json
```

## Main Metrics

```text
exact_match
numeric_exact_match
faithfulness_rate
unsupported_claim_rate
citation_precision
citation_consistency_score
chunk_valid_rate
quote_hit_rate
number_coverage_rate
entity_or_token_coverage_rate
missing_evidence_rate
wrong_citation_rate
fabricated_number_rate
calculation_error_rate
over_refusal_rate
forced_answer_rate
generic_answer_rate
format_error_rate
schema_pass_rate
```

## Limitations

Faithfulness, unsupported-claim scores, and Citation Consistency Score are
rule-based and weak-semantic proxy metrics. They are not full factuality
judgments. Final claims must be supported by manual audit and badcase analysis.

FinanceBench is used only as an external audit set in v1 and is not used for
training.

See `docs/a100_runbook.md` for the GPU execution order.
