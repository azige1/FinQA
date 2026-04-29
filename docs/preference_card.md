# Preference Card

Project preference objective:

```text
Error-Type Balanced DPO for Financial Grounded Generation.
```

## DPO v1 Target

```text
1000 high-confidence Error-Type Balanced DPO pairs.
```

## Pair Sources

```text
model_mined: target 40% after SFT checkpoint exists.
rule_generated: part of the seed pool.
numeric_corruption / wrong_citation_corruption / hard_negative:
  answerability-aware and error-type balanced hard negatives.
```

The local phase initially builds rule/hard-negative candidate pairs. Final DPO
v1 should mix in model-mined rejected after SFT.

## Reject Types

```text
unsupported_claim
wrong_citation
fabricated_number
calculation_error
missing_evidence
context_contradiction
over_refusal
forced_answer
generic_answer
wrong_format
```

## Answerability-Aware Hard Negatives

```text
answerable + over_refusal:
  context is sufficient, but rejected output refuses or uses low confidence.

unanswerable + forced_answer:
  context is insufficient, but rejected output gives a high-confidence answer.
```

The local rule seed currently includes both types. Final DPO v1 must still mix
rule pairs with model-mined rejected from the SFT checkpoint.

## Audit Gate

```text
is_chosen_better >= 85%
ambiguous_pair_rate <= 10%
chosen_also_bad_rate <= 10%
rejected_too_easy_rate <= 25%
format_bias_rate <= 15%
chosen/rejected length ratio not abnormal
```
