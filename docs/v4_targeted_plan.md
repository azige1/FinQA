# V4 Targeted DPO Plan

## Goal

V4 is an anti-regression preference-data iteration. It is not intended to make
the base model broadly smarter. Its goal is to preserve useful DPO citation
improvements while reducing numeric scale regressions and answerability
regressions.

## Current Failure Pattern

Post-hoc v3 audit shows that generic guarded DPO is still unstable:

```text
DPO v3 vs DPO v2 checkpoint-50:
exact improved: 4
exact regressed: 4
numeric improved: 2
numeric regressed: 3
```

Blocking examples:

```text
-4.1 -> -0.041
18.75 -> 18750000.0
345.0 -> 0.345
unanswerable -> unsupported numeric answer
```

## Data Rule

Use eval badcases only as taxonomy. Do not put held-out eval rows directly into
training data.

Valid v4 sources:

```text
data/unified/train_unified.jsonl
data/dpo/dpo_reweighted_v2.jsonl
data/dpo/dpo_guarded_v3.jsonl
```

Invalid v4 source:

```text
data/eval/eval.jsonl
results/*_scored.jsonl as direct chosen/rejected training rows
```

## Pair Buckets

### 1. Numeric Scale Guard

Purpose:

```text
prevent decimal, percent, sign, million/thousand, and unit-scale drift
```

Rejected variants:

```text
decimal_scale_shift: 345.0 -> 0.345
large_scale_shift: 18.75 -> 18750000.0
percent_fraction_shift: 4.1% -> 0.041
sign_flip: -4.1 -> 4.1
rounding_loss: 18.75 -> 18.8
last_digit_drift: 345.0 -> 346.0
```

Target rows:

```text
answer_type=number
grounding_type in {numeric_grounded, direct_grounded, calculation_hard}
answer contains extractable number
```

### 2. Protect-Correct Guard

Purpose:

```text
avoid DPO drift on patterns already handled well by SFT/v2
```

Construction:

```text
chosen: canonical grounded JSON answer from train row
rejected: answer with correct citation style but wrong number or truncated answer
```

### 3. Unanswerable Refusal Guard

Purpose:

```text
reduce forced-answer behavior on insufficient-evidence samples
```

Construction:

```text
chosen: low-confidence insufficient-evidence answer
rejected: fabricated numeric or entity answer with plausible citation
```

### 4. Citation Repair, Separate From Numeric

Purpose:

```text
keep v2 citation improvement without letting citation repair dominate numeric behavior
```

Construction:

```text
chosen: correct answer with valid quote/chunk_id
rejected: same or near-same answer with wrong citation or unsupported quote
```

Constraint:

```text
Do not mix citation-only repair rows into numeric guard buckets.
```

## Suggested Mix

First v4 pool:

```text
numeric_scale_guard: 240
protect_correct_guard: 160
unanswerable_refusal_guard: 120
citation_repair_guard: 160
total: 680
```

The total intentionally stays close to v2/v3 scale to avoid making the DPO
objective too different from prior runs.

Implemented local builder:

```bash
python scripts/build_dpo_targeted_v4.py
```

Expected artifacts:

```text
data/dpo/dpo_targeted_v4.jsonl
reports/dpo_targeted_v4_report.json
results/dpo_targeted_v4_pair_audit_120.jsonl
```

The generated v4 pool must keep `quality.invalid_rows=0`, all four bucket
quotas filled, and `leakage_guard.train_source_split={"train": 5000}` before
being used for a candidate run.

Current reviewed audit artifacts:

```text
results/dpo_targeted_v4_pair_audit_120.reviewed.jsonl
reports/dpo_targeted_v4_audit_report.json
```

The current v4 data-audit gate passes with 120/120 reviewed pairs marked
`ok_chosen_better`. This gate approves v4 for experimental candidate training,
not for model promotion.

## Training Plan

Use short checkpoint sweeps only:

```text
MAX_STEPS=50
MAX_STEPS=75
MAX_STEPS=100
BETA=0.05
LR=1e-5
```

Candidate A100 command:

```bash
MODEL_PATH=$MODEL_PATH \
SFT_ADAPTER_PATH=results/sft/qwen25_7b_finground_sft \
  bash scripts/run_dpo_v4_targeted_sweep_a10040.sh
```

This script trains an experimental v4 candidate with `MAX_STEPS=100`,
`SAVE_STEPS=25`, then evaluates checkpoints 50, 75, and 100. It sets
`ALLOW_UNREVIEWED_DPO=1` only to bypass the legacy v1 audit gate inside
`run_dpo_a10040.sh`; it separately requires
`reports/dpo_targeted_v4_audit_report.json` to have `gate_pass=true` unless
`ALLOW_UNREVIEWED_V4_DPO=1` is explicitly set for smoke/debug.

Stop if any of these regress materially versus DPO v2 checkpoint-50:

```text
numeric_exact_match
citation_consistency_score
forced_answer_rate
fabricated_number_rate
calculation_error_rate
```

## Acceptance Criteria

V4 can be considered a better DPO candidate only if:

```text
numeric_exact_match >= dpo_v2_s50
citation_consistency_score >= dpo_v2_s50 - 0.005
forced_answer_rate <= dpo_v2_s50
fabricated_number_rate <= dpo_v2_s50
posthoc high-severity regression count < dpo_v2_s50
```

V4 can replace SFT only if a manual audit confirms that DPO regressions are not
high severity. Current evidence does not support replacing SFT.
