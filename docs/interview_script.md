# Interview Script

## One-Minute Version

This project is not a financial expert model or a RAG system. I use financial
report QA as a grounded generation case study to analyze how SFT and DPO change
model behavior when evidence contexts are provided.

The key work is data quality and preference quality:

```text
1. Convert TAT-QA and FinQA into a unified question-context-evidence schema.
2. Validate evidence quote hits, numeric grounding, table linearization, and leakage.
3. Train SFT to follow a grounded JSON answer protocol.
4. Mine SFT failure cases and combine them with hard negatives for DPO.
5. Audit preference pairs to control length bias, format bias, and weak rejected samples.
6. Evaluate Base -> SFT -> DPO with faithfulness/citation/over-refusal proxies and manual badcases.
```

## Main Result

```text
Base EM: 19.75%
SFT EM: 33.00%
DPO v2 checkpoint-50 EM: 32.75%
DPO v2 checkpoint-50 citation precision: 95.75%
DPO v2 checkpoint-50 wrong-citation rate: 3.00%
```

The important point is not that DPO universally beats SFT. SFT remains the
safest final baseline. DPO v2 checkpoint-50 is the best DPO candidate because
it improves citation quality while keeping exact match close to SFT.

## Tradeoff Story

```text
DPO improved citation behavior but introduced some numeric and answerability
regressions. For example, v3 still produced scale errors such as:

-4.1 -> -0.041
18.75 -> 18750000.0
345.0 -> 0.345
```

So the next iteration is not simply more DPO steps. It is targeted guardrail
preference data: numeric scale guards, unanswerable refusal guards,
protect-correct pairs, and citation repair separated from numeric reasoning.

## What This Shows

```text
1. I can build a complete post-training experiment pipeline.
2. I care about data quality and preference quality, not only training loss.
3. I can analyze DPO tradeoffs instead of over-claiming a single metric.
4. I use badcase analysis to decide the next data iteration.
```
