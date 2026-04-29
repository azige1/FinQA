# Resume Project Brief

## Resume Title

```text
Financial Report Grounded QA Post-training: SFT, DPO, and Error-Type Evaluation
```

## One-Sentence Pitch

Built a financial report QA post-training pipeline on Qwen2.5-7B-Instruct,
covering evidence-grounded SFT, model-mined DPO preference construction,
multi-version DPO checkpoint selection, and fine-grained hallucination/numeric
regression analysis.

## Strong Resume Bullets

```text
- Built a grounded financial QA post-training pipeline over TAT-QA and FinQA,
  converting heterogeneous samples into a unified question-context-evidence
  schema with quote-level citation validation and train/eval leakage checks.

- Trained Qwen2.5-7B-Instruct with QLoRA SFT and multiple DPO variants using
  model-mined rejected answers, rule-generated hard negatives, and
  answerability-aware preference pairs.

- Designed error-type evaluation covering exact match, numeric exact match,
  citation precision, citation consistency, wrong citation, fabricated number,
  calculation error, over-refusal, and forced-answer behavior.

- Improved base-model EM from 19.75% to 33.00% with SFT on a 400-sample held-out
  eval set; identified DPO v2 checkpoint-50 as the best DPO candidate with
  citation precision 95.75% and wrong-citation rate 3.00%.

- Performed post-DPO badcase and flip analysis, finding that DPO improves
  citation behavior but can introduce numeric-scale and unanswerable forced-answer
  regressions; designed a targeted v4 guardrail data plan instead of blindly
  increasing DPO steps.
```

## Short Interview Version

```text
This is not a full financial large model or a production RAG system. I use
financial report QA as a controlled grounded-generation case study. The main
work is to study how SFT and DPO change model behavior when the model is already
given evidence contexts.

The strongest result is not that DPO universally wins. SFT is still the safest
baseline. DPO v2 improves citation precision and wrong-citation rate, but the
post-hoc audit shows tradeoffs in numeric exactness and unanswerable behavior.
That is why I built an error-type audit and designed a targeted v4 DPO data
plan focused on anti-regression rather than blind preference optimization.
```

## Numbers To Remember

```text
Main eval size: 400
SFT train size: 5000
Base EM: 19.75%
SFT EM: 33.00%
DPO v2 checkpoint-50 EM: 32.75%
DPO v2 checkpoint-50 numeric EM: 17.63%
SFT citation precision: 89.50%
DPO v2 checkpoint-50 citation precision: 95.75%
SFT wrong-citation rate: 8.25%
DPO v2 checkpoint-50 wrong-citation rate: 3.00%
DPO v3 delta audit rows: 18
Persistent all-wrong rows across SFT/v2/v3: 261
```

## What To Emphasize

```text
1. I understand post-training tradeoffs, not just how to run SFT/DPO scripts.
2. I built data quality checks before training.
3. I separated citation improvements from numeric regressions.
4. I used badcase analysis to drive the next data iteration.
5. I avoided claiming DPO as a universal win when the audit did not support it.
```

## What Not To Claim

```text
- Do not claim this is a financial foundation model.
- Do not claim this is a complete RAG system.
- Do not claim DPO v3 beats all previous versions.
- Do not claim the proxy faithfulness metrics are full factuality judgments.
- Do not claim FinanceBench was used for training.
```

## Recommended Chinese Resume Bullet

```text
金融报告问答 Grounded Generation 后训练项目：基于 Qwen2.5-7B-Instruct 构建
TAT-QA/FinQA 统一 evidence schema，完成 QLoRA SFT、model-mined DPO preference
构造、多版本 DPO checkpoint selection 与 badcase audit。SFT 将 400 条 held-out
eval EM 从 19.75% 提升至 33.00%；DPO v2 checkpoint-50 将 citation precision 从
89.50% 提升至 95.75%、wrong-citation rate 从 8.25% 降至 3.00%。进一步通过
post-hoc flip analysis 发现 DPO 的 numeric scale regression 和 unanswerable
forced-answer 副作用，并设计 targeted v4 guardrail preference data。
```

