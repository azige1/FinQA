# Codex Post-DPO Candidate Review: v2-s50

Generated: 2026-04-29T13:09:18

This is a Codex-assisted review, not a formal human acceptance gate.

## Decision

**NO-GO for replacing SFT with DPO v2-s50 as the final model.**

v2-s50 remains the best DPO checkpoint so far, but the post-DPO audit found blocking regressions.

## Counts

| judgment | rows |
| --- | ---: |
| dpo_better | 10 |
| marginal_better_but_still_wrong | 16 |
| metric_artifact_or_tie | 2 |
| sft_better | 10 |
| tie_bad | 1 |
| tie_good | 3 |

| severity | rows |
| --- | ---: |
| critical | 2 |
| high | 4 |
| low | 6 |
| medium | 15 |
| none | 15 |

## Blockers

- Two unanswerable samples regress from correct refusal to forced unsupported answers.
- Several FinQA numeric samples regress from exact SFT answers to wrong DPO answers.
- Two TAT-QA samples keep correct answer text but introduce wrong citation flags.
- Some apparent DPO improvements are evaluator formatting artifacts rather than semantic gains.

## Blocking Rows

| id | severity | judgment | note |
| --- | --- | --- | --- |
| finqa_test_finqa336 | high | sft_better | DPO changes an exact SFT answer 148.36 into 0.4436. It fixes citation flag but introduces a severe numeric/calculation error. |
| finqa_test_finqa714 | high | sft_better | DPO changes exact 410.08 into 411.08, a material numeric regression. |
| synthetic_unanswerable_heldout_12 | critical | sft_better | DPO forces a fabricated numeric answer on an unanswerable sample where SFT correctly refused. |
| synthetic_unanswerable_heldout_32 | critical | sft_better | DPO forces an answer-like statement on an unanswerable sample where SFT correctly refused. |
| tatqa_validation_3404ada9eba910b36a2e3d0e0a8eee47 | high | sft_better | DPO collapses a correct explanatory span into only $5.3 million, losing the requested rationale. |
| synthetic_unanswerable_heldout_78 | high | tie_bad | Both SFT and DPO force unsupported numeric answers on an unanswerable sample; DPO is not worse, but the candidate still fails this case. |

## Next Action

Build DPO v3 data by adding hard negatives from post-DPO regressions and increasing unanswerable/precision protection, then run a short beta=0.05 checkpoint sweep.
