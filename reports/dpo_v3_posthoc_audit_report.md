# DPO v3 Post-hoc Audit

Generated: 2026-04-29T15:11:36

Conclusion: **NO_GO_FOR_PROMOTING_V3. V3 fixes some v2 rows but introduces offsetting exact/numeric regressions and remains below v2_s50 on the main DPO selection balance.**

Delta audit rows: **18**

Persistent all-wrong rows tracked separately: **261**

## Flip Summary

| comparison | improved_exact | regressed_exact | net_exact | improved_numeric | regressed_numeric | net_numeric |
| --- | --- | --- | --- | --- | --- | --- |
| dpo_v2_s50_vs_sft | 7 | 8 | -1 | 6 | 5 | 1 |
| dpo_v3_guarded_s100_vs_sft | 6 | 7 | -1 | 5 | 5 | 0 |
| dpo_v3_guarded_s100_vs_dpo_v2_s50 | 4 | 4 | 0 | 2 | 3 | -1 |

## Category Counts

| category | count |
| --- | --- |
| v3_regressed_vs_sft_exact | 7 |
| persistent_all_wrong | 6 |
| v3_fixed_fabricated_number | 5 |
| v3_introduced_unsupported_claim | 5 |
| v3_fixed_unsupported_claim | 4 |
| v3_fixed_v2_sft_regression | 4 |
| v3_improved_vs_v2_exact | 4 |
| v3_introduced_fabricated_number | 4 |
| v3_regressed_vs_v2_exact | 4 |
| v3_introduced_calculation_error | 3 |
| v3_introduced_missing_evidence | 3 |
| v3_introduced_over_refusal | 3 |
| v3_new_exact_regression_vs_sft_and_v2 | 3 |
| v3_numeric_regression_vs_v2 | 3 |
| v3_fixed_calculation_error | 2 |
| v3_fixed_forced_answer | 2 |

## Blocking Examples

- finqa_test_finqa413 (FinQA, direct_grounded): v3_introduced_calculation_error, v3_introduced_fabricated_number, v3_introduced_unsupported_claim, v3_new_exact_regression_vs_sft_and_v2; gold='-4.1'; v2='-4.1'; v3='-0.041'
- tatqa_validation_948d7e92b4cbf2aa786183e6b321c2a9 (TAT-QA, direct_grounded): v3_new_exact_regression_vs_sft_and_v2, v3_regressed_vs_sft_exact, v3_regressed_vs_v2_exact; gold='$55 million'; v2='$55 million'; v3='(In millions)'
- finqa_test_finqa820 (FinQA, direct_grounded): v3_introduced_calculation_error, v3_introduced_fabricated_number, v3_introduced_unsupported_claim, v3_numeric_regression_vs_v2; gold='18.75'; v2='18.75'; v3='18750000.0'
- finqa_valid_finqa633 (FinQA, direct_grounded): v3_introduced_calculation_error, v3_introduced_fabricated_number, v3_introduced_unsupported_claim, v3_new_exact_regression_vs_sft_and_v2; gold='345.0'; v2='345.0'; v3='0.345'
- synthetic_unanswerable_heldout_78 (SyntheticUnanswerable, unanswerable): persistent_all_wrong, v3_forced_answer_unanswerable; gold='Insufficient evidence to answer from the provided contexts.'; v2='-0.25'; v3='-0.1855'

## Next Data Actions

- Do not increase generic DPO steps before auditing the v3 regression rows.
- Create a small targeted v4 pool from v3_new_exact_regression_vs_sft_and_v2 and v3_numeric_regression_vs_v2.
- Keep unanswerable guards, but reduce pressure that converts correct refusals into answer-like outputs.
- Separate citation-repair pairs from numeric-answer pairs so citation gains do not trade off exact numeric accuracy.
