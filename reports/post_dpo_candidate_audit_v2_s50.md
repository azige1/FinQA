# Post-DPO Candidate Audit: v2-s50

Generated: 2026-04-29T13:04:58

Candidate adapter: `results/dpo/qwen25_7b_finground_dpo_v2_reweighted_s100/checkpoint-50`

Status: **pending manual review**. This is not a formal acceptance gate pass.

## Files

- `reports/dpo_candidate_manifest_v2_s50.json`
- `results/post_dpo_candidate_audit_v2_s50.jsonl`
- `results/post_dpo_candidate_audit_v2_s50.csv`
- `reports/post_dpo_candidate_audit_summary_v2_s50.json`

## Counts

| reason | rows |
| --- | ---: |
| regressed_exact_vs_sft | 8 |
| regressed_numeric_exact_vs_sft | 5 |
| introduced_wrong_citation_vs_sft | 3 |
| introduced_calculation_error_vs_sft | 5 |
| unanswerable_slice | 6 |
| improved_exact_vs_sft | 7 |
| improved_numeric_exact_vs_sft | 6 |
| fixed_wrong_citation_vs_sft | 24 |
| fixed_calculation_error_vs_sft | 6 |

Unique rows requiring review: **42**

## Manual Gate

Review every row and fill `manual_is_dpo_better`, `manual_severity`, and `manual_notes`. Formal DPO acceptance remains blocked until this review passes.
