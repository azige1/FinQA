# DPO Model Selection Report

Generated: 2026-04-29T15:11:36

Formal recommendation: **keep_sft_as_final_baseline**

Best DPO candidate: **dpo_v2_s50**

Reason: DPO v2 s50 still has the best DPO score and numeric/citation balance. DPO v3 does not beat v2 s50 on numeric exact match or citation consistency, and v3 remains blocked by pending manual pair audit.

## Overall Metrics

| model | exact_match | numeric_exact_match | faithfulness_rate | citation_precision | citation_consistency_score | wrong_citation_rate | fabricated_number_rate | calculation_error_rate | selection_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | 0.1975 | 0.0504 | 0.1675 | 0.2487 | 0.2202 | 0.7600 | 0.6000 | 0.6300 | -1.1376 |
| sft | 0.3300 | 0.1727 | 0.4200 | 0.8950 | 0.6331 | 0.0825 | 0.5350 | 0.5750 | 1.1365 |
| dpo_v1_s500 | 0.2975 | 0.1475 | 0.4000 | 0.9450 | 0.6225 | 0.0125 | 0.5625 | 0.5925 | 1.0145 |
| dpo_v2_s100 | 0.3175 | 0.1691 | 0.4075 | 0.9525 | 0.6356 | 0.0350 | 0.5600 | 0.5775 | 1.1281 |
| dpo_v2_s50 | 0.3275 | 0.1763 | 0.4175 | 0.9575 | 0.6425 | 0.0300 | 0.5525 | 0.5725 | 1.1946 |
| dpo_v3_guarded_s100 | 0.3275 | 0.1727 | 0.4125 | 0.9450 | 0.6369 | 0.0300 | 0.5500 | 0.5750 | 1.1717 |

## Ranking By Selection Score

- dpo_v2_s50: 1.1946
- dpo_v3_guarded_s100: 1.1717
- sft: 1.1365
- dpo_v2_s100: 1.1281
- dpo_v1_s500: 1.0145
- base: -1.1376

## Gate Notes

- SFT remains the safest final baseline because post-DPO audit blocks replacing it.
- DPO v2 s50 remains the best DPO candidate for analysis, not a formal replacement.
- DPO v3 guarded s100 is experimental: formal_human_gate_pass=false and gate_pass=false.

## Key Deltas Vs SFT

| model | exact_match | numeric_exact_match | citation_precision | fabricated_number_rate | forced_answer_rate |
| --- | --- | --- | --- | --- | --- |
| base | -0.1325 | -0.1223 | -0.6462 | 0.0650 | -0.0025 |
| dpo_v1_s500 | -0.0325 | -0.0252 | 0.0500 | 0.0275 | 0.0000 |
| dpo_v2_s100 | -0.0125 | -0.0036 | 0.0575 | 0.0250 | 0.0050 |
| dpo_v2_s50 | -0.0025 | 0.0036 | 0.0625 | 0.0175 | 0.0050 |
| dpo_v3_guarded_s100 | -0.0025 | 0.0000 | 0.0500 | 0.0150 | 0.0000 |
