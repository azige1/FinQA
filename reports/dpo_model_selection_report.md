# DPO Model Selection Report

Generated: 2026-04-29T12:54:42

Recommended checkpoint: **dpo_v2_s50**
Adapter path: `results/dpo/qwen25_7b_finground_dpo_v2_reweighted_s100/checkpoint-50`

## Overall Metrics

| model | exact_match | numeric_exact_match | faithfulness_rate | citation_precision | citation_consistency_score | number_coverage_rate | wrong_citation_rate | unsupported_claim_rate | fabricated_number_rate | calculation_error_rate | over_refusal_rate | forced_answer_rate | schema_pass_rate | selection_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Base | 0.1975 | 0.0504 | 0.1675 | 0.2487 | 0.2202 | 0.2150 | 0.7600 | 0.6250 | 0.6000 | 0.6300 | 0.0700 | 0.0000 | 0.2500 | -0.6619 |
| SFT | 0.3300 | 0.1727 | 0.4200 | 0.8950 | 0.6331 | 0.3500 | 0.0825 | 0.5400 | 0.5350 | 0.5750 | 0.0100 | 0.0025 | 0.9925 | 2.1856 |
| DPO v1 s500 | 0.2975 | 0.1475 | 0.4000 | 0.9450 | 0.6225 | 0.3250 | 0.0125 | 0.5900 | 0.5625 | 0.5925 | 0.0300 | 0.0025 | 0.9950 | 1.8900 |
| DPO v2 s100 | 0.3175 | 0.1691 | 0.4075 | 0.9525 | 0.6356 | 0.3250 | 0.0350 | 0.5625 | 0.5600 | 0.5775 | 0.0050 | 0.0075 | 0.9950 | 2.1221 |
| DPO v2 s50 | 0.3275 | 0.1763 | 0.4175 | 0.9575 | 0.6425 | 0.3325 | 0.0300 | 0.5550 | 0.5525 | 0.5725 | 0.0050 | 0.0075 | 0.9950 | 2.2200 |

## DPO Ranking

- dpo_v2_s50: 2.2200
- dpo_v2_s100: 2.1221
- dpo_v1_s500: 1.8900

## Key Deltas vs SFT

| model | exact_match | numeric_exact_match | faithfulness_rate | fabricated_number_rate | calculation_error_rate | over_refusal_rate |
| --- | --- | --- | --- | --- | --- | --- |
| dpo_v1_s500 | -0.0325 | -0.0252 | -0.0200 | 0.0275 | 0.0175 | 0.0200 |
| dpo_v2_s100 | -0.0125 | -0.0036 | -0.0125 | 0.0250 | 0.0025 | -0.0050 |
| dpo_v2_s50 | -0.0025 | 0.0036 | -0.0025 | 0.0175 | -0.0025 | -0.0050 |

## Notes

- Lower is better for error rates: wrong_citation, unsupported_claim, fabricated_number, calculation_error, over_refusal, forced_answer.
- The selection score is a compact heuristic for checkpoint choice, not a replacement for targeted error inspection.
