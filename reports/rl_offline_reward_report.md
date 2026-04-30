# RL Offline Reward Analysis

## Setup

This report scores existing held-out predictions with a transparent grounded QA rule reward. It is an RL-style diagnostic, not a learned human preference reward model.

Inputs:

```text
data/eval/eval.jsonl
results/base_predictions.jsonl
results/sft_predictions.jsonl
results/dpo_predictions.jsonl
results/dpo_v2_reweighted_s50_predictions.jsonl
results/dpo_v2_reweighted_s100_predictions.jsonl
results/dpo_v3_guarded_s100_predictions.jsonl
```

Outputs:

```text
reports/rl_offline_reward_report.json
results/rl_offline_reward_scored.jsonl
```

## Reward Components

Positive terms:

```text
json_valid
schema_pass
exact_match
numeric_exact_match
faithfulness_proxy
citation_precision
citation_consistency_score
```

Penalty terms:

```text
wrong_citation
unsupported_claim
fabricated_number
calculation_error
over_refusal
forced_answer
format_error
schema_error
generic_answer
```

## Results

| model | reward_mean | raw_reward_mean | exact | citation_precision | wrong_citation | calculation_error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | -0.2995 | -1.7903 | 0.1975 | 0.2487 | 0.7600 | 0.6300 |
| SFT | 0.3237 | 2.0261 | 0.3300 | 0.8950 | 0.0825 | 0.5750 |
| DPO v1 s500 | 0.3088 | 1.9330 | 0.2975 | 0.9450 | 0.0125 | 0.5925 |
| DPO v2 s50 | **0.3387** | **2.1250** | 0.3275 | **0.9575** | 0.0300 | **0.5725** |
| DPO v2 s100 | 0.3276 | 2.0516 | 0.3175 | 0.9525 | 0.0350 | 0.5775 |
| DPO v3 s100 | 0.3343 | 2.0959 | 0.3275 | 0.9450 | 0.0300 | 0.5750 |

Pairwise reward comparison:

```text
DPO v2 s50 vs SFT:
comparable prompts: 400
DPO v2 s50 win_rate: 7.75%
tie_rate: 88.25%
SFT win_rate: 4.00%
```

## Interpretation

DPO v2 s50 has the highest rule reward, matching the existing model-selection decision. However, SFT and DPO v2 s50 tie on most prompts under this reward, which means the reward is not a license to blindly continue RL optimization.

Future PPO/GRPO runs must monitor:

```text
KL
schema_pass_rate
exact_match
numeric_exact_match
citation_precision
wrong_citation_rate
forced_answer_rate
over_refusal_rate
```

Reward increase without these gates should be treated as reward hacking.
