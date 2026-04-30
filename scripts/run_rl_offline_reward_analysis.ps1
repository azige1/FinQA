$ErrorActionPreference = "Stop"

python scripts\analyze_offline_rewards.py `
  --eval-file data\eval\eval.jsonl `
  --prediction base=results\base_predictions.jsonl `
  --prediction sft=results\sft_predictions.jsonl `
  --prediction dpo_v1_s500=results\dpo_predictions.jsonl `
  --prediction dpo_v2_s50=results\dpo_v2_reweighted_s50_predictions.jsonl `
  --prediction dpo_v2_s100=results\dpo_v2_reweighted_s100_predictions.jsonl `
  --prediction dpo_v3_s100=results\dpo_v3_guarded_s100_predictions.jsonl `
  --output results\rl_offline_reward_scored.jsonl `
  --report reports\rl_offline_reward_report.json
