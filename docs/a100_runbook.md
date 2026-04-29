# A100 40GB Runbook

This runbook keeps the GPU instance focused on inference, SFT, DPO, and required evaluation. Do not use paid GPU time for data debugging or report writing.

## 1. Upload

Package the local project without models or checkpoints:

```bash
tar -czf FinGround-QA-upload.tar.gz \
  --exclude=FinGround-QA/.git \
  --exclude=FinGround-QA/models \
  --exclude=FinGround-QA/checkpoints \
  --exclude=FinGround-QA/logs \
  --exclude=FinGround-QA/**/__pycache__ \
  FinGround-QA
```

On the server:

```bash
tar -xzf FinGround-QA-upload.tar.gz
cd FinGround-QA
pip install -r requirements.txt
```

## 2. Model

```bash
export HF_ENDPOINT=https://hf-mirror.com
mkdir -p /root/autodl-tmp/models
hf download Qwen/Qwen2.5-7B-Instruct \
  --local-dir /root/autodl-tmp/models/Qwen2.5-7B-Instruct
export MODEL_PATH=/root/autodl-tmp/models/Qwen2.5-7B-Instruct
```

## 3. Base Eval

```bash
MODEL_PATH=$MODEL_PATH bash scripts/run_base_eval_a10040.sh
```

Expected outputs:

```text
results/base_predictions.jsonl
results/base_metrics.json
results/base_scored.jsonl
```

## 4. SFT

Use `data/sft/sft_train.jsonl` and `data/sft/sft_val.jsonl`.

Recommended first run:

```text
QLoRA, bf16, model_max_length=2048, batch_size=1,
gradient_accumulation_steps=16, lora_rank=8,
learning_rate=1e-4, epochs=1
```

Save the adapter under:

```text
results/sft/qwen25_7b_finground_sft
```

Run:

```bash
MODEL_PATH=$MODEL_PATH EPOCHS=1 bash scripts/run_sft_a10040.sh
MODEL_PATH=$MODEL_PATH \
SFT_ADAPTER_PATH=results/sft/qwen25_7b_finground_sft \
  bash scripts/run_sft_eval_a10040.sh
```

After training, evaluate the SFT checkpoint and save:

```text
results/sft_predictions.jsonl
results/sft_metrics.json
results/sft_scored.jsonl
```

## 5. Model-Mined Rejected

Generate SFT outputs on train/val/eval-like prompts, then run:

```bash
BASE_MODEL_PATH=$MODEL_PATH \
SFT_ADAPTER_PATH=results/sft/qwen25_7b_finground_sft \
  bash scripts/run_model_mining_a10040.sh
```

Expected outputs:

```text
results/sft_mining_predictions.jsonl
data/dpo/model_mined_pairs.jsonl
reports/model_mined_rejected_report.json
```

## 6. DPO Mix

After manual review of `results/preference_pair_audit_100.jsonl`, mix final DPO v1:

```bash
bash scripts/run_summarize_preference_audit.sh
bash scripts/run_mix_dpo_v1.sh
```

`run_mix_dpo_v1.sh` refuses to create formal `dpo_balanced_v1.jsonl` without
`data/dpo/model_mined_pairs.jsonl`. Use `ALLOW_RULE_ONLY_DPO=1` only for
smoke/debug, never for reported DPO results.

Expected output:

```text
data/dpo/dpo_balanced_v1.jsonl
reports/preference_pair_quality_report.json
reports/pair_difficulty_report.json
reports/preference_pair_audit_report.json
```

Only start DPO if the audit gate passes:

```text
is_chosen_better >= 85%
ambiguous_pair_rate <= 10%
chosen_also_bad_rate <= 10%
rejected_too_easy_rate <= 25%
format_bias_rate <= 15%
```

`scripts/run_dpo_a10040.sh` enforces this gate by default. Use
`ALLOW_UNREVIEWED_DPO=1` only for smoke/debug runs, never for formal reported
results.

## 7. DPO Eval And Error Delta

Run DPO:

```bash
MODEL_PATH=$MODEL_PATH \
SFT_ADAPTER_PATH=results/sft/qwen25_7b_finground_sft \
MAX_STEPS=500 \
  bash scripts/run_dpo_a10040.sh
```

Then evaluate:

```bash
MODEL_PATH=$MODEL_PATH \
DPO_ADAPTER_PATH=results/dpo/qwen25_7b_finground_dpo \
  bash scripts/run_dpo_eval_a10040.sh
```

After DPO eval, save:

```text
results/dpo_predictions.jsonl
results/dpo_metrics.json
results/dpo_scored.jsonl
```

Then run:

```bash
bash scripts/run_error_delta.sh
bash scripts/run_export_badcases.sh
```

Expected output:

```text
reports/error_delta_report.json
results/badcase_50.jsonl
```

## 8. Shut Down

Copy back these files before shutting down the GPU:

```text
results/base_predictions.jsonl
results/sft_predictions.jsonl
results/dpo_predictions.jsonl
results/base_metrics.json
results/sft_metrics.json
results/dpo_metrics.json
results/*_scored.jsonl
reports/error_delta_report.json
reports/model_mined_rejected_report.json
reports/preference_pair_quality_report.json
reports/pair_difficulty_report.json
data/dpo/model_mined_pairs.jsonl
data/dpo/dpo_balanced_v1.jsonl
```

Badcase analysis, README polishing, cards, and interview notes should be finished locally after the GPU is stopped.
