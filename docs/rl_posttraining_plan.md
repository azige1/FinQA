# RL-style 后训练扩展计划

## 目标

这个扩展不是为了把金融 QA 指标硬刷到最高，而是把项目从：

```text
SFT + DPO
```

升级成：

```text
SFT + DPO + rule-reward / RL-style post-training analysis
```

面向后训练 / RLHF / 强化学习算法岗，重点展示：

```text
reward design
group-relative advantage
reward hacking 分析
PPO / GRPO 训练接入计划
KL 与格式稳定性约束
DPO 与 RL-style optimization 的 tradeoff
```

## 当前依赖限制

当前项目依赖固定为：

```text
trl==0.9.6
```

这版 TRL 适合 DPO / PPO，但没有稳定的 GRPOTrainer。因此当前仓库先实现：

```text
1. rule reward function
2. offline reward analysis
3. group-relative advantage diagnostic
4. PPO/GRPO 下一步训练计划
```

如果后续要正式跑 GRPO，建议新建实验分支并升级 TRL，而不是直接改主线依赖。

## Rule Reward 设计

已实现：

```text
src/finground_qa/reward.py
```

reward 由以下部分组成：

正向项：

```text
json_valid
schema_pass
exact_match
numeric_exact_match
faithfulness_proxy
citation_precision
citation_consistency_score
```

惩罚项：

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

reward 被归一化到 `[-1, 1]`，方便 PPO/GRPO 类算法使用。它是透明规则 reward，不是 learned reward model。

## Offline Reward Analysis

运行命令：

```powershell
.\scripts\run_rl_offline_reward_analysis.ps1
```

或在 Linux/Git Bash 上：

```bash
bash scripts/run_rl_offline_reward_analysis.sh
```

输出：

```text
reports/rl_offline_reward_report.json
results/rl_offline_reward_scored.jsonl
```

`results/rl_offline_reward_scored.jsonl` 为每个 prompt 的多个模型回答计算：

```text
reward
raw_reward
features
group_reward_mean
group_reward_std
group_relative_advantage
```

其中 `group_relative_advantage` 可以视为 GRPO-style diagnostic：同一个 prompt 下，不同策略输出之间的相对优势。

## 当前 Offline Reward 结果

| model | reward_mean | raw_reward_mean | exact | citation_precision | wrong_citation | calculation_error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | -0.2995 | -1.7903 | 0.1975 | 0.2487 | 0.7600 | 0.6300 |
| SFT | 0.3237 | 2.0261 | 0.3300 | 0.8950 | 0.0825 | 0.5750 |
| DPO v1 s500 | 0.3088 | 1.9330 | 0.2975 | 0.9450 | 0.0125 | 0.5925 |
| DPO v2 s50 | **0.3387** | **2.1250** | 0.3275 | **0.9575** | 0.0300 | **0.5725** |
| DPO v2 s100 | 0.3276 | 2.0516 | 0.3175 | 0.9525 | 0.0350 | 0.5775 |
| DPO v3 s100 | 0.3343 | 2.0959 | 0.3275 | 0.9450 | 0.0300 | 0.5750 |

结论：

```text
DPO v2 s50 在 rule reward 上最高，和当前最佳 DPO candidate 结论一致。
SFT 的 EM 最高，但 citation reward 弱于 DPO v2。
DPO v2 s50 与 SFT 在 400 个 prompt 上 tie_rate 为 88.25%，说明二者多数样本 reward 接近。
DPO v2 s50 相对 SFT 的 reward win_rate 为 7.75%，SFT 相对 DPO v2 s50 的 win_rate 为 4.00%。
```

这说明 RL-style 继续优化时要非常谨慎：如果 reward 权重过度偏向 citation，可能复制 v4 的 citation grounding 退化或 EM 下降问题。

## 下一步 PPO / GRPO 训练方案

### 方案 A：PPO small-scale

适合当前 `trl==0.9.6`。

```text
policy_init: SFT adapter 或 DPO v2 s50 adapter
prompts: train split 中 300-800 条高质量 prompts
reward: grounded_reward
KL reference: SFT policy
max_new_tokens: 512
batch: 小 batch + gradient accumulation
目标: reward 提升，同时 EM/citation 不退化
```

需要重点监控：

```text
KL divergence
format_error_rate
schema_pass_rate
wrong_citation_rate
numeric_exact_match
over_refusal_rate
forced_answer_rate
```

### 方案 B：GRPO branch

需要升级 TRL 后单独开分支。

```text
每个 prompt 采样 K=4 responses
对 K 个 responses 计算 rule reward
组内标准化得到 advantage
用 GRPO 更新 policy
保留 KL / format / citation gate
```

GRPO 更贴近当前 offline analysis，因为我们已经有 group-relative advantage 的诊断产物。

## Acceptance Gate

RL-style 实验不能只看 reward 变高。必须满足：

```text
schema_pass_rate >= 0.99
exact_match >= SFT - 0.005
numeric_exact_match >= SFT
citation_precision >= DPO v2 s50 - 0.02
wrong_citation_rate <= DPO v2 s50 + 0.02
forced_answer_rate 不恶化
over_refusal_rate 不恶化
```

如果 reward 上升但这些 gate 失败，应判定为 reward hacking。

## 面试讲法

```text
在 SFT/DPO 之外，我进一步设计了 rule-based reward，把 JSON 格式、答案正确性、citation consistency、wrong citation、fabricated number、over-refusal 和 forced-answer 都纳入 reward。然后我对 Base/SFT/DPO 多个模型输出做 offline reward analysis，并计算同一 prompt 下的 group-relative advantage，作为 GRPO-style 诊断。结果显示 DPO v2 s50 的 rule reward 最高，和模型选择结论一致，但 SFT 与 DPO v2 大量样本 reward tie，也说明如果继续做 PPO/GRPO，必须控制 KL 和 reward hacking，不能只让 citation reward 变高。
```
