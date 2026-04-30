# 最终简历 Bullet

## 一行版

```text
金融报告 Grounded QA 后训练项目：基于 Qwen2.5-7B-Instruct 构建 TAT-QA/FinQA 统一 evidence schema，完成 QLoRA SFT、多版本 DPO preference learning、checkpoint selection 与 citation/numeric/answerability 细粒度评测；SFT 将 400 条 held-out eval EM 从 19.75% 提升至 33.00%，DPO v2 checkpoint-50 将 citation precision 提升至 95.75%、wrong-citation rate 降至 3.00%。
```

## 三 Bullet 版

```text
- 构建金融报告 grounded QA 后训练 pipeline，将 TAT-QA / FinQA 转换为统一 question-context-evidence schema，并实现 evidence quote hit、numeric grounding proxy、table linearization 与 train/eval leakage check。

- 基于 Qwen2.5-7B-Instruct 完成 QLoRA SFT 与多版本 DPO 训练，构造 model-mined rejected answers、rule-generated hard negatives、answerability-aware pairs 和 targeted guardrail pairs，并进行 preference pair audit 与 checkpoint sweep。

- 在 400 条 held-out eval 上，SFT 将 Base EM 从 19.75% 提升至 33.00%；DPO v2 checkpoint-50 在保持 EM 接近 SFT 的同时达到 95.75% citation precision、3.00% wrong-citation rate，并通过 v3/v4 badcase analysis 揭示 DPO 的 numeric scale、answerability 与 citation regression 风险。
```

## 精简三 Bullet 版

```text
- 构建 TAT-QA / FinQA 金融报告 grounded QA 后训练数据链路，统一 evidence schema，并完成 quote-level citation、numeric grounding 与 train/eval leakage 检查。

- 使用 Qwen2.5-7B-Instruct 进行 QLoRA SFT 和多版本 DPO preference learning，设计 model-mined hard negatives、answerability-aware pairs 与 targeted guardrail pairs。

- SFT 将 held-out EM 从 19.75% 提升至 33.00%；DPO v2 checkpoint-50 达到 95.75% citation precision 和 3.00% wrong-citation rate，并通过 v4 负结果分析验证 DPO 偏好数据的 tradeoff。
```

## 面试开场版

```text
这个项目的重点不是做金融大模型，而是用金融报告 QA 做一个 grounded generation 后训练实验。我把 TAT-QA 和 FinQA 统一成 evidence schema，先做 SFT，再用 SFT badcase 和规则 hard negatives 做 DPO。最后不是只看 EM，而是拆成 numeric、citation、answerability 指标做 checkpoint selection。结果是 SFT 最稳，DPO v2 s50 是最佳 DPO candidate；v4 targeted DPO 没有成功，我把它写成负结果消融。
```

## 数字速记

```text
Eval: 400
SFT train: 5000
Base EM: 19.75%
SFT EM: 33.00%
DPO v2 s50 EM: 32.75%
DPO v2 s50 Numeric EM: 17.63%
DPO v2 s50 Citation Precision: 95.75%
DPO v2 s50 Wrong Citation: 3.00%
v4 best EM: 32.25%
v4 best Citation Precision: 67.50%
```

## 简历风险控制

```text
可以说：后训练、偏好学习、评测、错误分析、负结果消融。
不要说：金融大模型、完整 RAG、生产系统、DPO 全面超过 SFT。
```

## 后训练/RL 方向强化版

```text
- 构建金融报告 grounded QA 后训练 pipeline，覆盖 QLoRA SFT、DPO preference learning、rule-reward 设计和 offline group-relative advantage 分析，形成 SFT -> DPO -> RL-style post-training 的实验闭环。

- 设计 grounded QA rule reward，将 JSON/schema、exact match、numeric match、citation consistency、wrong citation、fabricated number、over-refusal 与 forced-answer 纳入 reward，并对 Base/SFT/DPO 多策略输出计算 GRPO-style group-relative advantage。

- 在 400 条 held-out eval 上，SFT 将 Base EM 从 19.75% 提升至 33.00%；DPO v2 checkpoint-50 达到最高 rule reward mean 0.3387、95.75% citation precision 和 3.00% wrong-citation rate，并通过 v4/RL reward analysis 分析 reward hacking 与 DPO tradeoff。
```
