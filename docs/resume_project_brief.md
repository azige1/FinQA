# 简历项目摘要

## 项目标题

```text
金融报告 Grounded QA 后训练：SFT、DPO 与错误类型评测
```

## 一句话介绍

基于 Qwen2.5-7B-Instruct 构建金融报告问答后训练 pipeline，覆盖 TAT-QA / FinQA 统一 evidence schema、QLoRA SFT、model-mined DPO preference 构造、多版本 DPO checkpoint selection，以及 citation / numeric / answerability 细粒度评测和 badcase audit。

## 推荐简历描述

```text
金融报告 Grounded QA 后训练项目：基于 Qwen2.5-7B-Instruct 构建 TAT-QA/FinQA 统一 evidence schema，完成 QLoRA SFT、model-mined DPO preference 构造、多版本 DPO checkpoint selection 与 badcase audit。SFT 在 400 条 held-out eval 上将 EM 从 19.75% 提升至 33.00%；DPO v2 checkpoint-50 在保持 EM 接近 SFT 的同时，将 citation precision 提升至 95.75%，wrong-citation rate 降至 3.00%。进一步通过 post-hoc flip analysis 发现 DPO 的 numeric scale regression、unanswerable forced-answer 与 citation regression 风险，并设计 targeted v4 guardrail preference data 进行负结果消融。
```

## 强简历 Bullet

```text
- 构建金融报告 grounded QA 后训练 pipeline，将 TAT-QA / FinQA 样本统一为 question-context-evidence schema，并实现 quote-level citation validation、数值 grounding proxy 与 train/eval leakage check。

- 使用 Qwen2.5-7B-Instruct 完成 QLoRA SFT 和多版本 DPO 训练，偏好数据包含 model-mined rejected answers、rule-generated hard negatives、answerability-aware pairs 与 targeted guardrail pairs。

- 设计细粒度评测体系，覆盖 exact match、numeric exact match、citation precision、citation consistency、wrong citation、fabricated number、calculation error、over-refusal 与 forced-answer 行为。

- 在 400 条 held-out eval 上，SFT 将 Base EM 从 19.75% 提升至 33.00%；DPO v2 checkpoint-50 达到 32.75% EM、17.63% numeric EM、95.75% citation precision，并将 wrong-citation rate 降至 3.00%。

- 通过 DPO v3/v4 badcase 与 checkpoint sweep 发现偏好优化的副作用：DPO 能强化 citation 行为，但可能引入 numeric scale regression、answerability regression 和 citation grounding 退化；最终选择 DPO v2 checkpoint-50 作为最佳 DPO candidate。
```

## 面试短版

```text
这个项目不是金融大模型，也不是完整 RAG。我把金融报告问答作为 grounded generation 的受控场景，研究给定 evidence contexts 时，SFT 和 DPO 会如何改变模型行为。

最强结果不是“DPO 全面赢”，而是完整实验闭环：SFT 把 Base EM 从 19.75% 提升到 33.00%；DPO v2 checkpoint-50 在 EM 接近 SFT 的同时，把 citation precision 提升到 95.75%，wrong-citation rate 降到 3.00%。同时我做了 v3/v4 的负结果分析，发现偏好数据设计不稳会损伤数值尺度、answerability 和引用可靠性，所以最终没有盲目选择更长 step 或更新版本，而是选择 DPO v2 s50。
```

## 需要记住的数字

```text
Main eval size: 400
SFT train size: 5000
Base EM: 19.75%
SFT EM: 33.00%
DPO v2 checkpoint-50 EM: 32.75%
DPO v2 checkpoint-50 numeric EM: 17.63%
SFT citation precision: 89.50%
DPO v2 checkpoint-50 citation precision: 95.75%
SFT wrong-citation rate: 8.25%
DPO v2 checkpoint-50 wrong-citation rate: 3.00%
DPO v4 best EM: 32.25%
DPO v4 best citation precision: 67.50%
DPO v4 best wrong-citation rate: 30.00%
```

## 面试重点

```text
1. 重点讲后训练实验闭环，不要只讲“跑了 SFT/DPO”。
2. 重点讲数据质量和 preference quality，包括 leakage、quote hit、pair audit。
3. 重点讲 DPO tradeoff：citation 变好不代表整体无副作用。
4. 重点讲 checkpoint selection：v2 s50 比 s100 更稳。
5. 重点讲负结果：v4 targeted DPO 没成功，但这说明你会用实验否定假设。
```

## 不要这样声称

```text
- 不要说这是金融 foundation model。
- 不要说这是完整生产级 RAG 系统。
- 不要说 DPO 全面超过 SFT。
- 不要说 DPO v4 成功。
- 不要说 proxy faithfulness metric 等同于人工事实性判断。
- 不要说 FinanceBench 被用于训练。
```
