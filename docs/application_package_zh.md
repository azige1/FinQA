# FinGround-QA 投递材料包

## 1. GitHub 仓库设置

仓库地址：

```text
https://github.com/azige1/FinQA
```

建议 Description：

```text
Financial report grounded QA post-training with QLoRA SFT, DPO preference learning, citation/numeric evaluation, and badcase analysis.
```

建议 Topics：

```text
llm
sft
dpo
qlora
financial-qa
grounded-generation
evaluation
post-training
preference-learning
```

## 2. 简历项目经历

项目名：

```text
金融报告 Grounded QA 后训练：SFT、DPO 与错误类型评测
```

项目描述：

```text
基于 Qwen2.5-7B-Instruct 构建金融报告问答后训练 pipeline，覆盖 TAT-QA / FinQA 统一 evidence schema、QLoRA SFT、model-mined DPO preference 构造、多版本 DPO checkpoint selection，以及 citation / numeric / answerability 细粒度评测和 badcase audit。
```

三条 Bullet：

```text
- 构建金融报告 grounded QA 后训练 pipeline，将 TAT-QA / FinQA 转换为统一 question-context-evidence schema，并实现 evidence quote hit、numeric grounding proxy、table linearization 与 train/eval leakage check。

- 基于 Qwen2.5-7B-Instruct 完成 QLoRA SFT 与多版本 DPO 训练，构造 model-mined rejected answers、rule-generated hard negatives、answerability-aware pairs 和 targeted guardrail pairs，并进行 preference pair audit 与 checkpoint sweep。

- 在 400 条 held-out eval 上，SFT 将 Base EM 从 19.75% 提升至 33.00%；DPO v2 checkpoint-50 在保持 EM 接近 SFT 的同时达到 95.75% citation precision、3.00% wrong-citation rate，并通过 v3/v4 badcase analysis 揭示 DPO 的 numeric scale、answerability 与 citation regression 风险。
```

更短版本：

```text
- 构建 TAT-QA / FinQA 金融报告 grounded QA 后训练数据链路，统一 evidence schema，并完成 quote-level citation、numeric grounding 与 train/eval leakage 检查。

- 使用 Qwen2.5-7B-Instruct 进行 QLoRA SFT 和多版本 DPO preference learning，设计 model-mined hard negatives、answerability-aware pairs 与 targeted guardrail pairs。

- SFT 将 held-out EM 从 19.75% 提升至 33.00%；DPO v2 checkpoint-50 达到 95.75% citation precision 和 3.00% wrong-citation rate，并通过 v4 负结果分析验证 DPO 偏好数据的 tradeoff。
```

## 3. 两分钟面试自述

```text
这个项目是一个金融报告问答场景下的 grounded generation 后训练项目，不是完整金融大模型，也不是生产级 RAG。我关注的问题是：当 evidence contexts 已经给定时，SFT 和 DPO 会如何影响模型的答案正确性、引用质量、数值 grounding 和拒答行为。

我首先把 TAT-QA 和 FinQA 转成统一的 question-context-evidence schema，并做了 quote hit、numeric grounding、table linearization 和 train/eval leakage 检查。然后基于 Qwen2.5-7B-Instruct 做 QLoRA SFT，让模型遵守 grounded JSON answer protocol。

在 SFT 之后，我挖掘 SFT badcase，并结合 rule-generated hard negatives、answerability-aware pairs 构造 DPO preference data。评测不是只看 EM，而是拆成 exact match、numeric exact match、citation precision、wrong citation、fabricated number、calculation error、over-refusal 和 forced-answer 等指标。

结果上，SFT 把 Base EM 从 19.75% 提升到 33.00%。DPO v2 checkpoint-50 在 EM 基本接近 SFT 的情况下，把 citation precision 提升到 95.75%，wrong-citation rate 降到 3.00%。但我没有把 DPO 包装成全面胜利，因为 v3/v4 的 badcase 和指标显示，偏好优化会带来 numeric scale、answerability 和 citation regression 风险。

最终我选择 SFT 作为正式 baseline，DPO v2 checkpoint-50 作为最佳 DPO candidate，v4 targeted DPO 作为负结果消融。这个项目主要体现的是完整 post-training 实验闭环、preference data 质量控制、细粒度评测和负结果分析能力。
```

## 4. 三十秒版本

```text
我做了一个金融报告 grounded QA 后训练项目。核心不是做金融大模型，而是研究给定 evidence contexts 时 SFT 和 DPO 怎么改变模型行为。我统一了 TAT-QA/FinQA schema，做了 QLoRA SFT、多版本 DPO、checkpoint selection 和 citation/numeric/answerability 评测。SFT 把 Base EM 从 19.75% 提升到 33.00%；DPO v2 s50 在 EM 接近 SFT 的同时把 citation precision 提升到 95.75%，wrong-citation rate 降到 3.00%。同时 v4 是负结果，说明 DPO 偏好数据设计不稳会损伤 citation grounding。
```

## 5. 面试高频问答

### Q1: 这个项目和 RAG 有什么区别？

```text
这个项目不做 retrieval，也不解决 PDF/OCR。它假设 evidence contexts 已经给定，重点研究后训练如何影响 grounded generation 行为。换句话说，它是 post-training 和 evaluation 项目，不是端到端 RAG 系统。
```

### Q2: 为什么最终不是直接选 DPO？

```text
因为 DPO 不是单向提升。DPO v2 确实改善 citation precision 和 wrong-citation rate，但 SFT 在 EM、faithfulness 和 fabricated-number 指标上更稳。所以我把 SFT 作为正式 baseline，把 DPO v2 s50 作为最佳 DPO candidate，而不是声称 DPO 全面超过 SFT。
```

### Q3: v4 为什么失败？

```text
v4 用 680 条 targeted guardrail pairs 试图修复 numeric scale、unanswerable refusal 和 citation repair 问题，但 held-out eval 显示 citation precision 明显退化。我认为原因是 targeted pair 的局部分布和主 eval 分布不完全匹配，citation-repair pair 可能强化了引用格式，但没有稳定强化 evidence consistency。这个负结果说明 DPO preference data 需要非常谨慎地做分布和质量控制。
```

### Q4: 你最想强调的技术点是什么？

```text
我最想强调的是实验闭环：数据质量检查、SFT、DPO preference 构造、pair audit、checkpoint selection、细粒度评测和 badcase analysis。项目不是单纯跑脚本，而是用评测和负结果反推数据设计。
```

### Q5: 指标可信度怎么样？

```text
faithfulness 和 citation consistency 是规则与弱语义 proxy，不等价于人工事实性判断。所以我在报告里明确写了局限性，并结合 badcase audit 做结论，不只依赖单一自动指标。
```

## 6. 投递岗位关键词

```text
大模型算法实习
LLM post-training
SFT / DPO
RLHF / RLAIF
偏好学习
模型评测
金融 NLP
grounded generation
hallucination evaluation
```

## 7. 投递前检查

```text
GitHub README 首页能看到核心结果和图表。
简历里写 post-training / preference learning，不写金融大模型。
面试能讲清楚为什么 SFT 是 baseline，为什么 DPO v2 s50 是 candidate。
能主动解释 v4 负结果，而不是回避失败实验。
能说明 proxy metric 的局限性。
```

## 8. 后训练/RL 方向补充材料

简历可用 Bullet：

```text
- 构建金融报告 grounded QA 后训练 pipeline，覆盖 QLoRA SFT、DPO preference learning、rule-reward 设计和 offline group-relative advantage 分析，形成 SFT -> DPO -> RL-style post-training 的实验闭环。

- 设计 grounded QA rule reward，将 JSON/schema、exact match、numeric match、citation consistency、wrong citation、fabricated number、over-refusal 与 forced-answer 纳入 reward，并对 Base/SFT/DPO 多策略输出计算 GRPO-style group-relative advantage。

- 在 400 条 held-out eval 上，SFT 将 Base EM 从 19.75% 提升至 33.00%；DPO v2 checkpoint-50 达到最高 rule reward mean 0.3387、95.75% citation precision 和 3.00% wrong-citation rate，并通过 v4/RL reward analysis 分析 reward hacking 与 DPO tradeoff。
```

面试补充说法：

```text
在 DPO 之后，我还做了 RL-style extension。我设计了一个透明的 grounded QA rule reward，把 JSON/schema、EM、numeric EM、citation consistency、wrong citation、fabricated number、over-refusal 和 forced-answer 都纳入 reward。然后对 Base/SFT/DPO 多个策略在同一批 held-out prompts 上的输出做 offline reward analysis，计算 group-relative advantage，作为 GRPO-style 诊断。结果 DPO v2 s50 的 reward mean 最高，但它和 SFT 在 88.25% prompts 上 reward tie，所以后续 PPO/GRPO 必须控制 KL 和 reward hacking，而不能只让 citation reward 上升。
```
