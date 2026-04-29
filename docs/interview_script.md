# 面试讲稿

## 一分钟版本

这个项目不是金融专家模型，也不是完整 RAG 系统。我把金融报告问答作为 grounded generation 的实验场景，研究当 evidence contexts 已经给定时，SFT 和 DPO 会如何改变模型的答案正确性、引用质量、数值 grounding 和拒答行为。

核心工作不是只跑训练脚本，而是数据质量、偏好数据质量和评测闭环：

```text
1. 将 TAT-QA 和 FinQA 转成统一 question-context-evidence schema。
2. 做 evidence quote hit、数值 grounding、table linearization 和 train/eval leakage 检查。
3. 训练 SFT，让模型遵守 grounded JSON answer protocol。
4. 挖掘 SFT badcase，结合 hard negatives 构建 DPO preference pairs。
5. 审计 preference pairs，控制 length bias、format bias 和弱 rejected 样本。
6. 对 Base -> SFT -> 多版本 DPO 做 citation / numeric / answerability 细粒度评测。
7. 用 checkpoint sweep 和 badcase audit 做最终模型选择。
```

## 主结果

```text
Base EM: 19.75%
SFT EM: 33.00%
DPO v2 checkpoint-50 EM: 32.75%
DPO v2 checkpoint-50 numeric EM: 17.63%
DPO v2 checkpoint-50 citation precision: 95.75%
DPO v2 checkpoint-50 wrong-citation rate: 3.00%
```

最重要的点是：我没有把 DPO 包装成万能提升。SFT 是最稳的正式 baseline；DPO v2 checkpoint-50 是最佳 DPO candidate，因为它在 EM 接近 SFT 的同时明显提升 citation quality。

## Tradeoff 讲法

DPO 改善了 citation 行为，但也暴露出副作用。v3/v4 的 badcase 和指标显示，偏好优化可能引入 numeric scale regression、unanswerable forced-answer 和 citation grounding 退化。

典型数值尺度错误：

```text
-4.1 -> -0.041
18.75 -> 18750000.0
345.0 -> 0.345
```

所以我的下一步不是简单加 DPO steps，而是做 targeted guardrail preference data。v4 实验进一步验证：targeted pair 设计不稳时，虽然局部 audit 能通过，但 held-out eval 上 citation precision 会明显下降。这是一个有价值的负结果。

## v4 怎么讲

```text
v4 的目标是修 DPO 的副作用，所以我构建了 680 条 targeted pairs，包括 numeric scale guard、protect-correct guard、unanswerable refusal guard 和 citation repair guard。训练后做了 50/75/100 step sweep。

结果显示 v4 没有超过 v2。最好的 v4 s50 EM 是 32.25%，但 citation precision 只有 67.50%，wrong-citation rate 到 30.00%，明显差于 v2 s50 的 95.75% 和 3.00%。所以我没有晋升 v4，而是把它作为负结果消融，说明偏好数据设计和分布匹配比盲目增加训练更重要。
```

## 这个项目体现什么能力

```text
1. 能构建完整 LLM post-training 实验 pipeline。
2. 不是只看 training loss，而是关注 data quality 和 preference quality。
3. 能把 citation、numeric、answerability 拆开评测。
4. 能用 badcase analysis 指导下一轮数据设计。
5. 能接受负结果，并据此做模型选择，而不是过度包装。
```

## 被问到“最终选哪个模型”

推荐回答：

```text
如果作为正式 baseline，我选 SFT，因为它整体最稳，EM 和 faithfulness 最好。

如果作为 DPO candidate，我选 DPO v2 checkpoint-50，因为它在 EM 基本接近 SFT 的情况下显著改善 citation precision 和 wrong-citation rate。

我不会选 v4，因为 v4 虽然是 targeted DPO，但 held-out eval 显示 citation grounding 明显退化。
```

## 被问到“为什么 v4 失败”

推荐回答：

```text
我认为主要是 preference pair 的局部分布和主 eval 行为不完全匹配，尤其 citation-repair pair 可能强化了“输出引用形式”，但没有稳定强化 evidence consistency。DPO 对小规模偏好数据很敏感，680 pairs 训练到 100 step 已经约 2.35 epoch，继续训练会放大偏差。因此我把 v4 写成负结果，而不是继续盲目加 step。
```
