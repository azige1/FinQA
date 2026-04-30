# 面试速记卡

## 项目一句话

```text
金融报告 grounded QA 后训练项目，研究给定 evidence contexts 时，SFT 和 DPO 如何影响答案正确性、引用质量、数值 grounding 和拒答行为。
```

## 必背数字

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
v4 best Wrong Citation: 30.00%
```

## 最终选择

```text
正式 baseline: SFT
最佳 DPO candidate: DPO v2 checkpoint-50
v3: 不晋升
v4: 不晋升，作为负结果消融
```

## 为什么 SFT 是 baseline

```text
SFT 的 EM 和 faithfulness 最稳，整体风险最低。
```

## 为什么 DPO v2 s50 是最佳 DPO candidate

```text
EM 接近 SFT，numeric EM 略高，同时 citation precision 更高，wrong-citation rate 更低。
```

## 为什么不选 v4

```text
v4 targeted DPO 没有超过 v2，citation precision 从 95.75% 掉到 67.50%，wrong-citation rate 从 3.00% 升到 30.00%。
```

## 项目边界

```text
不是金融大模型。
不是完整 RAG。
不是 PDF/OCR。
不是金融投资建议。
是 LLM post-training + grounded generation evaluation。
```

## 技术关键词

```text
Qwen2.5-7B-Instruct
QLoRA
SFT
DPO
preference learning
model-mined rejected answers
answerability-aware hard negatives
checkpoint selection
citation consistency
numeric grounding
badcase audit
```

## 最容易被追问的点

```text
1. DPO 为什么不是全面提升？
2. v4 为什么失败？
3. citation precision 怎么评？
4. faithfulness proxy 的局限是什么？
5. 为什么不做 retrieval？
```

## 回答原则

```text
承认边界，不夸大。
先讲实验设计，再讲数字。
主动讲负结果。
强调数据质量和评测闭环。
```
