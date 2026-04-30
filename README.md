# FinGround-QA

金融报告 Grounded QA 后训练项目，基于 Qwen2.5-7B-Instruct 研究 SFT 和 DPO 在“问题 + 证据上下文已给定”场景下对答案正确性、数值稳定性、引用可靠性和拒答行为的影响。

> 项目定位：LLM post-training + preference learning + grounded generation evaluation。  
> 不是金融大模型、完整 RAG 系统、PDF/OCR 解析系统或金融投资建议系统。

## 核心结果

主评测集为 400 条 held-out samples。错误类指标越低越好，其余指标越高越好。

| model | EM | Numeric EM | Faithfulness | Citation Precision | Wrong Citation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 0.1975 | 0.0504 | 0.1675 | 0.2487 | 0.7600 |
| SFT | **0.3300** | 0.1727 | **0.4200** | 0.8950 | 0.0825 |
| DPO v2 s50 | 0.3275 | **0.1763** | 0.4175 | **0.9575** | **0.0300** |
| DPO v3 s100 | 0.3275 | 0.1727 | 0.4125 | 0.9450 | **0.0300** |
| DPO v4 s50 | 0.3225 | 0.1655 | 0.3425 | 0.6750 | 0.3000 |

![Final metrics comparison](reports/final_metrics_comparison.png)

最终选择：

```text
正式 baseline: SFT
最佳 DPO candidate: DPO v2 checkpoint-50
DPO v3: 不晋升，只作为 guarded DPO 消融
DPO v4: 不晋升，只作为 targeted DPO 负结果 / 消融
```

核心结论：

```text
SFT 将 Base EM 从 19.75% 提升到 33.00%，是最稳的正式 baseline。
DPO v2 s50 在 EM 接近 SFT 的同时，将 citation precision 提升到 95.75%，wrong-citation rate 降到 3.00%。
DPO 不是免费提升：v3/v4 实验显示，偏好数据设计不稳会带来数值尺度、answerability 和 citation regression。
v4 targeted DPO 没有打过 v2，尤其 citation precision 从 95.75% 掉到 67.50%，wrong-citation rate 升到 30.00%。
```

## 项目亮点

```text
1. 统一 TAT-QA / FinQA question-context-evidence schema。
2. 构建 grounded JSON answer protocol，便于同时评测答案与引用。
3. 实现 quote hit、numeric grounding、table linearization、train/eval leakage 等数据质量检查。
4. 完成 QLoRA SFT、多版本 DPO、checkpoint sweep 和 held-out eval。
5. 设计 citation / numeric / answerability 细粒度指标，不只看 EM。
6. 通过 v4 负结果分析展示 DPO preference data 的 tradeoff。
```

## 方法链路

```text
TAT-QA / FinQA 数据标准化
-> 统一 question-context-evidence schema
-> 证据引用与泄漏检查
-> grounded JSON 格式 SFT 数据
-> Base Qwen2.5-7B-Instruct 评测
-> QLoRA SFT
-> SFT 输出挖掘 rejected answers
-> Error-Type Balanced DPO
-> 多版本 DPO checkpoint sweep
-> citation / numeric / answerability 细粒度评测
-> badcase audit 与模型选择
```

## 数据快照

```text
SFT train: 5000
SFT val: 500
Main eval: 400
FinanceBench audit: 150
Unified rows: 24283
Train/eval exact question overlap: 0
Evidence quote hit rate: 90.6%
```

## 完整实验表

| model | exact_match | numeric_exact_match | faithfulness_rate | citation_precision | citation_consistency_score | wrong_citation_rate | fabricated_number_rate | calculation_error_rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 0.1975 | 0.0504 | 0.1675 | 0.2487 | 0.2202 | 0.7600 | 0.6000 | 0.6300 |
| SFT | 0.3300 | 0.1727 | 0.4200 | 0.8950 | 0.6331 | 0.0825 | 0.5350 | 0.5750 |
| DPO v1 s500 | 0.2975 | 0.1475 | 0.4000 | 0.9450 | 0.6225 | 0.0125 | 0.5625 | 0.5925 |
| DPO v2 s50 | 0.3275 | 0.1763 | 0.4175 | 0.9575 | 0.6425 | 0.0300 | 0.5525 | 0.5725 |
| DPO v2 s100 | 0.3175 | 0.1691 | 0.4075 | 0.9525 | 0.6356 | 0.0350 | 0.5600 | 0.5775 |
| DPO v3 guarded s100 | 0.3275 | 0.1727 | 0.4125 | 0.9450 | 0.6369 | 0.0300 | 0.5500 | 0.5750 |
| DPO v4 targeted s50 | 0.3225 | 0.1655 | 0.3425 | 0.6750 | 0.5681 | 0.3000 | 0.5550 | 0.5800 |
| DPO v4 targeted s75 | 0.3175 | 0.1583 | 0.3375 | 0.6400 | 0.5600 | 0.3275 | 0.5550 | 0.5850 |
| DPO v4 targeted s100 | 0.3200 | 0.1655 | 0.3400 | 0.6275 | 0.5581 | 0.3400 | 0.5475 | 0.5800 |

## 关键文件

```text
reports/final_experiment_report_zh.md        中文最终实验报告
reports/dpo_model_selection_report.md        中文模型选择报告
docs/project_showcase_zh.md                  GitHub/面试展示稿
docs/final_resume_bullets_zh.md              简历 bullet
docs/application_package_zh.md               投递材料包
docs/interview_script.md                     中文面试讲稿
docs/interview_flashcards_zh.md              面试速记卡
reports/final_metrics_comparison.png         核心结果图
scripts/build_dpo_targeted_v4.py             v4 targeted pair 构建脚本
scripts/run_dpo_v4_targeted_sweep_a10040.sh  v4 sweep 运行脚本
```

## 本地数据构建

```bash
python -m src.finground_qa.pipeline prepare-data --output-dir .
python -m src.finground_qa.pipeline validate-sft \
  --file data/sft/sft_train.jsonl \
  --output reports/validate_sft_train.json
python -m src.finground_qa.pipeline build-rule-dpo \
  --unified-train data/unified/train_unified.jsonl \
  --target 600 \
  --output data/dpo/rule_dpo_pairs.jsonl \
  --report reports/preference_pair_quality_report.json
```

## 主要指标

```text
exact_match
numeric_exact_match
faithfulness_rate
unsupported_claim_rate
citation_precision
citation_consistency_score
wrong_citation_rate
fabricated_number_rate
calculation_error_rate
over_refusal_rate
forced_answer_rate
schema_pass_rate
```

## 局限性

`faithfulness_rate`、`unsupported_claim_rate` 和 `citation_consistency_score` 是规则与弱语义 proxy，不等价于完整事实性评测。项目结论需要结合人工 badcase audit 解读，不能只看单一自动指标。

FinanceBench 只作为 external audit sample，不用于训练。
