# Interview Script

This project is not a financial expert model or a RAG system. I use financial
report QA as a grounded generation case study to analyze how SFT and DPO change
model behavior when evidence contexts are provided.

The key work is data quality and preference quality:

```text
1. Convert TAT-QA and FinQA into a unified question-context-evidence schema.
2. Validate evidence quote hits, numeric grounding, table linearization, and leakage.
3. Train SFT to follow a grounded JSON answer protocol.
4. Mine SFT failure cases and combine them with hard negatives for DPO.
5. Audit preference pairs to control length bias, format bias, and weak rejected samples.
6. Evaluate Base -> SFT -> DPO with faithfulness/citation/over-refusal proxies and manual badcases.
```
