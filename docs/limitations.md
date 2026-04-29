# Limitations

```text
1. Automatic faithfulness is a proxy, not a complete factuality judge.
2. Citation Consistency Score is a proxy built from chunk validity, quote hit,
   number coverage, and weak token/entity coverage.
3. The project does not implement a full RAG retrieval system.
4. The project does not parse PDFs or perform OCR.
5. The project does not solve complex financial table reasoning.
6. The project does not claim financial professional ability.
7. FinanceBench is used for external audit only, not training.
8. DPO v1 uses 1000 high-confidence pairs before any larger mixed set.
```

The correct reporting position is:

```text
This is a grounded generation post-training experiment, not a financial QA
leaderboard or a finance-domain expert model.
```
