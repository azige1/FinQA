"""Dataset converters for TAT-QA, FinQA variants, and FinanceBench."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from datasets import load_dataset

from .text_utils import classify_grounding, find_quote, normalize_text, stringify_answer, table_to_markdown


def infer_reasoning_type(answer_type: str, derivation: str = "", req_comparison: bool = False) -> str:
    if req_comparison:
        return "comparison"
    if derivation.strip() or answer_type in {"number", "arithmetic"}:
        return "calculation"
    if answer_type in {"multi-span", "multi_span"}:
        return "multi_hop"
    return "lookup"


def normalize_answer_type(value: Any) -> str:
    value = normalize_text(value).lower().replace("-", "_")
    if value in {"multi_span", "multi_span_answer"}:
        return "multi_span"
    if value in {"span", "number", "yes_no", "unanswerable"}:
        return value
    if value in {"multi-span"}:
        return "multi_span"
    if value in {"count", "arithmetic"}:
        return "number"
    return "span"


def convert_tatqa(split: str, limit: int | None = None) -> List[Dict[str, Any]]:
    ds = load_dataset("next-tat/TAT-QA", split=split)
    rows: List[Dict[str, Any]] = []
    for doc_idx, doc in enumerate(ds):
        table = doc.get("table") or {}
        table_id = table.get("uid") or f"tatqa_{split}_{doc_idx}_table"
        table_md, cell_meta = table_to_markdown(table.get("table") or [], table_id)
        contexts: List[Dict[str, Any]] = []
        if table_md:
            contexts.append(
                {
                    "doc_id": f"tatqa_{split}_{doc_idx}",
                    "chunk_id": f"{table_id}_table",
                    "text": table_md,
                    "type": "table",
                    "table_id": table_id,
                }
            )
        for para in doc.get("paragraphs") or []:
            uid = para.get("uid") or f"p{para.get('order', len(contexts))}"
            contexts.append(
                {
                    "doc_id": f"tatqa_{split}_{doc_idx}",
                    "chunk_id": f"{uid}",
                    "text": normalize_text(para.get("text", "")),
                    "type": "text",
                }
            )
        for q in doc.get("questions") or []:
            answer = stringify_answer(q.get("answer"), q.get("scale"))
            answer_type = normalize_answer_type(q.get("answer_type"))
            reasoning_type = infer_reasoning_type(answer_type, q.get("derivation") or "", bool(q.get("req_comparison")))
            quote = find_quote(answer, contexts)
            evidence = [quote] if quote else []
            grounding_type = classify_grounding(answer, contexts, answer_type, q.get("derivation") or "")
            row = {
                "id": f"tatqa_{split}_{q.get('uid')}",
                "dataset": "TAT-QA",
                "source_split": split,
                "question": normalize_text(q.get("question", "")),
                "contexts": contexts,
                "answer": answer,
                "evidence": evidence,
                "answer_type": answer_type,
                "reasoning_type": reasoning_type,
                "grounding_type": grounding_type,
                "gold_program": q.get("derivation") or "",
                "meta": {
                    "answer_from": q.get("answer_from"),
                    "rel_paragraphs": q.get("rel_paragraphs"),
                    "scale": q.get("scale"),
                    "cell_meta": cell_meta[:50],
                },
            }
            rows.append(row)
            if limit and len(rows) >= limit:
                return rows
    return rows


def convert_finqa_updated(split: str, limit: int | None = None) -> List[Dict[str, Any]]:
    ds = load_dataset("gagan3012/finqa-updated", split=split)
    rows: List[Dict[str, Any]] = []
    for idx, item in enumerate(ds):
        raw_text = normalize_text(item.get("text") or item.get("query") or "")
        text = extract_finqa_context(raw_text)
        q = extract_finqa_question(item)
        answer = stringify_answer(item.get("answer"))
        ctx = {
            "doc_id": f"finqa_{split}_{idx}",
            "chunk_id": f"finqa_{split}_{idx}_ctx0",
            "text": text,
            "type": "text",
        }
        contexts = [ctx]
        quote = find_quote(answer, contexts)
        evidence = [quote] if quote else [{"chunk_id": ctx["chunk_id"], "quote": text[:300]}]
        row = {
            "id": f"finqa_{split}_{item.get('id') or idx}",
            "dataset": "FinQA",
            "source_split": split,
            "question": q or normalize_text(item.get("query", "")),
            "contexts": contexts,
            "answer": answer,
            "evidence": evidence,
            "answer_type": "number" if answer and any(ch.isdigit() for ch in answer) else "span",
            "reasoning_type": "calculation",
            "grounding_type": classify_grounding(answer, contexts, "number"),
            "gold_program": "",
            "meta": {"source_repo": "gagan3012/finqa-updated"},
        }
        rows.append(row)
        if limit and len(rows) >= limit:
            return rows
    return rows


def extract_finqa_question(item: Dict[str, Any]) -> str:
    text_field = normalize_text(item.get("text", ""))
    if text_field:
        for marker in ["Important information:", "Key Information:", "Context:"]:
            if marker in text_field:
                q = text_field.split(marker, 1)[0].strip()
                if q:
                    return q
        if len(text_field.split()) <= 40:
            return text_field
    query = normalize_text(item.get("query", ""))
    prefix = "Please answer the following financial question based on the context provided."
    if query.startswith(prefix):
        query = query[len(prefix) :].strip()
    if "Context:" in query:
        before = query.split("Context:", 1)[0].strip()
        if before:
            return before
    return query[:300]


def extract_finqa_context(text: str) -> str:
    """Keep only evidence context, removing answer/program leakage.

    The selected FinQA mirror stores question, context, reasoning steps, and
    program in one text field. For grounded generation training, the model
    should see only the financial context, not gold reasoning/program traces.
    """
    text = normalize_text(text)
    for marker in ["Important information:", "Key Information:", "Context:"]:
        if marker in text:
            text = text.split(marker, 1)[1].strip()
            break
    for marker in ["Reasoning Steps:", "Program:", "Program (Nested):", "Answer:"]:
        if marker in text:
            text = text.split(marker, 1)[0].strip()
    return text


def convert_financebench(limit: int | None = None) -> List[Dict[str, Any]]:
    ds = load_dataset("PatronusAI/financebench", split="train")
    rows: List[Dict[str, Any]] = []
    for idx, item in enumerate(ds):
        contexts: List[Dict[str, Any]] = []
        evidence = item.get("evidence") or []
        for eidx, ev in enumerate(evidence):
            text = normalize_text(ev.get("evidence_text") or ev.get("evidence_text_full_page") or "")
            contexts.append(
                {
                    "doc_id": normalize_text(item.get("doc_name") or f"financebench_{idx}"),
                    "chunk_id": f"financebench_{idx}_ev{eidx}",
                    "text": text,
                    "type": "text",
                }
            )
        answer = stringify_answer(item.get("answer"))
        quote = find_quote(answer, contexts)
        row = {
            "id": item.get("financebench_id") or f"financebench_{idx}",
            "dataset": "FinanceBench",
            "source_split": "audit",
            "question": normalize_text(item.get("question", "")),
            "contexts": contexts,
            "answer": answer,
            "evidence": [quote] if quote else [
                {"chunk_id": contexts[0]["chunk_id"], "quote": normalize_text(item.get("justification", ""))[:300]}
            ]
            if contexts
            else [],
            "answer_type": "number" if answer and any(ch.isdigit() for ch in answer) else "span",
            "reasoning_type": "calculation" if item.get("question_reasoning") != "Information extraction" else "lookup",
            "grounding_type": classify_grounding(answer, contexts, "number" if any(ch.isdigit() for ch in answer) else "span"),
            "gold_program": "",
            "meta": {
                "company": item.get("company"),
                "doc_name": item.get("doc_name"),
                "doc_link": item.get("doc_link"),
                "question_type": item.get("question_type"),
                "question_reasoning": item.get("question_reasoning"),
            },
        }
        rows.append(row)
        if limit and len(rows) >= limit:
            return rows
    return rows
