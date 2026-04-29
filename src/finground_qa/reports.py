"""Report generation for unified data and preference pairs."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List

from .schema import validate_pair, validate_unified
from .text_utils import extract_numbers, normalize_for_match, token_overlap


def summarize_unified(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    invalid = []
    context_counts = []
    evidence_counts = []
    for row in rows:
        errs = validate_unified(row)
        if errs:
            invalid.append({"id": row.get("id"), "errors": errs})
        context_counts.append(len(row.get("contexts") or []))
        evidence_counts.append(len(row.get("evidence") or []))
    return {
        "rows": len(rows),
        "invalid_rows": len(invalid),
        "sample_invalid": invalid[:20],
        "dataset_counts": dict(Counter(row.get("dataset") for row in rows)),
        "source_split_counts": dict(Counter(row.get("source_split") for row in rows)),
        "context_count": distribution(context_counts),
        "evidence_count": distribution(evidence_counts),
    }


def distribution(values: List[int | float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0, "min": 0, "max": 0}
    sorted_values = sorted(values)
    return {
        "mean": sum(values) / len(values),
        "min": sorted_values[0],
        "p50": sorted_values[len(values) // 2],
        "max": sorted_values[-1],
    }


def evidence_quote_hit_report(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_quotes = 0
    quote_hits = 0
    empty_evidence = 0
    for row in rows:
        contexts = {str(ctx.get("chunk_id")): normalize_for_match(ctx.get("text", "")) for ctx in row.get("contexts", [])}
        evidence = row.get("evidence") or []
        if not evidence:
            empty_evidence += 1
        for ev in evidence:
            total_quotes += 1
            quote = normalize_for_match(ev.get("quote", ""))
            chunk_text = contexts.get(str(ev.get("chunk_id")), "")
            if quote and quote in chunk_text:
                quote_hits += 1
    return {
        "rows": len(rows),
        "empty_evidence_rows": empty_evidence,
        "total_quotes": total_quotes,
        "quote_hits": quote_hits,
        "quote_hit_rate": quote_hits / max(1, total_quotes),
    }


def numeric_grounding_report(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    numeric_rows = 0
    number_hit_rows = 0
    for row in rows:
        answer_nums = set(extract_numbers(row.get("answer", "")))
        if not answer_nums:
            continue
        numeric_rows += 1
        contexts_text = " ".join(str(ctx.get("text", "")) for ctx in row.get("contexts", []))
        ctx_nums = set(extract_numbers(contexts_text))
        if answer_nums & ctx_nums:
            number_hit_rows += 1
    return {
        "rows": len(rows),
        "numeric_rows": numeric_rows,
        "number_hit_rows": number_hit_rows,
        "number_hit_rate": number_hit_rows / max(1, numeric_rows),
    }


def table_linearization_report(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    table_rows = [row for row in rows if any(ctx.get("type") == "table" for ctx in row.get("contexts", []))]
    tables_with_ids = 0
    for row in table_rows:
        if any(ctx.get("table_id") for ctx in row.get("contexts", []) if ctx.get("type") == "table"):
            tables_with_ids += 1
    return {
        "rows": len(rows),
        "rows_with_table_context": len(table_rows),
        "table_context_rate": len(table_rows) / max(1, len(rows)),
        "tables_with_table_id": tables_with_ids,
    }


def categorical_report(rows: List[Dict[str, Any]], field: str) -> Dict[str, Any]:
    return {"rows": len(rows), field: dict(Counter(row.get(field) for row in rows))}


def leakage_report(train_rows: List[Dict[str, Any]], eval_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    train_questions = {normalize_for_match(row.get("question", "")): row.get("id") for row in train_rows}
    exact = []
    for row in eval_rows:
        q = normalize_for_match(row.get("question", ""))
        if q in train_questions:
            exact.append({"eval_id": row.get("id"), "train_id": train_questions[q]})
    return {"train_rows": len(train_rows), "eval_rows": len(eval_rows), "exact_question_overlap": len(exact), "sample": exact[:20]}


def answerability_report(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts = Counter(row.get("grounding_type") for row in rows)
    return {
        "rows": len(rows),
        "grounding_type": dict(counts),
        "answerable_rows": len(rows) - counts.get("unanswerable", 0),
        "unanswerable_rows": counts.get("unanswerable", 0),
    }


def preference_pair_quality_report(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    invalid = []
    chosen_lens = []
    rejected_lens = []
    for row in rows:
        errs = validate_pair(row)
        if errs:
            invalid.append({"id": row.get("id"), "errors": errs})
        chosen_lens.append(int(row.get("chosen_length") or len(str(row.get("chosen", "")))))
        rejected_lens.append(int(row.get("rejected_length") or len(str(row.get("rejected", "")))))
    chosen_avg = sum(chosen_lens) / max(1, len(chosen_lens))
    rejected_avg = sum(rejected_lens) / max(1, len(rejected_lens))
    length_ratio = chosen_avg / max(1, rejected_avg)
    reject_type = dict(Counter(row.get("reject_type") for row in rows))
    source = dict(Counter(row.get("source") for row in rows))
    difficulty = dict(Counter(row.get("difficulty") for row in rows))
    answerability = dict(Counter(row.get("answerability_type", "unknown") for row in rows))
    return {
        "rows": len(rows),
        "invalid_rows": len(invalid),
        "sample_invalid": invalid[:20],
        "reject_type": reject_type,
        "source": source,
        "difficulty": difficulty,
        "answerability_type": answerability,
        "reject_type_distribution": reject_type,
        "source_distribution": source,
        "difficulty_distribution": difficulty,
        "answerability_distribution": answerability,
        "chosen_avg_len": chosen_avg,
        "rejected_avg_len": rejected_avg,
        "length_ratio": length_ratio,
        "length_bias_report": {
            "chosen_avg_len": chosen_avg,
            "rejected_avg_len": rejected_avg,
            "length_ratio": length_ratio,
            "length_ratio_ok": 0.5 <= length_ratio <= 2.0,
        },
    }
