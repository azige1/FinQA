"""Evaluation and error delta utilities."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

from .checker import check_output
from .io_utils import iter_jsonl
from .text_utils import answer_equivalent


BOOL_KEYS = [
    "json_valid",
    "schema_pass",
    "format_error",
    "schema_error",
    "missing_evidence",
    "wrong_citation",
    "unsupported_claim",
    "fabricated_number",
    "calculation_error",
    "over_refusal",
    "forced_answer",
    "generic_answer",
]

SUM_KEYS = {
    "faithfulness_sum": "faithfulness_proxy",
    "citation_precision_sum": "citation_precision",
    "citation_consistency_sum": "citation_consistency_score",
    "chunk_valid_sum": "chunk_valid",
    "quote_hit_sum": "quote_hit",
    "number_coverage_sum": "number_coverage",
    "entity_or_token_coverage_sum": "entity_or_token_coverage",
}


def evaluate_predictions(eval_rows: List[Dict[str, Any]], prediction_rows: List[Dict[str, Any]]) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    eval_by_id = {row["id"]: row for row in eval_rows}
    scored: List[Dict[str, Any]] = []
    overall = new_accumulator()
    groups: Dict[str, Dict[str, Dict[str, Any]]] = {
        "dataset": {},
        "grounding_type": {},
        "answerability_type": {},
    }
    for pred in prediction_rows:
        row_id = pred.get("id") or pred.get("row_id")
        if row_id not in eval_by_id:
            continue
        gold = eval_by_id[row_id]
        text = pred.get("prediction") or pred.get("output") or pred.get("response") or ""
        check = check_output(gold, text)
        answer_text = extract_answer_from_prediction(text)
        update_accumulator(overall, gold, check, answer_text)
        update_group(groups["dataset"], str(gold.get("dataset") or "unknown"), gold, check, answer_text)
        update_group(groups["grounding_type"], str(gold.get("grounding_type") or "unknown"), gold, check, answer_text)
        update_group(groups["answerability_type"], answerability_type(gold), gold, check, answer_text)
        scored_row = dict(pred)
        scored_row["eval"] = check
        scored.append(scored_row)
    metrics = finalize_accumulator(overall)
    metrics["stratified"] = {
        name: {key: finalize_accumulator(acc) for key, acc in sorted(bucket.items())}
        for name, bucket in groups.items()
    }
    return metrics, scored


def new_accumulator() -> Dict[str, Any]:
    return {
        "total": 0,
        "exact": 0,
        "numeric_total": 0,
        "numeric_exact": 0,
        "counters": Counter(),
    }


def update_group(
    groups: Dict[str, Dict[str, Any]],
    name: str,
    gold: Dict[str, Any],
    check: Dict[str, Any],
    answer_text: str,
) -> None:
    groups.setdefault(name, new_accumulator())
    update_accumulator(groups[name], gold, check, answer_text)


def update_accumulator(acc: Dict[str, Any], gold: Dict[str, Any], check: Dict[str, Any], answer_text: str) -> None:
    acc["total"] += 1
    gold_answer = gold.get("answer", gold.get("gold_answer", ""))
    if answer_equivalent(answer_text, gold_answer):
        acc["exact"] += 1
    if gold.get("answer_type") == "number":
        acc["numeric_total"] += 1
        if answer_equivalent(answer_text, gold_answer):
            acc["numeric_exact"] += 1
    counters = acc["counters"]
    for key in BOOL_KEYS:
        if check.get(key):
            counters[key] += 1
    for sum_key, check_key in SUM_KEYS.items():
        counters[sum_key] += check.get(check_key, 0.0)


def finalize_accumulator(acc: Dict[str, Any]) -> Dict[str, Any]:
    total = acc["total"]
    counters = acc["counters"]
    return {
        "num_samples": total,
        "exact_match": acc["exact"] / max(1, total),
        "numeric_exact_match": acc["numeric_exact"] / max(1, acc["numeric_total"]),
        "json_valid_rate": counters["json_valid"] / max(1, total),
        "schema_pass_rate": counters["schema_pass"] / max(1, total),
        "format_error_rate": counters["format_error"] / max(1, total),
        "schema_error_rate": counters["schema_error"] / max(1, total),
        "faithfulness_rate": counters["faithfulness_sum"] / max(1, total),
        "citation_precision": counters["citation_precision_sum"] / max(1, total),
        "citation_consistency_score": counters["citation_consistency_sum"] / max(1, total),
        "chunk_valid_rate": counters["chunk_valid_sum"] / max(1, total),
        "quote_hit_rate": counters["quote_hit_sum"] / max(1, total),
        "number_coverage_rate": counters["number_coverage_sum"] / max(1, total),
        "entity_or_token_coverage_rate": counters["entity_or_token_coverage_sum"] / max(1, total),
        "missing_evidence_rate": counters["missing_evidence"] / max(1, total),
        "wrong_citation_rate": counters["wrong_citation"] / max(1, total),
        "unsupported_claim_rate": counters["unsupported_claim"] / max(1, total),
        "fabricated_number_rate": counters["fabricated_number"] / max(1, total),
        "calculation_error_rate": counters["calculation_error"] / max(1, total),
        "over_refusal_rate": counters["over_refusal"] / max(1, total),
        "forced_answer_rate": counters["forced_answer"] / max(1, total),
        "generic_answer_rate": counters["generic_answer"] / max(1, total),
    }


def answerability_type(row: Dict[str, Any]) -> str:
    if row.get("answerability_type") in {"answerable", "unanswerable"}:
        return str(row["answerability_type"])
    if row.get("grounding_type") == "unanswerable" or row.get("answer_type") == "unanswerable":
        return "unanswerable"
    return "answerable"


def extract_answer_from_prediction(text: str) -> str:
    import json

    try:
        obj = json.loads(text.strip())
        if isinstance(obj, dict):
            return str(obj.get("answer", ""))
    except Exception:
        pass
    return text.strip()


def error_delta(metrics_by_name: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    keys = [
        "format_error_rate",
        "schema_error_rate",
        "missing_evidence_rate",
        "wrong_citation_rate",
        "unsupported_claim_rate",
        "fabricated_number_rate",
        "calculation_error_rate",
        "over_refusal_rate",
        "forced_answer_rate",
        "generic_answer_rate",
        "schema_pass_rate",
        "faithfulness_rate",
        "citation_precision",
        "citation_consistency_score",
        "chunk_valid_rate",
        "quote_hit_rate",
        "number_coverage_rate",
        "entity_or_token_coverage_rate",
        "exact_match",
        "numeric_exact_match",
    ]
    names = list(metrics_by_name)
    rows = []
    for key in keys:
        item = {"metric": key}
        for name in names:
            item[name] = metrics_by_name[name].get(key)
        if len(names) >= 2:
            for left, right in zip(names, names[1:]):
                item[f"{left}_to_{right}_delta"] = metrics_by_name[right].get(key, 0) - metrics_by_name[left].get(key, 0)
            item[f"{names[0]}_to_{names[-1]}_delta"] = metrics_by_name[names[-1]].get(key, 0) - metrics_by_name[names[0]].get(key, 0)
        rows.append(item)
    error_keys = [
        "format_error_rate",
        "schema_error_rate",
        "missing_evidence_rate",
        "wrong_citation_rate",
        "unsupported_claim_rate",
        "fabricated_number_rate",
        "calculation_error_rate",
        "over_refusal_rate",
        "forced_answer_rate",
        "generic_answer_rate",
    ]
    return {
        "models": names,
        "error_type_delta": [row for row in rows if row["metric"] in error_keys],
        "metric_delta": rows,
    }
