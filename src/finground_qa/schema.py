"""Schemas and validators for FinGround-QA."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

ANSWER_TYPES = {"span", "number", "yes_no", "unanswerable", "multi_span"}
REASONING_TYPES = {"lookup", "comparison", "calculation", "multi_hop"}
GROUNDING_TYPES = {"direct_grounded", "numeric_grounded", "calculation_hard", "unanswerable"}
CONFIDENCE_VALUES = {"high", "medium", "low"}
REJECT_TYPES = {
    "unsupported_claim",
    "wrong_citation",
    "fabricated_number",
    "calculation_error",
    "missing_evidence",
    "context_contradiction",
    "over_refusal",
    "forced_answer",
    "generic_answer",
    "wrong_format",
}
PAIR_SOURCES = {
    "model_mined",
    "rule_generated",
    "hard_negative",
    "numeric_corruption",
    "wrong_citation_corruption",
    "v3_numeric_precision_guard",
    "v3_unanswerable_guard",
    "v4_numeric_scale_guard",
    "v4_protect_correct_guard",
    "v4_unanswerable_refusal_guard",
    "v4_citation_repair_guard",
}
PAIR_DIFFICULTIES = {"easy", "medium", "hard"}
ANSWERABILITY_TYPES = {"answerable", "unanswerable", "unknown"}


def validate_unified(row: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    for key in ["id", "dataset", "question", "contexts", "answer", "evidence", "answer_type", "reasoning_type", "grounding_type"]:
        if key not in row:
            errors.append(f"missing:{key}")
    if not isinstance(row.get("contexts"), list) or not row.get("contexts"):
        errors.append("empty_contexts")
    if not isinstance(row.get("evidence"), list):
        errors.append("invalid_evidence")
    if row.get("answer_type") not in ANSWER_TYPES:
        errors.append("invalid_answer_type")
    if row.get("reasoning_type") not in REASONING_TYPES:
        errors.append("invalid_reasoning_type")
    if row.get("grounding_type") not in GROUNDING_TYPES:
        errors.append("invalid_grounding_type")
    return errors


def validate_sft_response(obj: Dict[str, Any], valid_chunk_ids: set[str] | None = None) -> List[str]:
    errors: List[str] = []
    for key in ["answer", "evidence", "confidence", "reason"]:
        if key not in obj:
            errors.append(f"missing:{key}")
    if obj.get("confidence") not in CONFIDENCE_VALUES:
        errors.append("invalid_confidence")
    evidence = obj.get("evidence")
    if not isinstance(evidence, list):
        errors.append("invalid_evidence")
    else:
        for idx, item in enumerate(evidence):
            if not isinstance(item, dict):
                errors.append(f"invalid_evidence_item:{idx}")
                continue
            if not item.get("chunk_id"):
                errors.append(f"missing_evidence_chunk:{idx}")
            if not item.get("quote"):
                errors.append(f"missing_evidence_quote:{idx}")
            if valid_chunk_ids is not None and item.get("chunk_id") not in valid_chunk_ids:
                errors.append(f"unknown_chunk_id:{idx}")
    return errors


def validate_pair(row: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    for key in ["id", "prompt", "chosen", "rejected", "reject_type", "source", "difficulty"]:
        if key not in row:
            errors.append(f"missing:{key}")
    if row.get("reject_type") not in REJECT_TYPES:
        errors.append("invalid_reject_type")
    if row.get("source") not in PAIR_SOURCES:
        errors.append("invalid_source")
    if row.get("difficulty") not in PAIR_DIFFICULTIES:
        errors.append("invalid_difficulty")
    if row.get("answerability_type") is not None and row.get("answerability_type") not in ANSWERABILITY_TYPES:
        errors.append("invalid_answerability_type")
    return errors
