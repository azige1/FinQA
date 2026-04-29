"""Rule and weak-semantic proxy checkers for grounded outputs."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from .schema import validate_sft_response
from .text_utils import answer_equivalent, extract_numbers, normalize_for_match, token_overlap


def parse_json_output(text: str) -> tuple[Dict[str, Any] | None, str | None]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj, None
        return None, "json_not_object"
    except Exception as exc:
        return None, f"json_parse_error:{type(exc).__name__}"


def check_output(row: Dict[str, Any], prediction: str) -> Dict[str, Any]:
    obj, parse_error = parse_json_output(prediction)
    valid_chunk_ids = {str(ctx.get("chunk_id")) for ctx in row.get("contexts", [])}
    context_by_id = {str(ctx.get("chunk_id")): str(ctx.get("text", "")) for ctx in row.get("contexts", [])}
    if obj is None:
        return {
            "json_valid": False,
            "schema_pass": False,
            "format_error": True,
            "schema_error": True,
            "errors": [parse_error or "json_parse_error"],
            "citation_total": 0,
            "citation_hits": 0,
            "citation_precision": 0.0,
            "chunk_valid": 0.0,
            "quote_hit": 0.0,
            "number_coverage": 0.0,
            "entity_or_token_coverage": 0.0,
            "citation_consistency_score": 0.0,
            "faithfulness_proxy": 0.0,
            "unsupported_claim": True,
            "wrong_citation": True,
            "missing_evidence": False,
            "fabricated_number": False,
            "calculation_error": False,
            "over_refusal": False,
            "forced_answer": False,
            "generic_answer": False,
        }
    errors = validate_sft_response(obj, valid_chunk_ids)
    citation_hits = 0
    chunk_valid_hits = 0
    citation_total = 0
    wrong_citation = False
    evidence_text = []
    for ev in obj.get("evidence", []) if isinstance(obj.get("evidence"), list) else []:
        citation_total += 1
        cid = str(ev.get("chunk_id"))
        quote = str(ev.get("quote", ""))
        ctx_text = context_by_id.get(cid, "")
        if not ctx_text:
            wrong_citation = True
        else:
            chunk_valid_hits += 1
            quote_nums = set(extract_numbers(quote))
            ctx_nums = set(extract_numbers(ctx_text))
            numeric_quote_hit = bool(quote_nums and (quote_nums & ctx_nums))
            if (normalize_for_match(quote) and normalize_for_match(quote) in normalize_for_match(ctx_text)) or numeric_quote_hit:
                citation_hits += 1
                evidence_text.append(quote)
            else:
                wrong_citation = True
                evidence_text.append(ctx_text[:300])
    answer = str(obj.get("answer", ""))
    gold_answer = str(row.get("answer", row.get("gold_answer", "")))
    answer_matches_gold = bool(gold_answer) and answer_equivalent(answer, gold_answer)
    evidence_joined = " ".join(evidence_text)
    answer_nums = set(extract_numbers(answer))
    evidence_nums = set(extract_numbers(evidence_joined))
    numeric_supported = not answer_nums or bool(answer_nums & evidence_nums)
    number_coverage = 1.0 if not answer_nums else float(bool(answer_nums & evidence_nums))
    overlap = token_overlap(answer, evidence_joined)
    entity_or_token_coverage = min(1.0, overlap / 0.2) if evidence_joined else 0.0
    answerable = row.get("grounding_type") != "unanswerable"
    low_conf = obj.get("confidence") == "low"
    refusal = is_refusal(answer)
    missing_evidence = answerable and bool(answer.strip()) and citation_total == 0
    if answerable:
        unsupported = bool(answer.strip()) and (
            missing_evidence or (overlap < 0.15 and not numeric_supported and not answer_matches_gold)
        )
    else:
        unsupported = not low_conf and not refusal
    over_refusal = answerable and (low_conf or refusal)
    forced_answer = not answerable and not low_conf and not refusal
    generic = is_generic(answer, row.get("answer_type"))
    fabricated_number = bool(answer_nums) and number_coverage == 0.0 and not answer_matches_gold
    calculation_error = row.get("answer_type") == "number" and bool(answer.strip()) and not answer_matches_gold
    if not answerable and (low_conf or refusal) and citation_total == 0:
        chunk_valid = 1.0
        quote_hit = 1.0
        number_coverage = 1.0
        entity_or_token_coverage = 1.0
    else:
        chunk_valid = chunk_valid_hits / max(1, citation_total)
        quote_hit = citation_hits / max(1, citation_total)
    citation_consistency_score = 0.25 * chunk_valid + 0.25 * quote_hit + 0.25 * number_coverage + 0.25 * entity_or_token_coverage
    return {
        "json_valid": True,
        "schema_pass": not errors,
        "format_error": False,
        "schema_error": bool(errors),
        "errors": errors,
        "citation_total": citation_total,
        "citation_hits": citation_hits,
        "citation_precision": citation_hits / max(1, citation_total),
        "chunk_valid": chunk_valid,
        "quote_hit": quote_hit,
        "number_coverage": number_coverage,
        "entity_or_token_coverage": entity_or_token_coverage,
        "citation_consistency_score": citation_consistency_score,
        "wrong_citation": wrong_citation,
        "numeric_supported": numeric_supported,
        "token_overlap": overlap,
        "faithfulness_proxy": 1.0 if not unsupported and not wrong_citation else 0.0,
        "unsupported_claim": unsupported,
        "missing_evidence": missing_evidence,
        "fabricated_number": fabricated_number,
        "calculation_error": calculation_error,
        "over_refusal": over_refusal,
        "forced_answer": forced_answer,
        "generic_answer": generic,
    }


def is_refusal(text: str) -> bool:
    low = normalize_for_match(text)
    patterns = [
        "insufficient evidence",
        "not enough evidence",
        "cannot answer",
        "unable to answer",
        "provided context is insufficient",
    ]
    return any(p in low for p in patterns)


def is_generic(text: str, answer_type: str | None = None) -> bool:
    low = normalize_for_match(text)
    if answer_type in {"number", "yes_no"}:
        return False
    if any(ch.isdigit() for ch in low):
        return False
    generic_phrases = {
        "not enough information",
        "cannot determine",
        "insufficient information",
        "unknown",
        "n/a",
    }
    return low in generic_phrases
