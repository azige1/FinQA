"""Rule-generated DPO pairs and pair audit helpers."""

from __future__ import annotations

import json
import random
from typing import Any, Dict, List

from .prompts import build_prompt, build_target


def make_rule_pairs(rows: List[Dict[str, Any]], target: int = 1000, seed: int = 42) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    candidates = list(rows)
    rng.shuffle(candidates)
    pairs: List[Dict[str, Any]] = []
    plan = [
        ("fabricated_number", reject_numeric_corruption, int(target * 0.25)),
        ("wrong_citation", reject_wrong_citation, int(target * 0.25)),
        ("over_refusal", reject_over_refusal, int(target * 0.15)),
        ("forced_answer", reject_forced_answer, int(target * 0.10)),
        ("missing_evidence", reject_missing_evidence, int(target * 0.15)),
        (
            "wrong_format",
            reject_wrong_format,
            target
            - int(target * 0.25)
            - int(target * 0.25)
            - int(target * 0.15)
            - int(target * 0.10)
            - int(target * 0.15),
        ),
    ]
    used = set()
    for _, gen, quota in plan:
        made = 0
        for row in candidates:
            key = (row["id"], gen.__name__)
            if key in used:
                continue
            pair = gen(row)
            if not pair:
                continue
            used.add(key)
            pairs.append(standardize_pair(row, pair, len(pairs)))
            made += 1
            if made >= quota or len(pairs) >= target:
                break
        if len(pairs) >= target:
            break
    fallback_generators = [
        reject_numeric_corruption,
        reject_wrong_citation,
        reject_over_refusal,
        reject_forced_answer,
        reject_missing_evidence,
        reject_wrong_format,
    ]
    for row in candidates:
        if len(pairs) >= target:
            break
        for gen in fallback_generators:
            key = (row["id"], gen.__name__)
            if key in used:
                continue
            pair = gen(row)
            if not pair:
                continue
            used.add(key)
            pairs.append(standardize_pair(row, pair, len(pairs)))
            break
    return pairs


def standardize_pair(row: Dict[str, Any], pair: Dict[str, Any], index: int | None = None) -> Dict[str, Any]:
    """Attach common fields used by Error-Type Balanced DPO reports.

    Existing fields are preserved; missing fields are filled for compatibility
    with older rule/model-mined pairs.
    """
    chosen = pair.get("chosen") or build_target(row)
    rejected = str(pair.get("rejected", ""))
    if index is not None:
        pair.setdefault("id", f"{row['id']}_{pair.get('reject_type', 'unknown')}_{index}")
    pair.setdefault("prompt", build_prompt(row))
    pair["chosen"] = chosen
    pair["rejected"] = rejected
    pair.setdefault("row_id", row.get("id"))
    pair.setdefault("answerability_type", infer_answerability_type(row))
    pair["chosen_length"] = len(str(chosen))
    pair["rejected_length"] = len(str(rejected))
    return pair


def infer_answerability_type(row: Dict[str, Any]) -> str:
    if row.get("grounding_type") == "unanswerable" or row.get("answer_type") == "unanswerable":
        return "unanswerable"
    if row.get("contexts"):
        return "answerable"
    return "unknown"


def reject_missing_evidence(row: Dict[str, Any]) -> Dict[str, Any]:
    obj = {
        "answer": row.get("answer", ""),
        "evidence": [],
        "confidence": "high",
        "reason": "The answer is based on the provided context.",
    }
    return {
        "rejected": json.dumps(obj, ensure_ascii=False),
        "reject_type": "missing_evidence",
        "source": "rule_generated",
        "difficulty": "easy",
        "answerability_type": infer_answerability_type(row),
    }


def reject_wrong_format(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "rejected": f"Answer: {row.get('answer', '')}",
        "reject_type": "wrong_format",
        "source": "rule_generated",
        "difficulty": "easy",
        "answerability_type": infer_answerability_type(row),
    }


def reject_wrong_citation(row: Dict[str, Any]) -> Dict[str, Any] | None:
    contexts = row.get("contexts") or []
    evidence = row.get("evidence") or []
    if len(contexts) < 2 or not evidence:
        return None
    gold = evidence[0]
    wrong = next((ctx for ctx in contexts if ctx.get("chunk_id") != gold.get("chunk_id")), None)
    if not wrong:
        return None
    obj = {
        "answer": row.get("answer", ""),
        "evidence": [{"chunk_id": wrong.get("chunk_id"), "quote": str(wrong.get("text", ""))[:180]}],
        "confidence": "high",
        "reason": "The cited evidence supports the answer.",
    }
    return {
        "rejected": json.dumps(obj, ensure_ascii=False),
        "reject_type": "wrong_citation",
        "source": "wrong_citation_corruption",
        "difficulty": "medium",
        "answerability_type": "answerable",
    }


def reject_numeric_corruption(row: Dict[str, Any]) -> Dict[str, Any] | None:
    answer = str(row.get("answer", ""))
    digits = [ch for ch in answer if ch.isdigit()]
    if not digits:
        return None
    corrupted = answer
    for digit in digits[:1]:
        new_digit = "9" if digit != "9" else "8"
        corrupted = corrupted.replace(digit, new_digit, 1)
        break
    obj = {
        "answer": corrupted,
        "evidence": row.get("evidence", []),
        "confidence": "high",
        "reason": "The numeric answer is directly supported by the evidence.",
    }
    return {
        "rejected": json.dumps(obj, ensure_ascii=False),
        "reject_type": "fabricated_number",
        "source": "numeric_corruption",
        "difficulty": "hard",
        "answerability_type": "answerable",
    }


def reject_over_refusal(row: Dict[str, Any]) -> Dict[str, Any] | None:
    if row.get("grounding_type") == "unanswerable":
        return None
    obj = {
        "answer": "The provided evidence is insufficient to answer this question.",
        "evidence": [],
        "confidence": "low",
        "reason": "I cannot answer from the provided context.",
    }
    return {
        "rejected": json.dumps(obj, ensure_ascii=False),
        "reject_type": "over_refusal",
        "source": "rule_generated",
        "difficulty": "hard",
        "answerability_type": "answerable",
    }


def reject_forced_answer(row: Dict[str, Any]) -> Dict[str, Any] | None:
    if infer_answerability_type(row) != "unanswerable":
        return None
    contexts = row.get("contexts") or []
    if not contexts:
        return None
    ctx = contexts[0]
    quote = str(ctx.get("text", ""))[:180]
    obj = {
        "answer": "The provided evidence indicates a positive financial result.",
        "evidence": [{"chunk_id": ctx.get("chunk_id"), "quote": quote}],
        "confidence": "high",
        "reason": "The cited context is sufficient to provide a definitive answer.",
    }
    return {
        "rejected": json.dumps(obj, ensure_ascii=False),
        "reject_type": "forced_answer",
        "source": "hard_negative",
        "difficulty": "hard",
        "answerability_type": "unanswerable",
    }
