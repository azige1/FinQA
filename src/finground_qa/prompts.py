"""Prompt and target rendering."""

from __future__ import annotations

import json
from typing import Any, Dict, List


SYSTEM_PROMPT = (
    "You are a grounded financial QA assistant. Answer only from the provided "
    "evidence contexts. Return strict JSON with answer, evidence, confidence, and reason. "
    "If the evidence is insufficient, say so with low confidence. Do not invent facts."
)


def render_contexts(row: Dict[str, Any], max_chars: int = 6000) -> str:
    pieces: List[str] = []
    used = 0
    for ctx in row.get("contexts", []):
        text = str(ctx.get("text", ""))
        header = f"[{ctx.get('chunk_id')}] type={ctx.get('type')}"
        block = f"{header}\n{text.strip()}"
        if used + len(block) > max_chars:
            remain = max_chars - used
            if remain <= 200:
                break
            block = block[:remain]
        pieces.append(block)
        used += len(block)
    return "\n\n".join(pieces)


def build_prompt(row: Dict[str, Any]) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Question:\n{row['question']}\n\n"
        f"Evidence contexts:\n{render_contexts(row)}\n\n"
        "Return JSON only in this schema:\n"
        '{"answer": "...", "evidence": [{"chunk_id": "...", "quote": "..."}], '
        '"confidence": "high|medium|low", "reason": "..."}'
    )


def build_target(row: Dict[str, Any]) -> str:
    confidence = "high"
    if row.get("grounding_type") == "unanswerable":
        confidence = "low"
    elif row.get("grounding_type") == "calculation_hard":
        confidence = "medium"
    obj = {
        "answer": row.get("answer", ""),
        "evidence": row.get("evidence", []),
        "confidence": confidence,
        "reason": make_reason(row),
    }
    return json.dumps(obj, ensure_ascii=False)


def make_reason(row: Dict[str, Any]) -> str:
    gt = row.get("grounding_type")
    if gt == "numeric_grounded":
        return "The answer is grounded in the cited financial context and requires simple numeric grounding."
    if gt == "calculation_hard":
        return "The answer involves calculation over the cited financial context; verify the numeric derivation."
    if gt == "unanswerable":
        return "The provided contexts are insufficient to answer with high confidence."
    return "The answer is directly supported by the cited evidence."


def to_sft_record(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": row["id"],
        "dataset": row["dataset"],
        "instruction": build_prompt(row),
        "input": "",
        "output": build_target(row),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(row)},
            {"role": "assistant", "content": build_target(row)},
        ],
        "meta": {
            "answer_type": row.get("answer_type"),
            "reasoning_type": row.get("reasoning_type"),
            "grounding_type": row.get("grounding_type"),
        },
    }


def to_eval_record(row: Dict[str, Any]) -> Dict[str, Any]:
    answerability_type = "unanswerable" if row.get("grounding_type") == "unanswerable" or row.get("answer_type") == "unanswerable" else "answerable"
    return {
        "id": row["id"],
        "dataset": row["dataset"],
        "prompt": build_prompt(row),
        "gold": build_target(row),
        "gold_answer": row.get("answer", ""),
        "contexts": row.get("contexts", []),
        "evidence": row.get("evidence", []),
        "answer_type": row.get("answer_type"),
        "reasoning_type": row.get("reasoning_type"),
        "grounding_type": row.get("grounding_type"),
        "answerability_type": answerability_type,
    }
