"""Text, numeric, and table helpers for grounded QA data curation."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence


NUMBER_RE = re.compile(r"[-+]?\$?\(?\d[\d,]*(?:\.\d+)?\)?%?")
WORD_RE = re.compile(r"[A-Za-z0-9_.$%+-]+")


def normalize_text(text: Any) -> str:
    text = "" if text is None else str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\s+", " ", text).strip()


def normalize_for_match(text: Any) -> str:
    return normalize_text(text).lower()


def tokens(text: Any) -> List[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(normalize_text(text))]


def token_overlap(a: Any, b: Any) -> float:
    ta = tokens(a)
    tb = tokens(b)
    if not ta or not tb:
        return 0.0
    ca = Counter(ta)
    cb = Counter(tb)
    inter = sum((ca & cb).values())
    return inter / max(1, min(len(ta), len(tb)))


def extract_numbers(text: Any) -> List[str]:
    values: List[str] = []
    for raw in NUMBER_RE.findall(normalize_text(text)):
        cleaned = raw.replace("$", "").replace(",", "").replace("%", "")
        neg = cleaned.startswith("(") and cleaned.endswith(")")
        cleaned = cleaned.strip("()")
        if neg:
            cleaned = "-" + cleaned
        values.append(cleaned)
    return values


def number_hit(answer: Any, context: Any) -> bool:
    ans_nums = set(extract_numbers(answer))
    if not ans_nums:
        return True
    ctx_nums = set(extract_numbers(context))
    return bool(ans_nums & ctx_nums)


def answer_equivalent(prediction: Any, gold: Any) -> bool:
    pred_norm = normalize_for_match(prediction)
    gold_norm = normalize_for_match(gold)
    if pred_norm == gold_norm:
        return True
    pred_nums = extract_numbers(prediction)
    gold_nums = extract_numbers(gold)
    if pred_nums and gold_nums and set(pred_nums) == set(gold_nums):
        return True
    return False


def stringify_answer(answer: Any, scale: str | None = None) -> str:
    if isinstance(answer, list):
        text = ", ".join(normalize_text(x) for x in answer)
    elif isinstance(answer, dict):
        text = normalize_text(answer.get("answer") or answer.get("value") or answer)
    else:
        text = normalize_text(answer)
    if scale and scale.strip() and scale.strip().lower() not in {"none", "nan"}:
        return f"{text} {scale.strip()}".strip()
    return text


def table_to_markdown(table: Sequence[Sequence[Any]], table_id: str) -> tuple[str, List[Dict[str, Any]]]:
    rows = [[normalize_text(cell) for cell in row] for row in table if row is not None]
    if not rows:
        return "", []
    width = max(len(row) for row in rows)
    padded = [row + [""] * (width - len(row)) for row in rows]
    header = padded[0]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    cell_meta: List[Dict[str, Any]] = []
    for ridx, row in enumerate(padded[1:], start=1):
        lines.append("| " + " | ".join(row) + " |")
        for cidx, cell in enumerate(row):
            if cell:
                cell_meta.append(
                    {
                        "table_id": table_id,
                        "row_id": f"{table_id}_r{ridx}",
                        "col_id": f"{table_id}_c{cidx}",
                        "text": cell,
                    }
                )
    return "\n".join(lines), cell_meta


def find_quote(answer: str, contexts: Sequence[Dict[str, Any]]) -> Dict[str, str] | None:
    answer_norm = normalize_for_match(answer)
    if not answer_norm:
        return None
    for ctx in contexts:
        text = normalize_text(ctx.get("text", ""))
        if answer_norm and answer_norm in normalize_for_match(text):
            return {"chunk_id": str(ctx.get("chunk_id")), "quote": answer}
    answer_nums = extract_numbers(answer)
    for num in answer_nums:
        for ctx in contexts:
            text = normalize_text(ctx.get("text", ""))
            if num and num in {n for n in extract_numbers(text)}:
                return {"chunk_id": str(ctx.get("chunk_id")), "quote": num}
    best = None
    best_score = 0.0
    for ctx in contexts:
        score = token_overlap(answer, ctx.get("text", ""))
        if score > best_score:
            best_score = score
            best = ctx
    if best is not None and best_score >= 0.2:
        quote = normalize_text(best.get("text", ""))[:300]
        return {"chunk_id": str(best.get("chunk_id")), "quote": quote}
    return None


def classify_grounding(answer: str, contexts: Sequence[Dict[str, Any]], answer_type: str, derivation: str = "") -> str:
    all_context = " ".join(str(ctx.get("text", "")) for ctx in contexts)
    if answer_type == "unanswerable":
        return "unanswerable"
    if normalize_for_match(answer) and normalize_for_match(answer) in normalize_for_match(all_context):
        return "direct_grounded"
    if extract_numbers(answer):
        if number_hit(answer, all_context):
            return "numeric_grounded"
        if derivation.strip():
            return "calculation_hard"
    if derivation.strip():
        return "calculation_hard"
    return "direct_grounded"
