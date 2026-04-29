"""Build targeted DPO v4 data from train-split rows only.

v4 is an anti-regression preference pool. It uses held-out eval failures only
as a taxonomy and builds fresh train-split pairs in four separated buckets:
- numeric scale guard
- protect-correct guard
- unanswerable refusal guard
- citation repair guard
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.finground_qa.dpo import standardize_pair
from src.finground_qa.pipeline import pair_difficulty_report
from src.finground_qa.prompts import build_prompt, build_target, make_reason
from src.finground_qa.reports import preference_pair_quality_report
from src.finground_qa.text_utils import extract_numbers, normalize_text


RAW_NUMBER_RE = re.compile(r"[-+]?\$?\(?\d[\d,]*(?:\.\d+)?\)?%?")
INSUFFICIENT = "Insufficient evidence to answer from the provided contexts."

DEFAULT_QUOTAS = {
    "numeric_scale_guard": 240,
    "protect_correct_guard": 160,
    "unanswerable_refusal_guard": 120,
    "citation_repair_guard": 160,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def is_unanswerable(row: dict[str, Any]) -> bool:
    return row.get("grounding_type") == "unanswerable" or row.get("answer_type") == "unanswerable"


def is_answerable(row: dict[str, Any]) -> bool:
    return not is_unanswerable(row) and bool(row.get("contexts")) and bool(row.get("evidence"))


def json_answer(answer: str, evidence: list[dict[str, Any]], confidence: str, reason: str) -> str:
    return json.dumps(
        {"answer": answer, "evidence": evidence, "confidence": confidence, "reason": reason},
        ensure_ascii=False,
    )


def parse_first_number(text: Any) -> tuple[str, float] | None:
    nums = extract_numbers(text)
    if not nums:
        return None
    raw = nums[0]
    try:
        return raw, float(raw)
    except ValueError:
        return None


def format_number_like(value: float, raw: str) -> str:
    decimals = 0
    if "." in raw:
        decimals = len(raw.split(".", 1)[1])
    if decimals > 0:
        return f"{value:.{min(decimals, 6)}f}".rstrip("0").rstrip(".")
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def replace_first_number(text: Any, replacement: str) -> str:
    return RAW_NUMBER_RE.sub(replacement, str(text), count=1)


def first_context_evidence(row: dict[str, Any]) -> list[dict[str, Any]]:
    contexts = row.get("contexts") or []
    if not contexts:
        return []
    ctx = contexts[0]
    return [{"chunk_id": ctx.get("chunk_id"), "quote": normalize_text(ctx.get("text", ""))[:180]}]


def snippet_around(text: str, needle: str, window: int = 220) -> str:
    if not needle:
        return normalize_text(text)[:window]
    pos = text.lower().find(needle.lower())
    if pos < 0:
        return normalize_text(text)[:window]
    start = max(0, pos - window // 3)
    end = min(len(text), pos + len(needle) + window)
    return normalize_text(text[start:end])


def raw_number_forms(answer: Any) -> list[str]:
    forms = []
    for num in extract_numbers(answer):
        forms.append(num)
        if num.startswith("-"):
            forms.append(num[1:])
        elif num and num != "0":
            forms.append("-" + num)
    seen = set()
    out = []
    for form in forms:
        if form and form not in seen:
            out.append(form)
            seen.add(form)
    return out


def supporting_evidence(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Select a v4 training citation that is present in prompt contexts.

    Some source rows have weak original evidence for calculated answers. v4 is
    citation-sensitive, so use a local repair heuristic for the chosen side.
    """
    if is_unanswerable(row):
        return []
    contexts = row.get("contexts") or []
    answer = normalize_text(row.get("answer", ""))
    if not contexts:
        return row.get("evidence", [])

    if answer:
        for ctx in contexts:
            text = normalize_text(ctx.get("text", ""))
            if answer.lower() in text.lower():
                return [{"chunk_id": ctx.get("chunk_id"), "quote": snippet_around(text, answer)}]

    for form in raw_number_forms(answer):
        for ctx in contexts:
            text = normalize_text(ctx.get("text", ""))
            if form in text:
                return [{"chunk_id": ctx.get("chunk_id"), "quote": snippet_around(text, form)}]

    if row.get("grounding_type") == "calculation_hard" or row.get("reasoning_type") == "calculation":
        table = next((ctx for ctx in contexts if ctx.get("type") == "table"), None)
        if table:
            return [{"chunk_id": table.get("chunk_id"), "quote": normalize_text(table.get("text", ""))[:300]}]

    context_ids = {str(ctx.get("chunk_id")) for ctx in contexts}
    for ev in row.get("evidence") or []:
        chunk_id = str(ev.get("chunk_id"))
        quote = normalize_text(ev.get("quote", ""))
        ctx_text = normalize_text(next((ctx.get("text", "") for ctx in contexts if str(ctx.get("chunk_id")) == chunk_id), ""))
        if chunk_id in context_ids and quote and quote in ctx_text:
            return [{"chunk_id": ev.get("chunk_id"), "quote": quote}]

    return first_context_evidence(row)


def chosen_target(row: dict[str, Any]) -> str:
    if is_unanswerable(row):
        return build_target(row)
    confidence = "medium" if row.get("grounding_type") == "calculation_hard" else "high"
    return json_answer(str(row.get("answer", "")), supporting_evidence(row), confidence, make_reason(row))


def wrong_context_evidence(row: dict[str, Any]) -> list[dict[str, Any]] | None:
    contexts = row.get("contexts") or []
    evidence = row.get("evidence") or []
    if len(contexts) < 2 or not evidence:
        return None
    gold_chunk_ids = {str(ev.get("chunk_id")) for ev in evidence}
    answer = normalize_text(row.get("answer", ""))
    answer_numbers = raw_number_forms(answer)

    def is_supported_by_answer(ctx: dict[str, Any]) -> bool:
        text = normalize_text(ctx.get("text", ""))
        if answer and answer.lower() in text.lower():
            return True
        return any(num and num in text for num in answer_numbers)

    wrong = next(
        (
            ctx
            for ctx in contexts
            if str(ctx.get("chunk_id")) not in gold_chunk_ids and not is_supported_by_answer(ctx)
        ),
        None,
    )
    if not wrong:
        return None
    return [{"chunk_id": wrong.get("chunk_id"), "quote": normalize_text(wrong.get("text", ""))[:180]}]


def make_pair(row: dict[str, Any], pair: dict[str, Any]) -> dict[str, Any]:
    pair.setdefault("prompt", build_prompt(row))
    pair.setdefault("chosen", chosen_target(row))
    pair.setdefault("row_id", row.get("id"))
    pair.setdefault("source_split", row.get("source_split"))
    pair.setdefault("dataset", row.get("dataset"))
    pair.setdefault("answer_type", row.get("answer_type"))
    pair.setdefault("reasoning_type", row.get("reasoning_type"))
    pair.setdefault("grounding_type", row.get("grounding_type"))
    pair.setdefault("dpo_v4_policy", "targeted_anti_regression_train_only")
    return standardize_pair(row, pair)


def numeric_scale_candidates(row: dict[str, Any]) -> list[dict[str, Any]]:
    if row.get("answer_type") != "number" or not is_answerable(row):
        return []
    parsed = parse_first_number(row.get("answer", ""))
    if parsed is None:
        return []
    raw, value = parsed
    answer = str(row.get("answer", ""))
    candidates: list[tuple[str, float, str]] = []
    if value != 0:
        candidates.append(("sign_flip", -value, "fabricated_number"))
    if "." in raw and len(raw.split(".", 1)[1]) >= 2:
        candidates.append(("rounding_loss", round(value, max(0, len(raw.split(".", 1)[1]) - 1)), "calculation_error"))
    if "." in raw:
        decimals = len(raw.split(".", 1)[1])
        delta = 10 ** -decimals
        candidates.append(("last_digit_drift", value + delta, "calculation_error"))
    else:
        candidates.append(("last_digit_drift", value + 1, "calculation_error"))
    if abs(value) >= 1:
        candidates.append(("decimal_scale_shift", value / 1000, "fabricated_number"))
    elif 0 < abs(value) < 1:
        candidates.append(("decimal_scale_shift", value * 1000, "fabricated_number"))
    candidates.append(("large_scale_shift", value * 1_000_000, "fabricated_number"))
    if "%" in answer or 0 < abs(value) <= 100:
        shifted = value / 100 if abs(value) >= 1 else value * 100
        candidates.append(("percent_fraction_shift", shifted, "calculation_error"))

    out = []
    seen = {answer}
    for variant, corrupted_value, reject_type in candidates:
        corrupted = replace_first_number(answer, format_number_like(corrupted_value, raw))
        if corrupted in seen:
            continue
        seen.add(corrupted)
        out.append(
            make_pair(
                row,
                {
                    "id": f"{row['id']}_v4_numeric_{variant}",
                    "rejected": json_answer(
                        corrupted,
                        supporting_evidence(row),
                        "high",
                        "The numeric answer is directly supported by the cited evidence.",
                    ),
                    "reject_type": reject_type,
                    "source": "v4_numeric_scale_guard",
                    "bucket": "numeric_scale_guard",
                    "difficulty": "hard",
                    "answerability_type": "answerable",
                    "numeric_guard_variant": variant,
                },
            )
        )
    return out


def protect_correct_candidate(row: dict[str, Any]) -> dict[str, Any] | None:
    if not is_answerable(row):
        return None
    answer = normalize_text(row.get("answer", ""))
    if not answer:
        return None
    parsed = parse_first_number(answer)
    if parsed is not None:
        raw, value = parsed
        corrupted = replace_first_number(answer, format_number_like(value + 1 if value >= 0 else value - 1, raw))
        variant = "wrong_number_with_correct_citation"
        reject_type = "fabricated_number"
    else:
        parts = answer.split()
        if len(parts) >= 3:
            corrupted = " ".join(parts[:-1])
        elif len(answer) >= 8:
            corrupted = answer[: max(1, len(answer) // 2)].strip()
        else:
            corrupted = answer + " (partial)"
        if corrupted == answer:
            return None
        variant = "truncated_answer_with_correct_citation"
        reject_type = "unsupported_claim"
    return make_pair(
        row,
        {
            "id": f"{row['id']}_v4_protect_correct_{variant}",
            "rejected": json_answer(
                corrupted,
                supporting_evidence(row),
                "high",
                "The answer is directly supported by the cited evidence.",
            ),
            "reject_type": reject_type,
            "source": "v4_protect_correct_guard",
            "bucket": "protect_correct_guard",
            "difficulty": "hard",
            "answerability_type": "answerable",
            "protect_correct_variant": variant,
        },
    )


def unanswerable_candidates(row: dict[str, Any]) -> list[dict[str, Any]]:
    if not is_unanswerable(row) or not row.get("contexts"):
        return []
    evidence = first_context_evidence(row)
    context_text = " ".join(str(ctx.get("text", "")) for ctx in row.get("contexts") or [])
    nums = extract_numbers(context_text)
    answers = [
        ("generic_entity_answer", "The cited context provides enough information to answer the question."),
    ]
    if nums:
        answers.insert(0, ("fabricated_numeric_answer", nums[0]))
    out = []
    for variant, answer in answers[:2]:
        out.append(
            make_pair(
                row,
                {
                    "id": f"{row['id']}_v4_unanswerable_{variant}",
                    "rejected": json_answer(
                        answer,
                        evidence,
                        "high",
                        "The cited context is sufficient to provide a definitive answer.",
                    ),
                    "reject_type": "forced_answer",
                    "source": "v4_unanswerable_refusal_guard",
                    "bucket": "unanswerable_refusal_guard",
                    "difficulty": "hard",
                    "answerability_type": "unanswerable",
                    "unanswerable_guard_variant": variant,
                },
            )
        )
    return out


def citation_repair_candidate(row: dict[str, Any]) -> dict[str, Any] | None:
    if not is_answerable(row):
        return None
    wrong_evidence = wrong_context_evidence(row)
    if not wrong_evidence:
        return None
    return make_pair(
        row,
        {
            "id": f"{row['id']}_v4_citation_repair_wrong_chunk",
            "rejected": json_answer(
                str(row.get("answer", "")),
                wrong_evidence,
                "high",
                "The cited evidence supports the answer.",
            ),
            "reject_type": "wrong_citation",
            "source": "v4_citation_repair_guard",
            "bucket": "citation_repair_guard",
            "difficulty": "medium",
            "answerability_type": "answerable",
            "citation_guard_variant": "wrong_chunk_same_answer",
        },
    )


def select_balanced_numeric(candidates: list[dict[str, Any]], quota: int, rng: random.Random) -> list[dict[str, Any]]:
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_variant[str(row.get("numeric_guard_variant"))].append(row)
    for bucket in by_variant.values():
        rng.shuffle(bucket)
    variants = sorted(by_variant)
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    per_variant = max(1, quota // max(1, len(variants)))
    for variant in variants:
        for row in by_variant[variant][:per_variant]:
            if len(selected) >= quota:
                break
            row_id = str(row.get("id"))
            if row_id in used:
                continue
            selected.append(row)
            used.add(row_id)
    leftovers = [row for rows in by_variant.values() for row in rows]
    rng.shuffle(leftovers)
    for row in leftovers:
        if len(selected) >= quota:
            break
        row_id = str(row.get("id"))
        if row_id in used:
            continue
        selected.append(row)
        used.add(row_id)
    return selected


def select_unique(candidates: list[dict[str, Any]], quota: int, rng: random.Random) -> list[dict[str, Any]]:
    shuffled = candidates[:]
    rng.shuffle(shuffled)
    selected = []
    used = set()
    for row in shuffled:
        if len(selected) >= quota:
            break
        row_id = str(row.get("id"))
        if row_id in used:
            continue
        selected.append(row)
        used.add(row_id)
    return selected


def make_audit_sample(rows: list[dict[str, Any]], audit_size: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_bucket[str(row.get("bucket", "unknown"))].append(row)
    for bucket in by_bucket.values():
        rng.shuffle(bucket)
    sample: list[dict[str, Any]] = []
    used = set()
    per_bucket = max(1, audit_size // max(1, len(by_bucket)))
    for bucket_name in sorted(by_bucket):
        for row in by_bucket[bucket_name][:per_bucket]:
            if len(sample) >= audit_size:
                break
            rid = str(row.get("id"))
            if rid in used:
                continue
            sample.append(row)
            used.add(rid)
    leftovers = rows[:]
    rng.shuffle(leftovers)
    for row in leftovers:
        if len(sample) >= audit_size:
            break
        rid = str(row.get("id"))
        if rid in used:
            continue
        sample.append(row)
        used.add(rid)
    return [
        {
            "id": row.get("id"),
            "bucket": row.get("bucket"),
            "question": row.get("prompt", ""),
            "chosen": row.get("chosen"),
            "rejected": row.get("rejected"),
            "reject_type": row.get("reject_type"),
            "source": row.get("source"),
            "difficulty": row.get("difficulty"),
            "answerability_type": row.get("answerability_type"),
            "chosen_length": row.get("chosen_length") or len(str(row.get("chosen", ""))),
            "rejected_length": row.get("rejected_length") or len(str(row.get("rejected", ""))),
            "is_chosen_better": None,
            "issue": "needs_manual_review",
            "comment": "",
        }
        for row in sample
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/unified/train_unified.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/dpo/dpo_targeted_v4.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("reports/dpo_targeted_v4_report.json"))
    parser.add_argument("--audit-output", type=Path, default=Path("results/dpo_targeted_v4_pair_audit_120.jsonl"))
    parser.add_argument("--audit-size", type=int, default=120)
    parser.add_argument("--numeric-scale-guard", type=int, default=DEFAULT_QUOTAS["numeric_scale_guard"])
    parser.add_argument("--protect-correct-guard", type=int, default=DEFAULT_QUOTAS["protect_correct_guard"])
    parser.add_argument("--unanswerable-refusal-guard", type=int, default=DEFAULT_QUOTAS["unanswerable_refusal_guard"])
    parser.add_argument("--citation-repair-guard", type=int, default=DEFAULT_QUOTAS["citation_repair_guard"])
    parser.add_argument("--seed", type=int, default=45)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    train_rows = read_jsonl(args.train)
    if any(str(row.get("source_split")) != "train" for row in train_rows):
        raise SystemExit("train file contains non-train source_split rows; refusing to build v4")

    rows = train_rows[:]
    rng.shuffle(rows)

    numeric_candidates = [pair for row in rows for pair in numeric_scale_candidates(row)]
    protect_candidates = [pair for row in rows if (pair := protect_correct_candidate(row))]
    unanswerable_pool = [pair for row in rows for pair in unanswerable_candidates(row)]

    citation_source_rows = [row for row in rows if row.get("answer_type") != "number"]
    citation_candidates = [pair for row in citation_source_rows if (pair := citation_repair_candidate(row))]
    citation_numeric_fallback_used = 0
    if len(citation_candidates) < args.citation_repair_guard:
        fallback_rows = [row for row in rows if row.get("answer_type") == "number"]
        fallback = [pair for row in fallback_rows if (pair := citation_repair_candidate(row))]
        needed = args.citation_repair_guard - len(citation_candidates)
        citation_candidates.extend(select_unique(fallback, needed, rng))
        citation_numeric_fallback_used = min(len(fallback), needed)

    selected_by_bucket = {
        "numeric_scale_guard": select_balanced_numeric(numeric_candidates, args.numeric_scale_guard, rng),
        "protect_correct_guard": select_unique(protect_candidates, args.protect_correct_guard, rng),
        "unanswerable_refusal_guard": select_unique(unanswerable_pool, args.unanswerable_refusal_guard, rng),
        "citation_repair_guard": select_unique(citation_candidates, args.citation_repair_guard, rng),
    }

    selected = [row for bucket in selected_by_bucket.values() for row in bucket]
    used_ids: set[str] = set()
    deduped = []
    for row in selected:
        rid = str(row.get("id"))
        if rid in used_ids:
            continue
        used_ids.add(rid)
        deduped.append(row)
    rng.shuffle(deduped)

    write_jsonl(args.output, deduped)
    audit = make_audit_sample(deduped, args.audit_size, args.seed)
    write_jsonl(args.audit_output, audit)

    quality = preference_pair_quality_report(deduped)
    difficulty = pair_difficulty_report(deduped)
    requested = {
        "numeric_scale_guard": args.numeric_scale_guard,
        "protect_correct_guard": args.protect_correct_guard,
        "unanswerable_refusal_guard": args.unanswerable_refusal_guard,
        "citation_repair_guard": args.citation_repair_guard,
    }
    actual = dict(Counter(str(row.get("bucket")) for row in deduped))
    report = {
        "input_train": str(args.train),
        "output": str(args.output),
        "audit_output": str(args.audit_output),
        "seed": args.seed,
        "rows": len(deduped),
        "requested_buckets": requested,
        "actual_buckets": actual,
        "candidate_counts": {
            "numeric_scale_guard": len(numeric_candidates),
            "protect_correct_guard": len(protect_candidates),
            "unanswerable_refusal_guard": len(unanswerable_pool),
            "citation_repair_guard": len(citation_candidates),
        },
        "quota_pass": {name: actual.get(name, 0) >= quota for name, quota in requested.items()},
        "source": dict(Counter(str(row.get("source")) for row in deduped)),
        "reject_type": dict(Counter(str(row.get("reject_type")) for row in deduped)),
        "difficulty": dict(Counter(str(row.get("difficulty")) for row in deduped)),
        "answerability_type": dict(Counter(str(row.get("answerability_type")) for row in deduped)),
        "answer_type": dict(Counter(str(row.get("answer_type")) for row in deduped)),
        "numeric_guard_variant": dict(Counter(str(row.get("numeric_guard_variant")) for row in deduped if row.get("numeric_guard_variant"))),
        "quality": quality,
        "difficulty_report": difficulty,
        "leakage_guard": {
            "uses_eval_rows": False,
            "train_source_split": dict(Counter(str(row.get("source_split")) for row in train_rows)),
            "post_dpo_eval_failures_used_as_training_rows": False,
            "post_dpo_eval_failures_used_as_taxonomy_only": True,
        },
        "citation_repair_guard": {
            "numeric_answer_fallback_rows": citation_numeric_fallback_used,
            "note": "Citation repair is selected from non-numeric rows first; numeric fallback is only used if quota cannot be met.",
        },
        "formal_gate": {
            "status": "pending_manual_pair_audit",
            "gate_pass": False,
            "reason": "v4 adds fresh train-split targeted guard pairs; audit sample must be reviewed before formal DPO training is accepted.",
        },
        "notes": [
            "Built fresh train-split v4 pairs instead of directly training on held-out badcases.",
            "Separated numeric scale, protect-correct, unanswerable refusal, and citation repair buckets.",
            "Designed for short beta=0.05 checkpoint sweeps at 50/75/100 steps.",
        ],
    }
    report["gate_ready_for_manual_review"] = all(report["quota_pass"].values()) and quality["invalid_rows"] == 0
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
