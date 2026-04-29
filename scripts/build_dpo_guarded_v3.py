
"""Build guarded DPO v3 data from train-split rows only.

v3 uses the post-DPO audit as a failure taxonomy, not as training data.
It keeps v2 pairs and adds train-split guard pairs for:
- unanswerable forced-answer regressions
- numeric precision/sign/scale regressions
"""

from __future__ import annotations

import argparse
import sys
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from typing import Any

from src.finground_qa.dpo import standardize_pair
from src.finground_qa.prompts import build_prompt, build_target
from src.finground_qa.reports import preference_pair_quality_report
from src.finground_qa.pipeline import pair_difficulty_report
from src.finground_qa.text_utils import extract_numbers

INSUFFICIENT = "Insufficient evidence to answer from the provided contexts."


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


def json_answer(answer: str, evidence: list[dict[str, Any]], confidence: str, reason: str) -> str:
    return json.dumps(
        {"answer": answer, "evidence": evidence, "confidence": confidence, "reason": reason},
        ensure_ascii=False,
    )


def first_context_evidence(row: dict[str, Any]) -> list[dict[str, Any]]:
    contexts = row.get("contexts") or []
    if not contexts:
        return []
    ctx = contexts[0]
    return [{"chunk_id": ctx.get("chunk_id"), "quote": str(ctx.get("text", ""))[:180]}]


def make_unanswerable_forced_pairs(row: dict[str, Any], start_index: int) -> list[dict[str, Any]]:
    if not is_unanswerable(row):
        return []
    pairs = []
    evidence = first_context_evidence(row)
    generic = json_answer(
        "The provided evidence indicates a positive financial result.",
        evidence,
        "high",
        "The cited context is sufficient to provide a definitive answer.",
    )
    pairs.append(
        standardize_pair(
            row,
            {
                "id": f"{row['id']}_v3_forced_answer_generic_{start_index}",
                "prompt": build_prompt(row),
                "chosen": build_target(row),
                "rejected": generic,
                "reject_type": "forced_answer",
                "source": "v3_unanswerable_guard",
                "difficulty": "hard",
                "row_id": row["id"],
                "answerability_type": "unanswerable",
                "dpo_v3_policy": "guard_unanswerable_and_numeric_precision_train_only",
            },
        )
    )
    all_text = " ".join(str(ctx.get("text", "")) for ctx in row.get("contexts") or [])
    nums = extract_numbers(all_text)
    if nums:
        num = nums[0]
        numeric = json_answer(
            num,
            evidence,
            "high",
            "The cited context contains the numeric answer.",
        )
        pairs.append(
            standardize_pair(
                row,
                {
                    "id": f"{row['id']}_v3_forced_answer_numeric_{start_index}",
                    "prompt": build_prompt(row),
                    "chosen": build_target(row),
                    "rejected": numeric,
                    "reject_type": "forced_answer",
                    "source": "v3_unanswerable_guard",
                    "difficulty": "hard",
                    "row_id": row["id"],
                    "answerability_type": "unanswerable",
                    "dpo_v3_policy": "guard_unanswerable_and_numeric_precision_train_only",
                },
            )
        )
    return pairs


def parse_number(answer: str) -> float | None:
    nums = extract_numbers(answer)
    if not nums:
        return None
    try:
        return float(nums[0])
    except Exception:
        return None


def replace_first_number(text: str, replacement: str) -> str:
    return re.sub(r"[-+]?\$?\(?\d[\d,]*(?:\.\d+)?\)?%?", replacement, str(text), count=1)


def numeric_corruptions(answer: str) -> list[tuple[str, str, str]]:
    answer = str(answer)
    value = parse_number(answer)
    if value is None:
        return []
    out: list[tuple[str, str, str]] = []
    nums = extract_numbers(answer)
    raw = nums[0] if nums else str(value)
    if "." in raw:
        decimals = len(raw.split(".", 1)[1])
        if decimals >= 4:
            rounded = f"{value:.{max(0, decimals - 1)}f}"
            if rounded != raw:
                out.append(("calculation_error", replace_first_number(answer, rounded), "precision_rounding_loss"))
        if decimals >= 2:
            shifted = value + (10 ** -decimals)
            out.append(("calculation_error", replace_first_number(answer, f"{shifted:.{decimals}f}"), "last_decimal_drift"))
    if value != 0:
        out.append(("fabricated_number", replace_first_number(answer, str(-value)), "sign_flip"))
    if abs(value) >= 1:
        out.append(("calculation_error", replace_first_number(answer, str(value + 1)), "off_by_one"))
    if abs(value) >= 10:
        out.append(("fabricated_number", replace_first_number(answer, str(value / 10)), "decimal_scale_shift"))
    elif 0 < abs(value) < 1:
        out.append(("fabricated_number", replace_first_number(answer, str(value * 10)), "decimal_scale_shift"))
    dedup = []
    seen = {answer}
    for reject_type, corrupted, variant in out:
        if corrupted in seen:
            continue
        seen.add(corrupted)
        dedup.append((reject_type, corrupted, variant))
    return dedup


def make_numeric_guard_pairs(row: dict[str, Any], start_index: int) -> list[dict[str, Any]]:
    if row.get("answer_type") != "number":
        return []
    if is_unanswerable(row):
        return []
    answer = str(row.get("answer", ""))
    corruptions = numeric_corruptions(answer)
    pairs = []
    for idx, (reject_type, corrupted, variant) in enumerate(corruptions):
        rejected = json_answer(
            corrupted,
            row.get("evidence", []),
            "high",
            "The numeric answer is directly supported by the evidence.",
        )
        pairs.append(
            standardize_pair(
                row,
                {
                    "id": f"{row['id']}_v3_numeric_{variant}_{start_index}_{idx}",
                    "prompt": build_prompt(row),
                    "chosen": build_target(row),
                    "rejected": rejected,
                    "reject_type": reject_type,
                    "source": "v3_numeric_precision_guard",
                    "difficulty": "hard",
                    "row_id": row["id"],
                    "answerability_type": "answerable",
                    "numeric_guard_variant": variant,
                    "dpo_v3_policy": "guard_unanswerable_and_numeric_precision_train_only",
                },
            )
        )
    return pairs


def make_audit_sample(rows: list[dict[str, Any]], audit_size: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[str(row.get("source"))].append(row)
    for bucket in by_source.values():
        rng.shuffle(bucket)
    priority = ["v3_unanswerable_guard", "v3_numeric_precision_guard", "model_mined", "hard_negative", "numeric_corruption", "wrong_citation_corruption", "rule_generated"]
    sample: list[dict[str, Any]] = []
    used = set()
    for source in priority:
        bucket = by_source.get(source, [])
        quota = 20 if source.startswith("v3_") else 10
        for row in bucket[:quota]:
            if len(sample) >= audit_size:
                break
            rid = row.get("id")
            if rid in used:
                continue
            used.add(rid)
            sample.append(row)
    remaining = rows[:]
    rng.shuffle(remaining)
    for row in remaining:
        if len(sample) >= audit_size:
            break
        rid = row.get("id")
        if rid in used:
            continue
        used.add(rid)
        sample.append(row)
    audit = []
    for row in sample:
        audit.append(
            {
                "id": row.get("id"),
                "question": row.get("prompt", "")[:1400],
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
        )
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2", type=Path, default=Path("data/dpo/dpo_reweighted_v2.jsonl"))
    parser.add_argument("--train", type=Path, default=Path("data/unified/train_unified.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/dpo/dpo_guarded_v3.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("reports/dpo_guarded_v3_report.json"))
    parser.add_argument("--audit-output", type=Path, default=Path("results/dpo_guarded_v3_pair_audit_120.jsonl"))
    parser.add_argument("--audit-size", type=int, default=120)
    parser.add_argument("--max-forced-guards", type=int, default=150)
    parser.add_argument("--max-numeric-guards", type=int, default=220)
    parser.add_argument("--seed", type=int, default=44)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    v2_rows = read_jsonl(args.v2)
    train_rows = read_jsonl(args.train)
    if any(str(row.get("source_split")) != "train" for row in train_rows):
        raise SystemExit("train file contains non-train source_split rows; refusing to build v3")

    unanswerable_rows = [row for row in train_rows if is_unanswerable(row)]
    numeric_rows = [row for row in train_rows if row.get("answer_type") == "number" and not is_unanswerable(row)]
    rng.shuffle(unanswerable_rows)
    rng.shuffle(numeric_rows)

    forced_guards = []
    for idx, row in enumerate(unanswerable_rows):
        forced_guards.extend(make_unanswerable_forced_pairs(row, idx))
        if len(forced_guards) >= args.max_forced_guards:
            break
    forced_guards = forced_guards[: args.max_forced_guards]

    numeric_guards = []
    seen_numeric_row = Counter()
    for idx, row in enumerate(numeric_rows):
        candidates = make_numeric_guard_pairs(row, idx)
        rng.shuffle(candidates)
        for pair in candidates[:2]:
            numeric_guards.append(pair)
            seen_numeric_row[row["id"]] += 1
            if len(numeric_guards) >= args.max_numeric_guards:
                break
        if len(numeric_guards) >= args.max_numeric_guards:
            break

    selected = []
    used = set()
    for row in v2_rows + forced_guards + numeric_guards:
        rid = str(row.get("id"))
        if rid in used:
            continue
        new_row = dict(row)
        new_row.setdefault("dpo_v3_policy", "guard_unanswerable_and_numeric_precision_train_only")
        selected.append(new_row)
        used.add(rid)
    rng.shuffle(selected)

    write_jsonl(args.output, selected)
    audit = make_audit_sample(selected, args.audit_size, args.seed)
    write_jsonl(args.audit_output, audit)

    quality = preference_pair_quality_report(selected)
    difficulty = pair_difficulty_report(selected)
    report = {
        "input_v2": str(args.v2),
        "input_train": str(args.train),
        "output": str(args.output),
        "audit_output": str(args.audit_output),
        "seed": args.seed,
        "rows": len(selected),
        "base_v2_rows": len(v2_rows),
        "forced_guard_rows": len(forced_guards),
        "numeric_guard_rows": len(numeric_guards),
        "source": dict(Counter(str(row.get("source")) for row in selected)),
        "reject_type": dict(Counter(str(row.get("reject_type")) for row in selected)),
        "difficulty": dict(Counter(str(row.get("difficulty")) for row in selected)),
        "answerability_type": dict(Counter(str(row.get("answerability_type")) for row in selected)),
        "numeric_guard_variant": dict(Counter(str(row.get("numeric_guard_variant")) for row in selected if row.get("numeric_guard_variant"))),
        "quality": quality,
        "difficulty_report": difficulty,
        "leakage_guard": {
            "uses_eval_rows": False,
            "train_source_split": dict(Counter(str(row.get("source_split")) for row in train_rows)),
            "post_dpo_eval_failures_used_as_training_rows": False,
            "post_dpo_eval_failures_used_as_taxonomy_only": True,
        },
        "formal_gate": {
            "status": "pending_manual_pair_audit",
            "gate_pass": False,
            "reason": "v3 adds new train-split guard pairs; audit sample must be reviewed before formal DPO training is accepted.",
        },
        "notes": [
            "Kept v2 reweighted data intact.",
            "Added train-split unanswerable forced-answer guards based on post-DPO failure taxonomy, not eval rows.",
            "Added train-split numeric precision/sign/scale guards to reduce exact-answer regressions.",
        ],
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
