"""Command-line pipeline for FinGround-QA."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from .convert import convert_financebench, convert_finqa_updated, convert_tatqa
from .dpo import make_rule_pairs, standardize_pair
from .eval import error_delta, evaluate_predictions
from .checker import check_output
from .io_utils import ensure_dir, read_json, read_jsonl, write_json, write_jsonl
from .prompts import to_eval_record, to_sft_record
from .reports import (
    answerability_report,
    categorical_report,
    evidence_quote_hit_report,
    leakage_report,
    numeric_grounding_report,
    preference_pair_quality_report,
    summarize_unified,
    table_linearization_report,
)
from .schema import validate_sft_response
from .text_utils import normalize_for_match


def prepare_data(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    out = Path(args.output_dir)
    for sub in ["data/raw", "data/unified", "data/sft", "data/eval", "data/dpo", "reports", "results", "docs"]:
        ensure_dir(out / sub)

    rows: List[Dict[str, Any]] = []
    rows.extend(convert_tatqa("train", args.tatqa_train_limit))
    rows.extend(convert_tatqa("validation", args.tatqa_val_limit))
    rows.extend(convert_tatqa("test", args.tatqa_test_limit))
    rows.extend(convert_finqa_updated("train", args.finqa_train_limit))
    rows.extend(convert_finqa_updated("valid", args.finqa_val_limit))
    rows.extend(convert_finqa_updated("test", args.finqa_test_limit))
    financebench = convert_financebench(args.financebench_limit)

    rows = dedupe_by_question(rows)
    train_source_rows = [r for r in rows if r.get("source_split") == "train"]
    heldout_source_rows = [r for r in rows if r.get("source_split") != "train"]
    rows.extend(make_unanswerable_rows(train_source_rows, args.unanswerable_train_size, "train", args.seed + 1))
    rows.extend(make_unanswerable_rows(heldout_source_rows, args.unanswerable_heldout_size, "heldout", args.seed + 2))
    write_jsonl(out / "data/unified/all_unified.jsonl", rows)
    write_jsonl(out / "data/eval/financebench_audit.jsonl", [to_eval_record(r) for r in financebench])

    eligible = [
        r
        for r in rows
        if r.get("contexts") and (r.get("evidence") or r.get("grounding_type") == "unanswerable")
    ]
    train_pool = [r for r in eligible if r.get("source_split") == "train"]
    heldout_pool = [r for r in eligible if r.get("source_split") != "train"]
    random.shuffle(train_pool)
    random.shuffle(heldout_pool)
    eval_rows = heldout_pool[: args.eval_size]
    val_rows = heldout_pool[args.eval_size : args.eval_size + args.sft_val_size]
    train_rows = train_pool[: args.sft_train_size]
    if len(eval_rows) < args.eval_size or len(val_rows) < args.sft_val_size:
        raise RuntimeError(
            f"not enough heldout rows for eval/val: eval={len(eval_rows)} val={len(val_rows)}"
        )
    if len(train_rows) < args.sft_train_size:
        raise RuntimeError(f"not enough train rows: train={len(train_rows)}")

    write_jsonl(out / "data/unified/train_unified.jsonl", train_rows)
    write_jsonl(out / "data/unified/val_unified.jsonl", val_rows)
    write_jsonl(out / "data/unified/eval_unified.jsonl", eval_rows)
    write_jsonl(out / "data/sft/sft_train.jsonl", [to_sft_record(r) for r in train_rows])
    write_jsonl(out / "data/sft/sft_val.jsonl", [to_sft_record(r) for r in val_rows])
    write_jsonl(out / "data/eval/eval.jsonl", [to_eval_record(r) for r in eval_rows])

    report_rows = train_rows + val_rows + eval_rows
    write_json(out / "reports/data_quality_report.json", summarize_unified(report_rows))
    write_json(out / "reports/evidence_quote_hit_report.json", evidence_quote_hit_report(report_rows))
    write_json(out / "reports/numeric_grounding_report.json", numeric_grounding_report(report_rows))
    write_json(out / "reports/table_linearization_report.json", table_linearization_report(report_rows))
    write_json(out / "reports/answer_type_distribution.json", categorical_report(report_rows, "answer_type"))
    write_json(out / "reports/reasoning_type_distribution.json", categorical_report(report_rows, "reasoning_type"))
    write_json(out / "reports/answerability_report.json", answerability_report(report_rows))
    write_json(out / "reports/train_eval_leakage_report.json", leakage_report(train_rows + val_rows, eval_rows))
    write_json(
        out / "reports/prepare_summary.json",
        {
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "eval_rows": len(eval_rows),
            "financebench_audit_rows": len(financebench),
            "all_unified_rows": len(rows),
        },
    )
    print(json.dumps(read_json(out / "reports/prepare_summary.json"), ensure_ascii=False, indent=2))


def validate_sft(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.file)
    errors = []
    for row in rows:
        try:
            obj = json.loads(row["output"])
        except Exception as exc:
            errors.append({"id": row.get("id"), "errors": [f"json_parse:{type(exc).__name__}"]})
            continue
        errs = validate_sft_response(obj)
        if errs:
            errors.append({"id": row.get("id"), "errors": errs})
    report = {"file": args.file, "rows": len(rows), "invalid_rows": len(errors), "sample_errors": errors[:20]}
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def build_rule_dpo(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.unified_train)
    pairs = make_rule_pairs(rows, target=args.target, seed=args.seed)
    write_jsonl(args.output, pairs)
    report = preference_pair_quality_report(pairs)
    write_json(args.report, report)
    write_json(args.difficulty_report, pair_difficulty_report(pairs))
    print(json.dumps(read_json(args.report), ensure_ascii=False, indent=2))


def audit_pairs(args: argparse.Namespace) -> None:
    pairs = read_jsonl(args.pairs)
    random.seed(args.seed)
    sample = pairs[:]
    random.shuffle(sample)
    sample = sample[: args.audit_size]
    audit = []
    for row in sample:
        audit.append(
            {
                "id": row.get("id"),
                "question": row.get("prompt", "")[:1000],
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
    write_jsonl(args.output, audit)
    print(f"wrote {args.output}")


def summarize_audit(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.audit)
    reviewed = [r for r in rows if r.get("is_chosen_better") is not None]
    denom = max(1, len(reviewed))

    def issue_rate(name: str) -> float:
        return sum(1 for r in reviewed if r.get("issue") == name) / denom

    chosen_better = sum(1 for r in reviewed if r.get("is_chosen_better") is True) / denom
    ambiguous = issue_rate("ambiguous_pair")
    chosen_bad = issue_rate("chosen_also_bad")
    too_easy = issue_rate("rejected_too_easy")
    format_bias = issue_rate("format_bias")
    chosen_lens = [int(r.get("chosen_length") or len(str(r.get("chosen", "")))) for r in rows]
    rejected_lens = [int(r.get("rejected_length") or len(str(r.get("rejected", "")))) for r in rows]
    chosen_avg = sum(chosen_lens) / max(1, len(chosen_lens))
    rejected_avg = sum(rejected_lens) / max(1, len(rejected_lens))
    length_ratio = chosen_avg / max(1, rejected_avg)
    report = {
        "audit_rows": len(rows),
        "reviewed_rows": len(reviewed),
        "is_chosen_better_rate": chosen_better,
        "ambiguous_pair_rate": ambiguous,
        "chosen_also_bad_rate": chosen_bad,
        "rejected_too_easy_rate": too_easy,
        "format_bias_rate": format_bias,
        "chosen_avg_len": chosen_avg,
        "rejected_avg_len": rejected_avg,
        "length_ratio": length_ratio,
        "passes_gate": {
            "enough_reviewed": len(reviewed) >= args.min_reviewed,
            "is_chosen_better_ge_85pct": chosen_better >= 0.85,
            "ambiguous_pair_le_10pct": ambiguous <= 0.10,
            "chosen_also_bad_le_10pct": chosen_bad <= 0.10,
            "rejected_too_easy_le_25pct": too_easy <= 0.25,
            "format_bias_le_15pct": format_bias <= 0.15,
            "length_ratio_0_5_to_2_0": 0.5 <= length_ratio <= 2.0,
        },
    }
    report["gate_pass"] = all(report["passes_gate"].values())
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def mine_rejected(args: argparse.Namespace) -> None:
    source_rows = read_jsonl(args.source_rows)
    preds = read_jsonl(args.predictions)
    source_by_id = {row["id"]: row for row in source_rows}
    pairs = []
    counts: Dict[str, int] = {}
    for pred in preds:
        row_id = pred.get("id") or pred.get("row_id")
        if row_id not in source_by_id:
            continue
        row = source_by_id[row_id]
        text = pred.get("prediction") or pred.get("output") or pred.get("response") or ""
        check = check_output(row, text)
        reject_type = classify_reject_type(check)
        if not reject_type:
            continue
        difficulty = "hard" if reject_type in {"unsupported_claim", "wrong_citation", "fabricated_number", "calculation_error", "over_refusal", "forced_answer"} else "medium"
        pair = {
            "id": f"{row_id}_model_mined_{reject_type}_{len(pairs)}",
            "prompt": to_eval_record(row)["prompt"],
            "chosen": to_sft_record(row)["output"],
            "rejected": text,
            "reject_type": reject_type,
            "source": "model_mined",
            "difficulty": difficulty,
            "row_id": row_id,
            "checker": check,
        }
        pairs.append(standardize_pair(row, pair))
        counts[reject_type] = counts.get(reject_type, 0) + 1
        if len(pairs) >= args.max_pairs:
            break
    write_jsonl(args.output, pairs)
    write_json(args.report, {"rows": len(pairs), "reject_type": counts, "quality": preference_pair_quality_report(pairs)})
    print(json.dumps(read_json(args.report), ensure_ascii=False, indent=2))


def mix_dpo(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    rule_pairs = read_jsonl(args.rule_pairs) if Path(args.rule_pairs).exists() else []
    mined_pairs = read_jsonl(args.mined_pairs) if Path(args.mined_pairs).exists() else []
    rng.shuffle(rule_pairs)
    rng.shuffle(mined_pairs)
    mined_quota = min(len(mined_pairs), int(args.target * args.mined_ratio))
    mixed: List[Dict[str, Any]] = []
    used = set()
    target_group = {"fabricated_number", "wrong_citation", "unsupported_claim", "calculation_error"}

    def count_reject(*names: str) -> int:
        return sum(1 for p in mixed if p.get("reject_type") in names)

    def add_from(candidates: List[Dict[str, Any]], predicate: Any, limit: int | None = None) -> None:
        nonlocal mixed
        added = 0
        for pair in candidates:
            if len(mixed) >= args.target or (limit is not None and added >= limit):
                break
            pair_id = pair.get("id")
            if pair_id in used or not predicate(pair):
                continue
            mixed.append(pair)
            used.add(pair_id)
            added += 1

    add_from(mined_pairs, lambda _: True, mined_quota)
    over_min = int(args.target * 0.15)
    target_min = int(args.target * 0.50)
    format_cap = int(args.target * 0.10)

    add_from(
        rule_pairs,
        lambda p: p.get("reject_type") in {"over_refusal", "generic_answer"},
        max(0, over_min - count_reject("over_refusal", "generic_answer")),
    )
    add_from(
        rule_pairs,
        lambda p: p.get("reject_type") in target_group,
        max(0, target_min - count_reject(*target_group)),
    )
    add_from(rule_pairs, lambda p: p.get("reject_type") == "forced_answer")
    add_from(rule_pairs, lambda p: p.get("reject_type") == "missing_evidence")
    add_from(rule_pairs, lambda p: p.get("reject_type") in target_group)
    add_from(
        rule_pairs,
        lambda p: p.get("reject_type") == "wrong_format",
        max(0, format_cap - count_reject("wrong_format")),
    )
    if len(mixed) < args.target:
        add_from(rule_pairs + mined_pairs, lambda _: True)
    write_jsonl(args.output, mixed)
    quality = preference_pair_quality_report(mixed)
    difficulty = pair_difficulty_report(mixed)
    write_json(args.quality_report, quality)
    write_json(args.difficulty_report, difficulty)
    print(json.dumps({"rows": len(mixed), "quality": quality, "difficulty": difficulty}, ensure_ascii=False, indent=2))


def build_mining_eval(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.input)
    candidates = [
        r
        for r in rows
        if r.get("contexts") and (r.get("evidence") or row_answerability_type(r) == "unanswerable")
    ]
    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    if args.limit > 0:
        candidates = candidates[: args.limit]
    write_jsonl(args.output, [to_eval_record(r) for r in candidates])
    report = {
        "rows": len(candidates),
        "input": args.input,
        "output": args.output,
        "dataset": dict(Counter(r.get("dataset", "unknown") for r in candidates)),
        "grounding_type": dict(Counter(r.get("grounding_type", "unknown") for r in candidates)),
        "answer_type": dict(Counter(r.get("answer_type", "unknown") for r in candidates)),
        "answerability_type": dict(Counter(row_answerability_type(r) for r in candidates)),
        "source_split": dict(Counter(r.get("source_split", "unknown") for r in candidates)),
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def classify_reject_type(check: Dict[str, Any]) -> str | None:
    if not check.get("json_valid") or not check.get("schema_pass"):
        return "wrong_format"
    if check.get("wrong_citation"):
        return "wrong_citation"
    if check.get("forced_answer"):
        return "forced_answer"
    if check.get("fabricated_number"):
        return "fabricated_number"
    if check.get("calculation_error"):
        return "calculation_error"
    if check.get("missing_evidence"):
        return "missing_evidence"
    if check.get("unsupported_claim"):
        return "unsupported_claim"
    if check.get("over_refusal"):
        return "over_refusal"
    if check.get("generic_answer"):
        return "generic_answer"
    return None


def build_answerability_eval(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.input)
    rng = random.Random(args.seed)
    answerable = [r for r in rows if row_answerability_type(r) == "answerable" and r.get("contexts") and r.get("evidence")]
    unanswerable = [r for r in rows if row_answerability_type(r) == "unanswerable" and r.get("contexts")]
    heldout_answerable = [r for r in answerable if r.get("source_split") != "train"]
    heldout_unanswerable = [r for r in unanswerable if r.get("source_split") != "train"]
    if len(heldout_answerable) >= args.answerable_size:
        answerable = heldout_answerable
    if len(heldout_unanswerable) >= args.unanswerable_size:
        unanswerable = heldout_unanswerable
    rng.shuffle(answerable)
    rng.shuffle(unanswerable)
    selected_answerable = answerable[: args.answerable_size]
    selected_unanswerable = unanswerable[: args.unanswerable_size]
    selected = selected_answerable + selected_unanswerable
    rng.shuffle(selected)
    write_jsonl(args.output, [to_eval_record(r) for r in selected])
    report = {
        "rows": len(selected),
        "answerable_rows": len(selected_answerable),
        "unanswerable_rows": len(selected_unanswerable),
        "requested_answerable": args.answerable_size,
        "requested_unanswerable": args.unanswerable_size,
        "dataset": dict(Counter(r.get("dataset", "unknown") for r in selected)),
        "grounding_type": dict(Counter(r.get("grounding_type", "unknown") for r in selected)),
        "source_split": dict(Counter(r.get("source_split", "unknown") for r in selected)),
        "warning": "" if len(selected_unanswerable) >= args.unanswerable_size else "not enough unanswerable rows; wrote all available rows",
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def data_difficulty_audit(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.input)
    candidates = [r for r in rows if r.get("contexts") and (r.get("evidence") or row_answerability_type(r) == "unanswerable")]
    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    audit_rows = [difficulty_audit_row(r) for r in candidates[: args.limit]]
    write_jsonl(args.output, audit_rows)
    report = difficulty_report(audit_rows)
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def row_answerability_type(row: Dict[str, Any]) -> str:
    if row.get("answerability_type") in {"answerable", "unanswerable"}:
        return str(row["answerability_type"])
    if row.get("grounding_type") == "unanswerable" or row.get("answer_type") == "unanswerable":
        return "unanswerable"
    return "answerable"


def difficulty_audit_row(row: Dict[str, Any]) -> Dict[str, Any]:
    context_text = " ".join(str(c.get("text", "")) for c in row.get("contexts", []))
    normalized_context = normalize_for_match(context_text)
    normalized_answer = normalize_for_match(str(row.get("answer", "")))
    answer_copied = bool(normalized_answer and normalized_answer in normalized_context)
    requires_numeric = (
        row.get("answer_type") == "number"
        or row.get("reasoning_type") == "calculation"
        or row.get("grounding_type") in {"numeric_grounded", "calculation_hard"}
    )
    context_count = len(row.get("contexts", []))
    evidence_count = len(row.get("evidence", []))
    is_unanswerable = row_answerability_type(row) == "unanswerable"
    return {
        "id": row.get("id"),
        "dataset": row.get("dataset"),
        "question": row.get("question"),
        "answer": row.get("answer"),
        "answer_type": row.get("answer_type"),
        "reasoning_type": row.get("reasoning_type"),
        "grounding_type": row.get("grounding_type"),
        "answerability_type": row_answerability_type(row),
        "context_count": context_count,
        "evidence_count": evidence_count,
        "proxy_flags": {
            "is_trivial_lookup": bool(answer_copied and context_count <= 2 and row.get("reasoning_type") == "lookup"),
            "requires_numeric_reasoning": bool(requires_numeric),
            "has_distractor_context": context_count >= 3,
            "citation_confusable": context_count >= 2 and evidence_count > 0,
            "answer_directly_copied": answer_copied,
            "is_good_grounded_sample": bool((evidence_count > 0 or is_unanswerable) and context_count > 0),
        },
        "manual_audit": {
            "is_trivial_lookup": None,
            "requires_numeric_reasoning": None,
            "has_distractor_context": None,
            "citation_confusable": None,
            "answer_directly_copied": None,
            "is_good_grounded_sample": None,
            "comment": "",
        },
    }


def difficulty_report(audit_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows = len(audit_rows)
    flag_counts: Counter[str] = Counter()
    for row in audit_rows:
        for name, value in row.get("proxy_flags", {}).items():
            if value:
                flag_counts[name] += 1
    return {
        "rows": rows,
        "dataset": dict(Counter(r.get("dataset", "unknown") for r in audit_rows)),
        "grounding_type": dict(Counter(r.get("grounding_type", "unknown") for r in audit_rows)),
        "answer_type": dict(Counter(r.get("answer_type", "unknown") for r in audit_rows)),
        "answerability_type": dict(Counter(r.get("answerability_type", "unknown") for r in audit_rows)),
        "proxy_flag_counts": dict(flag_counts),
        "proxy_flag_rates": {name: count / max(1, rows) for name, count in flag_counts.items()},
        "notes": "Proxy flags are automatic diagnostics for manual audit, not final human labels.",
    }


def evaluate(args: argparse.Namespace) -> None:
    eval_rows = read_jsonl(args.eval_file)
    preds = read_jsonl(args.predictions)
    metrics, scored = evaluate_predictions(eval_rows, preds)
    write_json(args.metrics, metrics)
    if args.scored:
        write_jsonl(args.scored, scored)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def make_error_delta(args: argparse.Namespace) -> None:
    metrics = {name: read_json(path) for name, path in zip(args.names, args.metrics)}
    write_json(args.output, error_delta(metrics))
    print(f"wrote {args.output}")


def export_badcases(args: argparse.Namespace) -> None:
    all_cases: List[Dict[str, Any]] = []
    for name, path in zip(args.names, args.scored):
        if not Path(path).exists():
            continue
        for row in read_jsonl(path):
            ev = row.get("eval", {})
            err = classify_badcase_error(ev)
            if not err:
                continue
            all_cases.append(
                {
                    "source": name,
                    "id": row.get("id"),
                    "dataset": row.get("dataset"),
                    "error_type": err,
                    "prediction": row.get("prediction") or row.get("output") or row.get("response"),
                    "eval": ev,
                    "manual_label": "",
                    "comment": "",
                }
            )
    priority = {
        "wrong_citation": 0,
        "unsupported_claim": 1,
        "forced_answer": 2,
        "over_refusal": 3,
        "generic_answer": 4,
        "schema_error": 5,
        "format_error": 6,
    }
    all_cases.sort(key=lambda x: (priority.get(x["error_type"], 99), x["source"], str(x.get("id"))))
    selected = all_cases[: args.limit]
    write_jsonl(args.output, selected)
    print(f"wrote {args.output} ({len(selected)} rows)")


def classify_badcase_error(ev: Dict[str, Any]) -> str | None:
    if not ev.get("json_valid", True):
        return "format_error"
    if not ev.get("schema_pass", True):
        return "schema_error"
    for key in [
        "missing_evidence",
        "wrong_citation",
        "unsupported_claim",
        "fabricated_number",
        "calculation_error",
        "forced_answer",
        "over_refusal",
        "generic_answer",
    ]:
        if ev.get(key):
            return key
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare-data")
    p.add_argument("--output-dir", default=".")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sft-train-size", type=int, default=5000)
    p.add_argument("--sft-val-size", type=int, default=500)
    p.add_argument("--eval-size", type=int, default=400)
    p.add_argument("--tatqa-train-limit", type=int, default=None)
    p.add_argument("--tatqa-val-limit", type=int, default=None)
    p.add_argument("--tatqa-test-limit", type=int, default=None)
    p.add_argument("--finqa-train-limit", type=int, default=None)
    p.add_argument("--finqa-val-limit", type=int, default=None)
    p.add_argument("--finqa-test-limit", type=int, default=None)
    p.add_argument("--financebench-limit", type=int, default=None)
    p.add_argument("--unanswerable-train-size", type=int, default=300)
    p.add_argument("--unanswerable-heldout-size", type=int, default=100)
    p.set_defaults(func=prepare_data)

    p = sub.add_parser("validate-sft")
    p.add_argument("--file", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=validate_sft)

    p = sub.add_parser("build-rule-dpo")
    p.add_argument("--unified-train", default="data/unified/train_unified.jsonl")
    p.add_argument("--target", type=int, default=600)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="data/dpo/rule_dpo_pairs.jsonl")
    p.add_argument("--report", default="reports/preference_pair_quality_report.json")
    p.add_argument("--difficulty-report", default="reports/pair_difficulty_report.json")
    p.set_defaults(func=build_rule_dpo)

    p = sub.add_parser("audit-pairs")
    p.add_argument("--pairs", default="data/dpo/rule_dpo_pairs.jsonl")
    p.add_argument("--audit-size", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="results/preference_pair_audit_100.jsonl")
    p.set_defaults(func=audit_pairs)

    p = sub.add_parser("summarize-audit")
    p.add_argument("--audit", default="results/preference_pair_audit_100.jsonl")
    p.add_argument("--min-reviewed", type=int, default=100)
    p.add_argument("--output", default="reports/preference_pair_audit_report.json")
    p.set_defaults(func=summarize_audit)

    p = sub.add_parser("mine-rejected")
    p.add_argument("--source-rows", default="data/unified/eval_unified.jsonl")
    p.add_argument("--predictions", required=True)
    p.add_argument("--max-pairs", type=int, default=500)
    p.add_argument("--output", default="data/dpo/model_mined_pairs.jsonl")
    p.add_argument("--report", default="reports/model_mined_rejected_report.json")
    p.set_defaults(func=mine_rejected)

    p = sub.add_parser("mix-dpo")
    p.add_argument("--rule-pairs", default="data/dpo/rule_dpo_pairs.jsonl")
    p.add_argument("--mined-pairs", default="data/dpo/model_mined_pairs.jsonl")
    p.add_argument("--target", type=int, default=1000)
    p.add_argument("--mined-ratio", type=float, default=0.40)
    p.add_argument("--hard-ratio", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="data/dpo/dpo_balanced_v1.jsonl")
    p.add_argument("--quality-report", default="reports/preference_pair_quality_report.json")
    p.add_argument("--difficulty-report", default="reports/pair_difficulty_report.json")
    p.set_defaults(func=mix_dpo)

    p = sub.add_parser("build-mining-eval")
    p.add_argument("--input", default="data/unified/train_unified.jsonl")
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="data/eval/train_mining_eval.jsonl")
    p.add_argument("--report", default="reports/train_mining_eval_report.json")
    p.set_defaults(func=build_mining_eval)

    p = sub.add_parser("build-answerability-eval")
    p.add_argument("--input", default="data/unified/all_unified.jsonl")
    p.add_argument("--answerable-size", type=int, default=100)
    p.add_argument("--unanswerable-size", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="data/eval/answerability_eval.jsonl")
    p.add_argument("--report", default="reports/answerability_eval_report.json")
    p.set_defaults(func=build_answerability_eval)

    p = sub.add_parser("data-difficulty-audit")
    p.add_argument("--input", default="data/unified/all_unified.jsonl")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="results/data_difficulty_audit_100.jsonl")
    p.add_argument("--report", default="reports/data_difficulty_report.json")
    p.set_defaults(func=data_difficulty_audit)

    p = sub.add_parser("evaluate")
    p.add_argument("--eval-file", default="data/eval/eval.jsonl")
    p.add_argument("--predictions", required=True)
    p.add_argument("--metrics", required=True)
    p.add_argument("--scored")
    p.set_defaults(func=evaluate)

    p = sub.add_parser("error-delta")
    p.add_argument("--names", nargs="+", required=True)
    p.add_argument("--metrics", nargs="+", required=True)
    p.add_argument("--output", default="reports/error_delta_report.json")
    p.set_defaults(func=make_error_delta)

    p = sub.add_parser("export-badcases")
    p.add_argument("--names", nargs="+", required=True)
    p.add_argument("--scored", nargs="+", required=True)
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--output", default="results/badcase_50.jsonl")
    p.set_defaults(func=export_badcases)

    args = parser.parse_args()
    args.func(args)


def dedupe_by_question(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    kept = []
    for row in rows:
        key = (row.get("dataset"), normalize_for_match(row.get("question", "")))
        if not key[1] or key in seen:
            continue
        seen.add(key)
        kept.append(row)
    return kept


def make_unanswerable_rows(rows: List[Dict[str, Any]], count: int, split: str, seed: int) -> List[Dict[str, Any]]:
    if count <= 0 or len(rows) < 2:
        return []
    rng = random.Random(seed)
    questions = rows[:]
    contexts = rows[:]
    rng.shuffle(questions)
    rng.shuffle(contexts)
    out = []
    for idx, qrow in enumerate(questions):
        if len(out) >= count:
            break
        ctxrow = contexts[idx % len(contexts)]
        if normalize_for_match(qrow.get("question", "")) == normalize_for_match(ctxrow.get("question", "")):
            continue
        out.append(
            {
                "id": f"synthetic_unanswerable_{split}_{idx}",
                "dataset": "SyntheticUnanswerable",
                "source_split": "train" if split == "train" else "validation",
                "question": qrow.get("question", ""),
                "contexts": ctxrow.get("contexts", [])[:3],
                "answer": "Insufficient evidence to answer from the provided contexts.",
                "evidence": [],
                "answer_type": "unanswerable",
                "reasoning_type": "lookup",
                "grounding_type": "unanswerable",
                "gold_program": "",
                "meta": {
                    "question_source_id": qrow.get("id"),
                    "context_source_id": ctxrow.get("id"),
                    "synthetic": True,
                },
            }
        )
    return out


def pair_difficulty_report(pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    from collections import Counter

    total = len(pairs)
    reject_counts = Counter(p.get("reject_type") for p in pairs)
    hard_count = sum(1 for p in pairs if p.get("difficulty") == "hard")
    target_group = sum(
        reject_counts.get(name, 0)
        for name in ["fabricated_number", "wrong_citation", "unsupported_claim", "calculation_error"]
    )
    format_count = reject_counts.get("wrong_format", 0)
    over_generic = reject_counts.get("over_refusal", 0) + reject_counts.get("generic_answer", 0)
    answerability_counts = Counter(p.get("answerability_type", "unknown") for p in pairs)
    chosen_lens = [int(p.get("chosen_length") or len(str(p.get("chosen", "")))) for p in pairs]
    rejected_lens = [int(p.get("rejected_length") or len(str(p.get("rejected", "")))) for p in pairs]
    chosen_avg = sum(chosen_lens) / max(1, len(chosen_lens))
    rejected_avg = sum(rejected_lens) / max(1, len(rejected_lens))
    length_ratio = chosen_avg / max(1, rejected_avg)
    return {
        "rows": total,
        "reject_type": dict(reject_counts),
        "source": dict(Counter(p.get("source") for p in pairs)),
        "difficulty": dict(Counter(p.get("difficulty") for p in pairs)),
        "answerability_type": dict(answerability_counts),
        "hard_rejected_rate": hard_count / max(1, total),
        "numeric_wrongcitation_unsupported_rate": target_group / max(1, total),
        "format_error_rate": format_count / max(1, total),
        "over_refusal_generic_rate": over_generic / max(1, total),
        "forced_answer_rate": reject_counts.get("forced_answer", 0) / max(1, total),
        "length_bias_report": {
            "chosen_avg_len": chosen_avg,
            "rejected_avg_len": rejected_avg,
            "length_ratio": length_ratio,
            "length_ratio_ok": 0.5 <= length_ratio <= 2.0,
        },
        "passes_planned_constraints": {
            "format_error_le_10pct": format_count / max(1, total) <= 0.10,
            "over_refusal_generic_ge_15pct": over_generic / max(1, total) >= 0.15,
            "numeric_wrongcitation_unsupported_ge_50pct": target_group / max(1, total) >= 0.50,
            "hard_rejected_ge_25pct": hard_count / max(1, total) >= 0.25,
            "length_ratio_0_5_to_2_0": 0.5 <= length_ratio <= 2.0,
        },
    }


if __name__ == "__main__":
    main()
