"""Data audit utilities for FinGround-QA.

These audits are intentionally lightweight and deterministic. They produce
proxy diagnostics for data quality; they are not substitutes for manual audit.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Sequence

from .io_utils import ensure_dir, read_jsonl, write_json, write_jsonl
from .text_utils import normalize_for_match


LEAK_MARKERS = [
    r"\breasoning steps?\s*:",
    r"\bprogram(?:\s*\(nested\))?\s*:",
    r"\banswer\s*:",
    r"\bderivation\s*:",
    r"\btarget\s*:",
    r"\bgold\s*:",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--audit-size", type=int, default=100)
    args = parser.parse_args()
    root = Path(args.root)
    ensure_dir(root / "reports")
    ensure_dir(root / "results")

    data = load_project_data(root)
    write_json(root / "reports/leakage_deep_check_report.json", leakage_deep_check(data))
    write_json(root / "reports/eval_difficulty_report.json", eval_difficulty_report(data["eval_unified"]))
    write_json(root / "reports/split_distribution_report.json", split_distribution_report(data))
    numeric_report, numeric_audit = numeric_difficulty_report(data, args.audit_size)
    write_json(root / "reports/numeric_difficulty_report.json", numeric_report)
    write_jsonl(root / "results/numeric_audit_100.jsonl", numeric_audit)
    unanswerable_report, unanswerable_audit = unanswerable_quality_report(data, args.audit_size)
    write_json(root / "reports/unanswerable_quality_report.json", unanswerable_report)
    write_jsonl(root / "results/unanswerable_audit_100.jsonl", unanswerable_audit)
    write_json(root / "reports/rule_dpo_artifact_report.json", rule_dpo_artifact_report(data["rule_dpo"]))
    write_json(root / "reports/financebench_audit_report.json", financebench_audit_report(data))
    print(json.dumps({"status": "ok", "reports_written": 8, "audit_files_written": 2}, indent=2))


def load_project_data(root: Path) -> Dict[str, List[Dict[str, Any]]]:
    files = {
        "train_unified": "data/unified/train_unified.jsonl",
        "val_unified": "data/unified/val_unified.jsonl",
        "eval_unified": "data/unified/eval_unified.jsonl",
        "sft_train": "data/sft/sft_train.jsonl",
        "sft_val": "data/sft/sft_val.jsonl",
        "eval": "data/eval/eval.jsonl",
        "answerability_eval": "data/eval/answerability_eval.jsonl",
        "financebench": "data/eval/financebench_audit.jsonl",
        "rule_dpo": "data/dpo/rule_dpo_pairs.jsonl",
    }
    data: Dict[str, List[Dict[str, Any]]] = {}
    for key, rel in files.items():
        path = root / rel
        data[key] = read_jsonl(path) if path.exists() else []
    return data


def leakage_deep_check(data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    unified = data["train_unified"] + data["val_unified"] + data["eval_unified"]
    by_dataset: Dict[str, Dict[str, Any]] = {}
    for dataset, rows in group_rows(unified, "dataset").items():
        marker_counts: Counter[str] = Counter()
        marker_examples: List[Dict[str, Any]] = []
        suspicious_fields: Counter[str] = Counter()
        nonempty_gold_program = 0
        for row in rows:
            if row.get("gold_program"):
                nonempty_gold_program += 1
            for key in row:
                if key.lower() in {"reasoning_steps", "program", "derivation", "target", "gold"}:
                    suspicious_fields[key] += 1
            for ctx in row.get("contexts", []):
                text = str(ctx.get("text", ""))
                for pattern in LEAK_MARKERS:
                    if re.search(pattern, text, flags=re.I):
                        marker_counts[pattern] += 1
                        if len(marker_examples) < 20:
                            marker_examples.append(
                                {
                                    "id": row.get("id"),
                                    "dataset": dataset,
                                    "chunk_id": ctx.get("chunk_id"),
                                    "pattern": pattern,
                                    "text_preview": text[:400],
                                }
                            )
        by_dataset[dataset] = {
            "rows": len(rows),
            "context_marker_counts": dict(marker_counts),
            "marker_examples": marker_examples,
            "suspicious_extra_field_counts": dict(suspicious_fields),
            "nonempty_gold_program_rows": nonempty_gold_program,
        }

    train = data["train_unified"]
    val = data["val_unified"]
    eval_rows = data["eval_unified"]
    train_val = train + val
    overlap = {
        "train_vs_eval": split_overlap(train, eval_rows),
        "train_val_vs_eval": split_overlap(train_val, eval_rows),
    }
    return {
        "summary": {
            "train_rows": len(train),
            "val_rows": len(val),
            "eval_rows": len(eval_rows),
            "deep_leakage_risk": classify_leakage_risk(by_dataset, overlap),
        },
        "by_dataset": by_dataset,
        "split_overlap": overlap,
        "notes": [
            "Top-level answer/evidence fields are expected labels, not leakage by themselves.",
            "Context marker checks are conservative and may flag normal business text such as 'program:'.",
        ],
    }


def classify_leakage_risk(by_dataset: Dict[str, Dict[str, Any]], overlap: Dict[str, Any]) -> str:
    nonempty_gold = sum(v["nonempty_gold_program_rows"] for v in by_dataset.values())
    exact = overlap["train_val_vs_eval"]["exact_question_overlap"]
    norm = overlap["train_val_vs_eval"]["normalized_question_overlap"]
    if nonempty_gold or exact or norm:
        return "high"
    marker_total = sum(sum(v["context_marker_counts"].values()) for v in by_dataset.values())
    return "medium" if marker_total else "low"


def split_overlap(left: List[Dict[str, Any]], right: List[Dict[str, Any]]) -> Dict[str, Any]:
    left_exact = {str(r.get("question", "")) for r in left}
    right_exact = {str(r.get("question", "")) for r in right}
    left_norm = {normalize_for_match(r.get("question", "")) for r in left}
    right_norm = {normalize_for_match(r.get("question", "")) for r in right}
    left_doc = collect_context_ids(left, "doc_id")
    right_doc = collect_context_ids(right, "doc_id")
    left_table = collect_context_ids(left, "table_id")
    right_table = collect_context_ids(right, "table_id")
    left_chunk = collect_context_ids(left, "chunk_id")
    right_chunk = collect_context_ids(right, "chunk_id")
    return {
        "exact_question_overlap": len(left_exact & right_exact),
        "normalized_question_overlap": len(left_norm & right_norm),
        "same_doc_id_overlap": len(left_doc & right_doc),
        "same_table_id_overlap": len(left_table & right_table),
        "same_chunk_id_overlap": len(left_chunk & right_chunk),
        "sample_overlaps": {
            "questions": sorted(list((left_norm & right_norm) - {""}))[:10],
            "doc_ids": sorted(list((left_doc & right_doc) - {""}))[:10],
            "table_ids": sorted(list((left_table & right_table) - {""}))[:10],
            "chunk_ids": sorted(list((left_chunk & right_chunk) - {""}))[:10],
        },
    }


def collect_context_ids(rows: Iterable[Dict[str, Any]], key: str) -> set[str]:
    out = set()
    for row in rows:
        for ctx in row.get("contexts", []):
            value = ctx.get(key)
            if value:
                out.add(str(value))
    return out


def eval_difficulty_report(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    features = [row_difficulty_features(row) for row in rows]
    return {
        "rows": len(rows),
        "grounding_type": dict(Counter(r.get("grounding_type", "unknown") for r in rows)),
        "answer_type": dict(Counter(r.get("answer_type", "unknown") for r in rows)),
        "reasoning_type": dict(Counter(r.get("reasoning_type", "unknown") for r in rows)),
        "answer_exact_in_context_rate": rate(features, "answer_exact_in_context"),
        "answer_exact_in_evidence_rate": rate(features, "answer_exact_in_evidence"),
        "numeric_answer_directly_appears_rate": rate(features, "numeric_answer_directly_appears", denom_key="is_numeric"),
        "avg_context_chunks": safe_mean([f["context_chunks"] for f in features]),
        "avg_context_tokens": safe_mean([f["context_tokens"] for f in features]),
        "citation_confusable_proxy_rate": rate(features, "citation_confusable_proxy"),
        "trivial_lookup_proxy_rate": rate(features, "trivial_lookup_proxy"),
        "non_trivial_grounded_proxy_rate": rate(features, "non_trivial_grounded_proxy"),
        "notes": "Proxy difficulty metrics are heuristics for audit and stratified reporting.",
    }


def split_distribution_report(data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    splits = {
        "sft_train": data["sft_train"],
        "sft_val": data["sft_val"],
        "eval": data["eval"],
    }
    report = {name: split_summary(rows, sft=name.startswith("sft")) for name, rows in splits.items()}
    train = report["sft_train"]
    ev = report["eval"]
    report["diagnostics"] = {
        "eval_vs_train_prompt_mean_ratio": ratio(ev["prompt_length"]["mean"], train["prompt_length"]["mean"]),
        "eval_vs_train_context_mean_ratio": ratio(ev["context_length"]["mean"], train["context_length"]["mean"]),
        "eval_vs_train_answer_mean_ratio": ratio(ev["answer_length"]["mean"], train["answer_length"]["mean"]),
        "eval_direct_grounded_rate": distribution_rate(ev["grounding_type"], "direct_grounded"),
        "train_direct_grounded_rate": distribution_rate(train["grounding_type"], "direct_grounded"),
        "eval_appears_simpler_warning": eval_simpler_warning(train, ev),
    }
    return report


def split_summary(rows: List[Dict[str, Any]], sft: bool) -> Dict[str, Any]:
    prompt_lengths = []
    context_lengths = []
    answer_lengths = []
    evidence_lengths = []
    dataset = Counter()
    grounding = Counter()
    answer_type = Counter()
    reasoning = Counter()
    for row in rows:
        if sft:
            meta = row.get("meta", {})
            dataset[row.get("dataset", "unknown")] += 1
            grounding[meta.get("grounding_type", "unknown")] += 1
            answer_type[meta.get("answer_type", "unknown")] += 1
            reasoning[meta.get("reasoning_type", "unknown")] += 1
            prompt = row.get("instruction", "")
            output = parse_json_obj(row.get("output", ""))
            answer = str(output.get("answer", "")) if isinstance(output, dict) else ""
            evidence = output.get("evidence", []) if isinstance(output, dict) else []
            context_text = prompt
        else:
            dataset[row.get("dataset", "unknown")] += 1
            grounding[row.get("grounding_type", "unknown")] += 1
            answer_type[row.get("answer_type", "unknown")] += 1
            reasoning[row.get("reasoning_type", "unknown")] += 1
            prompt = row.get("prompt", "")
            answer = str(row.get("gold_answer", ""))
            evidence = row.get("evidence", [])
            context_text = " ".join(c.get("text", "") for c in row.get("contexts", []))
        prompt_lengths.append(len(prompt))
        context_lengths.append(len(context_text))
        answer_lengths.append(len(answer))
        evidence_lengths.extend([len(str(e.get("quote", ""))) for e in evidence])
    return {
        "rows": len(rows),
        "dataset": dict(dataset),
        "grounding_type": dict(grounding),
        "answer_type": dict(answer_type),
        "reasoning_type": dict(reasoning),
        "prompt_length": numeric_summary(prompt_lengths),
        "context_length": numeric_summary(context_lengths),
        "answer_length": numeric_summary(answer_lengths),
        "evidence_quote_length": numeric_summary(evidence_lengths),
    }


def numeric_difficulty_report(data: Dict[str, List[Dict[str, Any]]], audit_size: int) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    rows = [
        r
        for r in data["train_unified"] + data["val_unified"] + data["eval_unified"]
        if r.get("answer_type") == "number" or r.get("grounding_type") in {"numeric_grounded", "calculation_hard"}
    ]
    features = [numeric_features(r) for r in rows]
    audit = []
    for row, feat in zip(rows[:audit_size], features[:audit_size]):
        audit.append(
            {
                "id": row.get("id"),
                "dataset": row.get("dataset"),
                "question": row.get("question"),
                "answer": row.get("answer"),
                "grounding_type": row.get("grounding_type"),
                "reasoning_type": row.get("reasoning_type"),
                "proxy": feat,
                "contexts_preview": [c.get("text", "")[:500] for c in row.get("contexts", [])[:2]],
                "manual_audit": {
                    "requires_real_calculation": None,
                    "is_direct_lookup": None,
                    "has_multiple_candidate_numbers": None,
                    "has_year_alignment": None,
                    "has_unit_alignment": None,
                    "is_good_numeric_sample": None,
                    "issue": "",
                    "notes": "",
                },
            }
        )
    report = {
        "rows": len(rows),
        "dataset": dict(Counter(r.get("dataset", "unknown") for r in rows)),
        "grounding_type": dict(Counter(r.get("grounding_type", "unknown") for r in rows)),
        "direct_lookup_proxy_rate": rate(features, "direct_lookup_proxy"),
        "simple_add_sub_proxy_rate": rate(features, "simple_add_sub_proxy"),
        "percentage_change_proxy_rate": rate(features, "percentage_change_proxy"),
        "year_alignment_proxy_rate": rate(features, "year_alignment_proxy"),
        "unit_alignment_proxy_rate": rate(features, "unit_alignment_proxy"),
        "multiple_candidate_numbers_rate": rate(features, "multiple_candidate_numbers"),
        "numeric_answer_directly_appears_rate": rate(features, "numeric_answer_directly_appears"),
        "notes": "Numeric categories are heuristic proxies; audit JSONL is for human verification.",
    }
    return report, audit


def unanswerable_quality_report(data: Dict[str, List[Dict[str, Any]]], audit_size: int) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    unans = [
        r
        for r in data["train_unified"] + data["val_unified"] + data["eval_unified"]
        if r.get("grounding_type") == "unanswerable" or r.get("answer_type") == "unanswerable"
    ]
    ans = [
        r
        for r in data["train_unified"] + data["val_unified"] + data["eval_unified"]
        if not (r.get("grounding_type") == "unanswerable" or r.get("answer_type") == "unanswerable")
    ]
    features = [unanswerable_features(r) for r in unans]
    answerable_prompt_lens = [len(r.get("question", "")) + sum(len(c.get("text", "")) for c in r.get("contexts", [])) for r in ans]
    unanswerable_prompt_lens = [len(r.get("question", "")) + sum(len(c.get("text", "")) for c in r.get("contexts", [])) for r in unans]
    audit = []
    for row, feat in zip(unans[:audit_size], features[:audit_size]):
        audit.append(
            {
                "id": row.get("id"),
                "question": row.get("question"),
                "answer": row.get("answer"),
                "source_ids": row.get("meta", {}),
                "proxy": feat,
                "contexts_preview": [c.get("text", "")[:500] for c in row.get("contexts", [])[:3]],
                "manual_audit": {
                    "is_truly_unanswerable": None,
                    "is_too_template_like": None,
                    "has_partial_evidence": None,
                    "should_refuse": None,
                    "bad_sample_reason": "",
                    "notes": "",
                },
            }
        )
    report = {
        "rows": len(unans),
        "template_answer_rate": sum(1 for r in unans if "insufficient evidence" in str(r.get("answer", "")).lower()) / max(1, len(unans)),
        "question_template_prefix_distribution": dict(Counter(first_words(r.get("question", ""), 4) for r in unans).most_common(20)),
        "partial_evidence_proxy_rate": rate(features, "partial_evidence_proxy"),
        "question_overlap_with_context_proxy_rate": rate(features, "question_overlap_with_context_proxy"),
        "answerable_prompt_length": numeric_summary(answerable_prompt_lens),
        "unanswerable_prompt_length": numeric_summary(unanswerable_prompt_lens),
        "unanswerable_vs_answerable_prompt_mean_ratio": ratio(safe_mean(unanswerable_prompt_lens), safe_mean(answerable_prompt_lens)),
        "notes": "Synthetic unanswerable quality needs manual audit; proxy rates estimate template and partial-evidence risk.",
    }
    return report, audit


def rule_dpo_artifact_report(pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    parsed = [(p, parse_json_obj(p.get("chosen", "")), parse_json_obj(p.get("rejected", ""))) for p in pairs]
    chosen_lens = [int(p.get("chosen_length") or len(str(p.get("chosen", "")))) for p, _, _ in parsed]
    rejected_lens = [int(p.get("rejected_length") or len(str(p.get("rejected", "")))) for p, _, _ in parsed]
    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for p, ch, rj in parsed:
        by_type[p.get("reject_type", "unknown")].append({"pair": p, "chosen": ch, "rejected": rj})
    type_report = {}
    for typ, rows in by_type.items():
        type_report[typ] = {
            "rows": len(rows),
            "avg_rejected_length": safe_mean([len(str(x["pair"].get("rejected", ""))) for x in rows]),
            "unique_rejected_answer_rate": len({str(x["rejected"].get("answer", "")) for x in rows if isinstance(x["rejected"], dict)}) / max(1, len(rows)),
            "unique_rejected_reason_rate": len({str(x["rejected"].get("reason", "")) for x in rows if isinstance(x["rejected"], dict)}) / max(1, len(rows)),
            "schema_valid_rejected_rate": sum(1 for x in rows if isinstance(x["rejected"], dict) and {"answer", "evidence", "confidence", "reason"} <= set(x["rejected"])) / max(1, len(rows)),
            "same_evidence_as_chosen_rate": same_evidence_rate(rows),
            "template_phrase_rate": template_phrase_rate(rows),
        }
    return {
        "rows": len(pairs),
        "reject_type": dict(Counter(p.get("reject_type", "unknown") for p in pairs)),
        "source": dict(Counter(p.get("source", "unknown") for p in pairs)),
        "difficulty": dict(Counter(p.get("difficulty", "unknown") for p in pairs)),
        "answerability_type": dict(Counter(p.get("answerability_type", "unknown") for p in pairs)),
        "length_bias": {
            "chosen_length": numeric_summary(chosen_lens),
            "rejected_length": numeric_summary(rejected_lens),
            "chosen_rejected_mean_ratio": ratio(safe_mean(chosen_lens), safe_mean(rejected_lens)),
        },
        "by_reject_type": type_report,
        "artifact_risks": {
            "wrong_format_easy_to_detect": type_report.get("wrong_format", {}).get("schema_valid_rejected_rate", 1.0) < 0.25,
            "over_refusal_template_like": type_report.get("over_refusal", {}).get("unique_rejected_answer_rate", 1.0) < 0.25,
            "forced_answer_template_like": type_report.get("forced_answer", {}).get("unique_rejected_reason_rate", 1.0) < 0.25,
            "fabricated_number_mechanical": type_report.get("fabricated_number", {}).get("same_evidence_as_chosen_rate", 0.0) > 0.8,
            "wrong_citation_potentially_confusing": type_report.get("wrong_citation", {}).get("same_evidence_as_chosen_rate", 1.0) < 0.5,
        },
        "notes": "Rule DPO is a seed set. Artifact risks must be diluted by model-mined pairs and preference audit.",
    }


def financebench_audit_report(data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    finance = data["financebench"]
    train = data["train_unified"]
    finance_features = [row_difficulty_features(r) for r in finance]
    train_features = [row_difficulty_features(r) for r in train]
    train_questions = {normalize_for_match(r.get("question", "")) for r in train}
    finance_questions = {normalize_for_match(r.get("prompt", r.get("question", ""))) for r in finance}
    return {
        "rows": len(finance),
        "in_training": False,
        "normalized_question_overlap_with_train": len(train_questions & finance_questions),
        "answer_type": dict(Counter(r.get("answer_type", "unknown") for r in finance)),
        "grounding_type": dict(Counter(r.get("grounding_type", "unknown") for r in finance)),
        "prompt_length": numeric_summary([len(r.get("prompt", "")) for r in finance]),
        "context_quality": {
            "avg_context_chunks": safe_mean([len(r.get("contexts", [])) for r in finance]),
            "quote_hit_rate": evidence_quote_hit_rate(finance),
            "citation_confusable_proxy_rate": rate(finance_features, "citation_confusable_proxy"),
        },
        "distribution_difference_vs_train": {
            "financebench_answer_type": dict(Counter(r.get("answer_type", "unknown") for r in finance)),
            "train_answer_type": dict(Counter(r.get("answer_type", "unknown") for r in train)),
            "financebench_avg_context_tokens": safe_mean([f["context_tokens"] for f in finance_features]),
            "train_avg_context_tokens": safe_mean([f["context_tokens"] for f in train_features]),
            "financebench_answer_exact_in_context_rate": rate(finance_features, "answer_exact_in_context"),
            "train_answer_exact_in_context_rate": rate(train_features, "answer_exact_in_context"),
        },
        "notes": "FinanceBench is treated as external audit only and is not read by SFT/DPO data builders.",
    }


def row_difficulty_features(row: Dict[str, Any]) -> Dict[str, Any]:
    contexts = row.get("contexts", [])
    context_text = " ".join(c.get("text", "") for c in contexts)
    evidence_text = " ".join(e.get("quote", "") for e in row.get("evidence", []))
    answer = row.get("answer", row.get("gold_answer", ""))
    answer_norm = normalize_for_match(answer)
    ctx_norm = normalize_for_match(context_text)
    ev_norm = normalize_for_match(evidence_text)
    numeric = bool(extract_numbers(answer))
    numeric_direct = any(n in {x.replace(",", "") for x in extract_numbers(context_text)} for n in [x.replace(",", "") for x in extract_numbers(answer)])
    context_tokens = len(re.findall(r"\w+", context_text))
    answer_in_context = bool(answer_norm and answer_norm in ctx_norm)
    answer_in_evidence = bool(answer_norm and answer_norm in ev_norm)
    citation_confusable = len(contexts) >= 3 or len(collect_numbers(context_text)) >= 8
    trivial = answer_in_context and row.get("reasoning_type") == "lookup" and len(contexts) <= 2
    non_trivial = not trivial and (row.get("grounding_type") in {"numeric_grounded", "calculation_hard"} or citation_confusable or row.get("reasoning_type") in {"calculation", "multi_hop", "comparison"})
    return {
        "answer_exact_in_context": answer_in_context,
        "answer_exact_in_evidence": answer_in_evidence,
        "is_numeric": numeric,
        "numeric_answer_directly_appears": numeric_direct,
        "context_chunks": len(contexts),
        "context_tokens": context_tokens,
        "citation_confusable_proxy": citation_confusable,
        "trivial_lookup_proxy": trivial,
        "non_trivial_grounded_proxy": non_trivial,
    }


def numeric_features(row: Dict[str, Any]) -> Dict[str, Any]:
    question = str(row.get("question", "")).lower()
    context_text = " ".join(c.get("text", "") for c in row.get("contexts", []))
    nums = collect_numbers(context_text)
    answer_nums = [n.replace(",", "") for n in extract_numbers(row.get("answer", ""))]
    context_nums = {n.replace(",", "") for n in nums}
    numeric_direct = any(n in context_nums for n in answer_nums)
    return {
        "direct_lookup_proxy": numeric_direct and row.get("reasoning_type") == "lookup",
        "simple_add_sub_proxy": any(w in question for w in ["change", "difference", "increase", "decrease", "higher", "lower"]),
        "percentage_change_proxy": any(w in question for w in ["percent", "percentage", "growth rate", "%"]),
        "year_alignment_proxy": bool(re.search(r"\b20\d{2}\b|\b19\d{2}\b", question)),
        "unit_alignment_proxy": any(w in question for w in ["million", "thousand", "usd", "$", "%", "percent"]),
        "multiple_candidate_numbers": len(context_nums) >= 8,
        "numeric_answer_directly_appears": numeric_direct,
    }


def unanswerable_features(row: Dict[str, Any]) -> Dict[str, Any]:
    question_tokens = set(re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", str(row.get("question", "")).lower()))
    context_text = " ".join(c.get("text", "") for c in row.get("contexts", [])).lower()
    context_tokens = set(re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", context_text))
    overlap = len(question_tokens & context_tokens) / max(1, len(question_tokens))
    return {
        "partial_evidence_proxy": overlap >= 0.35,
        "question_overlap_with_context_proxy": overlap,
        "context_chunks": len(row.get("contexts", [])),
    }


def same_evidence_rate(rows: List[Dict[str, Any]]) -> float:
    same = 0
    for item in rows:
        ch = item["chosen"] if isinstance(item["chosen"], dict) else {}
        rj = item["rejected"] if isinstance(item["rejected"], dict) else {}
        if ch.get("evidence") == rj.get("evidence"):
            same += 1
    return same / max(1, len(rows))


def template_phrase_rate(rows: List[Dict[str, Any]]) -> float:
    phrases = ["insufficient evidence", "directly supported", "numeric answer", "provided evidence"]
    count = 0
    for item in rows:
        text = json.dumps(item["rejected"], ensure_ascii=False).lower()
        if any(p in text for p in phrases):
            count += 1
    return count / max(1, len(rows))


def parse_json_obj(text: Any) -> Dict[str, Any]:
    if isinstance(text, dict):
        return text
    try:
        obj = json.loads(str(text))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def evidence_quote_hit_rate(rows: List[Dict[str, Any]]) -> float:
    total = hit = 0
    for row in rows:
        context = normalize_for_match(" ".join(c.get("text", "") for c in row.get("contexts", [])))
        for ev in row.get("evidence", []):
            quote = normalize_for_match(ev.get("quote", ""))
            if not quote:
                continue
            total += 1
            if quote in context:
                hit += 1
    return hit / max(1, total)


def group_rows(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key, "unknown"))].append(row)
    return groups


def collect_numbers(text: str) -> List[str]:
    return extract_numbers(text)


def extract_numbers(text: Any) -> List[str]:
    return re.findall(r"-?\d[\d,]*(?:\.\d+)?%?", str(text))


def first_words(text: Any, n: int) -> str:
    words = re.findall(r"\w+", str(text).lower())
    return " ".join(words[:n])


def rate(features: List[Dict[str, Any]], key: str, denom_key: str | None = None) -> float:
    if denom_key:
        denom = [f for f in features if f.get(denom_key)]
    else:
        denom = features
    return sum(1 for f in denom if f.get(key)) / max(1, len(denom))


def numeric_summary(values: List[float]) -> Dict[str, Any]:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return {"count": 0, "mean": 0, "p50": 0, "max": 0}
    vals_sorted = sorted(vals)
    return {
        "count": len(vals),
        "mean": safe_mean(vals),
        "p50": median(vals_sorted),
        "p90": vals_sorted[min(len(vals_sorted) - 1, int(len(vals_sorted) * 0.9))],
        "max": max(vals_sorted),
    }


def safe_mean(values: List[float]) -> float:
    return float(mean(values)) if values else 0.0


def ratio(a: float, b: float) -> float:
    return a / b if b else 0.0


def distribution_rate(dist: Dict[str, int], key: str) -> float:
    return dist.get(key, 0) / max(1, sum(dist.values()))


def eval_simpler_warning(train: Dict[str, Any], ev: Dict[str, Any]) -> bool:
    train_direct = distribution_rate(train["grounding_type"], "direct_grounded")
    eval_direct = distribution_rate(ev["grounding_type"], "direct_grounded")
    prompt_ratio = ratio(ev["prompt_length"]["mean"], train["prompt_length"]["mean"])
    return eval_direct > train_direct + 0.10 or prompt_ratio < 0.70


if __name__ == "__main__":
    main()
