"""Post-hoc DPO model selection and v3 regression audit.

This script compares SFT, DPO v2 checkpoint-50, and DPO v3 guarded s100 using
existing scored eval artifacts. It does not run model inference.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.finground_qa.eval import extract_answer_from_prediction
from src.finground_qa.text_utils import answer_equivalent, normalize_text


METRIC_KEYS = [
    "exact_match",
    "numeric_exact_match",
    "faithfulness_rate",
    "citation_precision",
    "citation_consistency_score",
    "number_coverage_rate",
    "wrong_citation_rate",
    "unsupported_claim_rate",
    "fabricated_number_rate",
    "calculation_error_rate",
    "over_refusal_rate",
    "forced_answer_rate",
    "schema_pass_rate",
]

ERROR_KEYS = [
    "wrong_citation",
    "unsupported_claim",
    "fabricated_number",
    "calculation_error",
    "over_refusal",
    "forced_answer",
    "missing_evidence",
    "schema_error",
    "format_error",
]

LOWER_BETTER = {
    "wrong_citation_rate",
    "unsupported_claim_rate",
    "fabricated_number_rate",
    "calculation_error_rate",
    "over_refusal_rate",
    "forced_answer_rate",
}

DEFAULT_MODELS = {
    "base": {
        "label": "Base",
        "metrics": "results/base_metrics.json",
        "scored": "results/base_scored.jsonl",
        "adapter": "",
    },
    "sft": {
        "label": "SFT",
        "metrics": "results/sft_metrics.json",
        "scored": "results/sft_scored.jsonl",
        "adapter": "results/sft/qwen25_7b_finground_sft",
    },
    "dpo_v1_s500": {
        "label": "DPO v1 s500",
        "metrics": "results/dpo_metrics.json",
        "scored": "results/dpo_scored.jsonl",
        "adapter": "results/dpo/qwen25_7b_finground_dpo",
    },
    "dpo_v2_s100": {
        "label": "DPO v2 s100",
        "metrics": "results/dpo_v2_reweighted_s100_metrics.json",
        "scored": "results/dpo_v2_reweighted_s100_scored.jsonl",
        "adapter": "results/dpo/qwen25_7b_finground_dpo_v2_reweighted_s100",
    },
    "dpo_v2_s50": {
        "label": "DPO v2 s50",
        "metrics": "results/dpo_v2_reweighted_s50_metrics.json",
        "scored": "results/dpo_v2_reweighted_s50_scored.jsonl",
        "adapter": "results/dpo/qwen25_7b_finground_dpo_v2_reweighted_s100/checkpoint-50",
    },
    "dpo_v3_guarded_s100": {
        "label": "DPO v3 guarded s100",
        "metrics": "results/dpo_v3_guarded_s100_metrics.json",
        "scored": "results/dpo_v3_guarded_s100_scored.jsonl",
        "adapter": "results/dpo/qwen25_7b_finground_dpo_v3_guarded_s100_candidate",
    },
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def selection_score(metrics: dict[str, Any]) -> float:
    """Compact heuristic: reward accuracy/grounding, penalize factuality failures."""
    return (
        2.0 * float(metrics.get("exact_match", 0.0))
        + 1.5 * float(metrics.get("numeric_exact_match", 0.0))
        + 1.0 * float(metrics.get("faithfulness_rate", 0.0))
        + 0.6 * float(metrics.get("citation_precision", 0.0))
        + 0.4 * float(metrics.get("citation_consistency_score", 0.0))
        + 0.4 * float(metrics.get("schema_pass_rate", 0.0))
        - 0.8 * float(metrics.get("wrong_citation_rate", 0.0))
        - 0.8 * float(metrics.get("unsupported_claim_rate", 0.0))
        - 0.8 * float(metrics.get("fabricated_number_rate", 0.0))
        - 0.8 * float(metrics.get("calculation_error_rate", 0.0))
        - 0.3 * float(metrics.get("over_refusal_rate", 0.0))
        - 0.3 * float(metrics.get("forced_answer_rate", 0.0))
    )


def metric_delta(metrics: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in METRIC_KEYS:
        if isinstance(metrics.get(key), (int, float)) and isinstance(baseline.get(key), (int, float)):
            out[key] = float(metrics[key]) - float(baseline[key])
    return out


def exact_for(row: dict[str, Any]) -> bool:
    answer = extract_answer_from_prediction(str(row.get("prediction") or row.get("output") or ""))
    gold = row.get("gold_answer") or row.get("answer") or ""
    return answer_equivalent(answer, gold)


def row_summary(row: dict[str, Any]) -> dict[str, Any]:
    text = str(row.get("prediction") or row.get("output") or "")
    return {
        "answer": extract_answer_from_prediction(text),
        "prediction": normalize_text(text)[:900],
        "exact": exact_for(row),
        "errors": {key: bool((row.get("eval") or {}).get(key)) for key in ERROR_KEYS},
    }


def compare_exact(
    candidate: dict[str, dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    eval_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    counts = Counter()
    dataset = defaultdict(Counter)
    numeric_counts = Counter()
    sample_ids: dict[str, list[str]] = {
        "improved_exact": [],
        "regressed_exact": [],
        "improved_numeric_exact": [],
        "regressed_numeric_exact": [],
    }
    for row_id, base_row in baseline.items():
        if row_id not in candidate:
            continue
        cand_exact = exact_for(candidate[row_id])
        base_exact = exact_for(base_row)
        meta = eval_rows.get(row_id, base_row)
        if cand_exact and not base_exact:
            label = "improved_exact"
        elif base_exact and not cand_exact:
            label = "regressed_exact"
        elif base_exact and cand_exact:
            label = "same_correct"
        else:
            label = "same_wrong"
        counts[label] += 1
        dataset[str(meta.get("dataset") or "unknown")][label] += 1
        if label in sample_ids and len(sample_ids[label]) < 40:
            sample_ids[label].append(row_id)
        if meta.get("answer_type") == "number":
            numeric_counts[label] += 1
            if label == "improved_exact" and len(sample_ids["improved_numeric_exact"]) < 40:
                sample_ids["improved_numeric_exact"].append(row_id)
            if label == "regressed_exact" and len(sample_ids["regressed_numeric_exact"]) < 40:
                sample_ids["regressed_numeric_exact"].append(row_id)
    return {
        "counts": dict(counts),
        "net_exact": counts["improved_exact"] - counts["regressed_exact"],
        "numeric_counts": dict(numeric_counts),
        "net_numeric_exact": numeric_counts["improved_exact"] - numeric_counts["regressed_exact"],
        "dataset_flips": {key: dict(value) for key, value in sorted(dataset.items())},
        "sample_ids": sample_ids,
    }


def compare_errors(
    candidate: dict[str, dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ERROR_KEYS:
        counts = Counter()
        sample_ids = {"fixed": [], "introduced": []}
        for row_id, base_row in baseline.items():
            if row_id not in candidate:
                continue
            base_bad = bool((base_row.get("eval") or {}).get(key))
            cand_bad = bool((candidate[row_id].get("eval") or {}).get(key))
            if base_bad and not cand_bad:
                label = "fixed"
            elif cand_bad and not base_bad:
                label = "introduced"
            elif cand_bad and base_bad:
                label = "same_bad"
            else:
                label = "same_good"
            counts[label] += 1
            if label in sample_ids and len(sample_ids[label]) < 40:
                sample_ids[label].append(row_id)
        out[key] = {
            **dict(counts),
            "net_fixed": counts["fixed"] - counts["introduced"],
            "sample_ids": sample_ids,
        }
    return out


def audit_rows_for(
    eval_rows: dict[str, dict[str, Any]],
    scored: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    sft = scored["sft"]
    v2 = scored["dpo_v2_s50"]
    v3 = scored["dpo_v3_guarded_s100"]
    rows: list[dict[str, Any]] = []
    for row_id, meta in eval_rows.items():
        if row_id not in sft or row_id not in v2 or row_id not in v3:
            continue
        sft_exact = exact_for(sft[row_id])
        v2_exact = exact_for(v2[row_id])
        v3_exact = exact_for(v3[row_id])
        v2_errors = v2[row_id].get("eval") or {}
        v3_errors = v3[row_id].get("eval") or {}
        categories: list[str] = []
        if sft_exact and v2_exact and not v3_exact:
            categories.append("v3_new_exact_regression_vs_sft_and_v2")
        if v2_exact and not v3_exact:
            categories.append("v3_regressed_vs_v2_exact")
        if sft_exact and not v3_exact:
            categories.append("v3_regressed_vs_sft_exact")
        if not v2_exact and v3_exact:
            categories.append("v3_improved_vs_v2_exact")
        if sft_exact and not v2_exact and v3_exact:
            categories.append("v3_fixed_v2_sft_regression")
        if not sft_exact and not v2_exact and not v3_exact:
            categories.append("persistent_all_wrong")
        if meta.get("answer_type") == "number" and v2_exact and not v3_exact:
            categories.append("v3_numeric_regression_vs_v2")
        if meta.get("answer_type") == "number" and not v2_exact and v3_exact:
            categories.append("v3_numeric_improvement_vs_v2")
        if meta.get("grounding_type") == "unanswerable" or meta.get("answer_type") == "unanswerable":
            if v3_errors.get("forced_answer"):
                categories.append("v3_forced_answer_unanswerable")
            if v2_errors.get("forced_answer") and not v3_errors.get("forced_answer"):
                categories.append("v3_fixed_v2_forced_answer")
        for key in ERROR_KEYS:
            if v3_errors.get(key) and not v2_errors.get(key):
                categories.append(f"v3_introduced_{key}")
            if v2_errors.get(key) and not v3_errors.get(key):
                categories.append(f"v3_fixed_{key}")
        unique_categories = sorted(set(categories))
        if not unique_categories or unique_categories == ["persistent_all_wrong"]:
            continue
        rows.append(
            {
                "id": row_id,
                "categories": unique_categories,
                "dataset": meta.get("dataset"),
                "answer_type": meta.get("answer_type"),
                "reasoning_type": meta.get("reasoning_type"),
                "grounding_type": meta.get("grounding_type"),
                "gold_answer": meta.get("answer") or meta.get("gold_answer"),
                "question": meta.get("question"),
                "sft": row_summary(sft[row_id]),
                "dpo_v2_s50": row_summary(v2[row_id]),
                "dpo_v3_guarded_s100": row_summary(v3[row_id]),
            }
        )
    return rows


def persistent_all_wrong_total(
    eval_rows: dict[str, dict[str, Any]],
    scored: dict[str, dict[str, dict[str, Any]]],
) -> int:
    total = 0
    for row_id in eval_rows:
        rows = [
            scored["sft"].get(row_id),
            scored["dpo_v2_s50"].get(row_id),
            scored["dpo_v3_guarded_s100"].get(row_id),
        ]
        if any(row is None for row in rows):
            continue
        if all(not exact_for(row) for row in rows if row is not None):
            total += 1
    return total


def summarize_audit_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    category_counts = Counter()
    dataset_counts = defaultdict(Counter)
    severity_counts = Counter()
    for row in rows:
        cats = set(row["categories"])
        for cat in cats:
            category_counts[cat] += 1
            dataset_counts[str(row.get("dataset") or "unknown")][cat] += 1
        if "v3_new_exact_regression_vs_sft_and_v2" in cats or "v3_forced_answer_unanswerable" in cats:
            severity_counts["critical"] += 1
        elif any(cat.endswith("_regression_vs_v2") or cat.startswith("v3_regressed") for cat in cats):
            severity_counts["high"] += 1
        elif any(cat.startswith("v3_introduced") for cat in cats):
            severity_counts["medium"] += 1
        else:
            severity_counts["low"] += 1
    return {
        "flagged_rows": len(rows),
        "category_counts": dict(category_counts),
        "dataset_category_counts": {key: dict(value) for key, value in sorted(dataset_counts.items())},
        "severity_counts": dict(severity_counts),
    }


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_cell(value) for value in row) + " |")
    return "\n".join(lines)


def format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def make_selection_markdown(report: dict[str, Any]) -> str:
    rows = []
    for name, item in report["summary"].items():
        metrics = item["metrics"]
        rows.append(
            [
                name,
                metrics.get("exact_match"),
                metrics.get("numeric_exact_match"),
                metrics.get("faithfulness_rate"),
                metrics.get("citation_precision"),
                metrics.get("citation_consistency_score"),
                metrics.get("wrong_citation_rate"),
                metrics.get("fabricated_number_rate"),
                metrics.get("calculation_error_rate"),
                item["selection_score"],
            ]
        )
    ranking = "\n".join(
        f"- {item['model']}: {item['score']:.4f}" for item in report["ranking_by_selection_score"]
    )
    recommendation = report["recommendation"]
    return (
        "# DPO Model Selection Report\n\n"
        f"Generated: {report['generated_at']}\n\n"
        f"Formal recommendation: **{recommendation['formal_recommendation']}**\n\n"
        f"Best DPO candidate: **{recommendation['best_dpo_candidate']}**\n\n"
        f"Reason: {recommendation['reason']}\n\n"
        "## Overall Metrics\n\n"
        + markdown_table(
            [
                "model",
                "exact_match",
                "numeric_exact_match",
                "faithfulness_rate",
                "citation_precision",
                "citation_consistency_score",
                "wrong_citation_rate",
                "fabricated_number_rate",
                "calculation_error_rate",
                "selection_score",
            ],
            rows,
        )
        + "\n\n## Ranking By Selection Score\n\n"
        + ranking
        + "\n\n## Gate Notes\n\n"
        + "\n".join(f"- {note}" for note in recommendation["gate_notes"])
        + "\n\n## Key Deltas Vs SFT\n\n"
        + markdown_table(
            ["model", "exact_match", "numeric_exact_match", "citation_precision", "fabricated_number_rate", "forced_answer_rate"],
            [
                [
                    name,
                    deltas.get("exact_match", 0.0),
                    deltas.get("numeric_exact_match", 0.0),
                    deltas.get("citation_precision", 0.0),
                    deltas.get("fabricated_number_rate", 0.0),
                    deltas.get("forced_answer_rate", 0.0),
                ]
                for name, deltas in report["deltas_vs_sft"].items()
            ],
        )
        + "\n"
    )


def make_audit_markdown(report: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    summary = report["audit_summary"]
    top_categories = sorted(summary["category_counts"].items(), key=lambda item: (-item[1], item[0]))[:16]
    blockers = [
        row
        for row in rows
        if "v3_new_exact_regression_vs_sft_and_v2" in row["categories"]
        or "v3_forced_answer_unanswerable" in row["categories"]
        or "v3_numeric_regression_vs_v2" in row["categories"]
    ][:12]
    blocker_lines = []
    for row in blockers:
        blocker_lines.append(
            "- "
            + row["id"]
            + f" ({row.get('dataset')}, {row.get('grounding_type')}): "
            + ", ".join(row["categories"][:4])
            + f"; gold={row.get('gold_answer')!r}; v2={row['dpo_v2_s50']['answer']!r}; v3={row['dpo_v3_guarded_s100']['answer']!r}"
        )
    return (
        "# DPO v3 Post-hoc Audit\n\n"
        f"Generated: {report['generated_at']}\n\n"
        f"Conclusion: **{report['conclusion']}**\n\n"
        f"Delta audit rows: **{summary['flagged_rows']}**\n\n"
        f"Persistent all-wrong rows tracked separately: **{report['persistent_all_wrong_total']}**\n\n"
        "## Flip Summary\n\n"
        + markdown_table(
            ["comparison", "improved_exact", "regressed_exact", "net_exact", "improved_numeric", "regressed_numeric", "net_numeric"],
            [
                [
                    name,
                    comp["exact"]["counts"].get("improved_exact", 0),
                    comp["exact"]["counts"].get("regressed_exact", 0),
                    comp["exact"]["net_exact"],
                    comp["exact"]["numeric_counts"].get("improved_exact", 0),
                    comp["exact"]["numeric_counts"].get("regressed_exact", 0),
                    comp["exact"]["net_numeric_exact"],
                ]
                for name, comp in report["comparisons"].items()
            ],
        )
        + "\n\n## Category Counts\n\n"
        + markdown_table(["category", "count"], [[key, value] for key, value in top_categories])
        + "\n\n## Blocking Examples\n\n"
        + ("\n".join(blocker_lines) if blocker_lines else "- No blocking examples found.")
        + "\n\n## Next Data Actions\n\n"
        + "\n".join(f"- {item}" for item in report["next_data_actions"])
        + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-file", type=Path, default=Path("data/eval/eval.jsonl"))
    parser.add_argument("--selection-json", type=Path, default=Path("reports/dpo_model_selection_report.json"))
    parser.add_argument("--selection-md", type=Path, default=Path("reports/dpo_model_selection_report.md"))
    parser.add_argument("--flip-json", type=Path, default=Path("reports/dpo_model_selection_flip_report.json"))
    parser.add_argument("--audit-json", type=Path, default=Path("reports/dpo_v3_posthoc_audit_report.json"))
    parser.add_argument("--audit-md", type=Path, default=Path("reports/dpo_v3_posthoc_audit_report.md"))
    parser.add_argument("--audit-jsonl", type=Path, default=Path("results/dpo_v3_posthoc_audit.jsonl"))
    args = parser.parse_args()

    generated_at = datetime.now().replace(microsecond=0).isoformat()
    eval_rows = {row["id"]: row for row in read_jsonl(args.eval_file)}
    metrics: dict[str, dict[str, Any]] = {}
    scored: dict[str, dict[str, dict[str, Any]]] = {}
    for name, spec in DEFAULT_MODELS.items():
        metrics_path = Path(spec["metrics"])
        scored_path = Path(spec["scored"])
        if metrics_path.exists():
            metrics[name] = read_json(metrics_path)
        if scored_path.exists():
            scored[name] = {row["id"]: row for row in read_jsonl(scored_path)}

    summary = {}
    for name, item in DEFAULT_MODELS.items():
        if name not in metrics:
            continue
        summary[name] = {
            "label": item["label"],
            "metrics_path": item["metrics"],
            "scored_path": item["scored"],
            "adapter_path": item["adapter"],
            "metrics": {key: metrics[name].get(key) for key in METRIC_KEYS},
            "selection_score": selection_score(metrics[name]),
        }

    ranking = sorted(
        [{"model": name, "score": item["selection_score"]} for name, item in summary.items()],
        key=lambda item: item["score"],
        reverse=True,
    )
    dpo_ranking = [item for item in ranking if item["model"].startswith("dpo_")]
    deltas_vs_sft = {
        name: metric_delta(metrics[name], metrics["sft"])
        for name in summary
        if name != "sft" and "sft" in metrics
    }
    best_by_metric: dict[str, Any] = {}
    for key in METRIC_KEYS:
        candidates = [(name, data["metrics"].get(key)) for name, data in summary.items()]
        candidates = [(name, value) for name, value in candidates if isinstance(value, (int, float))]
        if not candidates:
            continue
        chooser = min if key in LOWER_BETTER else max
        model, value = chooser(candidates, key=lambda item: item[1])
        best_by_metric[key] = {"model": model, "value": value}

    recommendation = {
        "formal_recommendation": "keep_sft_as_final_baseline",
        "best_dpo_candidate": "dpo_v2_s50",
        "best_dpo_adapter_path": DEFAULT_MODELS["dpo_v2_s50"]["adapter"],
        "reason": (
            "DPO v2 s50 still has the best DPO score and numeric/citation balance. "
            "DPO v3 does not beat v2 s50 on numeric exact match or citation consistency, "
            "and v3 remains blocked by pending manual pair audit."
        ),
        "gate_notes": [
            "SFT remains the safest final baseline because post-DPO audit blocks replacing it.",
            "DPO v2 s50 remains the best DPO candidate for analysis, not a formal replacement.",
            "DPO v3 guarded s100 is experimental: formal_human_gate_pass=false and gate_pass=false.",
        ],
    }
    selection_report = {
        "generated_at": generated_at,
        "input_metrics": {name: spec["metrics"] for name, spec in DEFAULT_MODELS.items()},
        "summary": summary,
        "deltas_vs_sft": deltas_vs_sft,
        "best_by_metric": best_by_metric,
        "ranking_by_selection_score": ranking,
        "dpo_ranking_by_selection_score": dpo_ranking,
        "selection_score_definition": (
            "2*exact + 1.5*numeric_exact + faithfulness + 0.6*citation_precision "
            "+ 0.4*citation_consistency + 0.4*schema_pass - weighted factuality errors"
        ),
        "recommendation": recommendation,
    }

    comparisons = {
        "dpo_v2_s50_vs_sft": {
            "exact": compare_exact(scored["dpo_v2_s50"], scored["sft"], eval_rows),
            "errors": compare_errors(scored["dpo_v2_s50"], scored["sft"]),
        },
        "dpo_v3_guarded_s100_vs_sft": {
            "exact": compare_exact(scored["dpo_v3_guarded_s100"], scored["sft"], eval_rows),
            "errors": compare_errors(scored["dpo_v3_guarded_s100"], scored["sft"]),
        },
        "dpo_v3_guarded_s100_vs_dpo_v2_s50": {
            "exact": compare_exact(scored["dpo_v3_guarded_s100"], scored["dpo_v2_s50"], eval_rows),
            "errors": compare_errors(scored["dpo_v3_guarded_s100"], scored["dpo_v2_s50"]),
        },
    }
    audit_rows = audit_rows_for(eval_rows, scored)
    audit_summary = summarize_audit_rows(audit_rows)
    audit_report = {
        "generated_at": generated_at,
        "models": {
            "sft": DEFAULT_MODELS["sft"],
            "dpo_v2_s50": DEFAULT_MODELS["dpo_v2_s50"],
            "dpo_v3_guarded_s100": DEFAULT_MODELS["dpo_v3_guarded_s100"],
        },
        "metrics": {name: summary[name]["metrics"] for name in ["sft", "dpo_v2_s50", "dpo_v3_guarded_s100"]},
        "comparisons": comparisons,
        "audit_summary": audit_summary,
        "persistent_all_wrong_total": persistent_all_wrong_total(eval_rows, scored),
        "conclusion": (
            "NO_GO_FOR_PROMOTING_V3. V3 fixes some v2 rows but introduces offsetting exact/numeric "
            "regressions and remains below v2_s50 on the main DPO selection balance."
        ),
        "next_data_actions": [
            "Do not increase generic DPO steps before auditing the v3 regression rows.",
            "Create a small targeted v4 pool from v3_new_exact_regression_vs_sft_and_v2 and v3_numeric_regression_vs_v2.",
            "Keep unanswerable guards, but reduce pressure that converts correct refusals into answer-like outputs.",
            "Separate citation-repair pairs from numeric-answer pairs so citation gains do not trade off exact numeric accuracy.",
        ],
    }

    write_json(args.selection_json, selection_report)
    args.selection_md.write_text(make_selection_markdown(selection_report), encoding="utf-8")
    write_json(args.flip_json, {"generated_at": generated_at, "comparisons": comparisons})
    write_json(args.audit_json, audit_report)
    args.audit_md.write_text(make_audit_markdown(audit_report, audit_rows), encoding="utf-8")
    write_jsonl(args.audit_jsonl, audit_rows)

    print(f"wrote {args.selection_json}")
    print(f"wrote {args.selection_md}")
    print(f"wrote {args.flip_json}")
    print(f"wrote {args.audit_json}")
    print(f"wrote {args.audit_md}")
    print(f"wrote {args.audit_jsonl}")
    print(f"best_dpo_candidate={recommendation['best_dpo_candidate']}")
    print(f"audit_flagged_rows={len(audit_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
