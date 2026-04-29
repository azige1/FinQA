"""Summarize DPO checkpoint sweep metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


KEYS = [
    "exact_match",
    "numeric_exact_match",
    "faithfulness_rate",
    "citation_precision",
    "citation_consistency_score",
    "schema_pass_rate",
    "wrong_citation_rate",
    "unsupported_claim_rate",
    "fabricated_number_rate",
    "calculation_error_rate",
    "missing_evidence_rate",
    "over_refusal_rate",
]


LOWER_BETTER = {
    "wrong_citation_rate",
    "unsupported_claim_rate",
    "fabricated_number_rate",
    "calculation_error_rate",
    "missing_evidence_rate",
    "over_refusal_rate",
}


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def composite_score(metrics: dict[str, Any]) -> float:
    """Conservative selection score: accuracy first, citation/error quality second."""
    return (
        2.0 * float(metrics.get("exact_match", 0.0))
        + 1.5 * float(metrics.get("numeric_exact_match", 0.0))
        + 1.0 * float(metrics.get("faithfulness_rate", 0.0))
        + 0.6 * float(metrics.get("citation_precision", 0.0))
        + 0.4 * float(metrics.get("schema_pass_rate", 0.0))
        - 0.8 * float(metrics.get("wrong_citation_rate", 0.0))
        - 0.8 * float(metrics.get("unsupported_claim_rate", 0.0))
        - 0.8 * float(metrics.get("fabricated_number_rate", 0.0))
        - 0.8 * float(metrics.get("calculation_error_rate", 0.0))
        - 0.3 * float(metrics.get("over_refusal_rate", 0.0))
    )


def delta(value: Any, baseline: Any) -> float | None:
    if isinstance(value, (int, float)) and isinstance(baseline, (int, float)):
        return float(value) - float(baseline)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, default=Path("results/checkpoint_sweep"))
    parser.add_argument("--base-metrics", type=Path, default=Path("results/base_metrics.json"))
    parser.add_argument("--sft-metrics", type=Path, default=Path("results/sft_metrics.json"))
    parser.add_argument("--final-dpo-metrics", type=Path, default=Path("results/dpo_metrics.json"))
    parser.add_argument("--output-json", type=Path, default=Path("reports/dpo_checkpoint_sweep_report.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("reports/dpo_checkpoint_sweep_metrics.csv"))
    args = parser.parse_args()

    base = read_json(args.base_metrics)
    sft = read_json(args.sft_metrics)
    final_dpo = read_json(args.final_dpo_metrics)

    rows: list[dict[str, Any]] = []
    for path in sorted(args.sweep_dir.glob("dpo_ckpt_*_metrics.json")):
        step_text = path.stem.removeprefix("dpo_ckpt_").removesuffix("_metrics")
        try:
            step = int(step_text)
        except ValueError:
            continue
        metrics = read_json(path)
        if not metrics:
            continue
        row: dict[str, Any] = {
            "name": f"checkpoint-{step}",
            "step": step,
            "metrics_path": str(path),
            "composite_score": composite_score(metrics),
        }
        for key in KEYS:
            row[key] = metrics.get(key)
            if sft:
                row[f"{key}_vs_sft"] = delta(metrics.get(key), sft.get(key))
        rows.append(row)

    if final_dpo:
        row = {
            "name": "final-500",
            "step": 500,
            "metrics_path": str(args.final_dpo_metrics),
            "composite_score": composite_score(final_dpo),
        }
        for key in KEYS:
            row[key] = final_dpo.get(key)
            if sft:
                row[f"{key}_vs_sft"] = delta(final_dpo.get(key), sft.get(key))
        rows.append(row)

    rows.sort(key=lambda item: (int(item["step"]), str(item["name"])))
    best = max(rows, key=lambda item: float(item["composite_score"])) if rows else None
    best_accuracy = max(rows, key=lambda item: float(item.get("exact_match") or 0.0)) if rows else None
    best_citation = min(rows, key=lambda item: float(item.get("wrong_citation_rate") or 1.0)) if rows else None
    report = {
        "base_metrics": base,
        "sft_metrics": sft,
        "checkpoints": rows,
        "best_by_composite": best,
        "best_by_exact_match": best_accuracy,
        "best_by_wrong_citation_rate": best_citation,
        "selection_note": (
            "Prefer checkpoints that improve citation quality without materially hurting exact/numeric accuracy."
        ),
        "lower_better": sorted(LOWER_BETTER),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_csv}")
    if best:
        print(
            "best_by_composite",
            best["name"],
            "score",
            f"{float(best['composite_score']):.6f}",
            "exact",
            best.get("exact_match"),
            "wrong_citation",
            best.get("wrong_citation_rate"),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
