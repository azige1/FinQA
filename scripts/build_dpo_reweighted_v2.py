"""Build a conservative reweighted DPO v2 dataset from v1 pairs."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_QUOTAS = {
    # Keep hard model-mined numeric failures; these are closest to observed SFT mistakes.
    ("model_mined", "fabricated_number"): 167,
    ("model_mined", "calculation_error"): 25,
    ("numeric_corruption", "fabricated_number"): 168,
    # Keep citation learning, but reduce the synthetic wrong-citation pressure that dominated v1 behavior.
    ("model_mined", "wrong_citation"): 40,
    ("wrong_citation_corruption", "wrong_citation"): 60,
    # Keep answerability constraints, but lower over-refusal/missing-evidence pressure.
    ("rule_generated", "missing_evidence"): 80,
    ("rule_generated", "over_refusal"): 70,
    ("hard_negative", "forced_answer"): 70,
    ("model_mined", "forced_answer"): 1,
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/dpo/dpo_balanced_v1.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/dpo/dpo_reweighted_v2.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("reports/dpo_reweighted_v2_report.json"))
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = read_jsonl(args.input)
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(str(row.get("source")), str(row.get("reject_type")))].append(row)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    quota_hits: dict[str, dict[str, int]] = {}
    for key, quota in DEFAULT_QUOTAS.items():
        candidates = buckets.get(key, [])
        take = candidates[:quota]
        quota_hits[f"{key[0]}::{key[1]}"] = {"available": len(candidates), "quota": quota, "selected": len(take)}
        for row in take:
            row_id = str(row.get("id"))
            if row_id in selected_ids:
                continue
            new_row = dict(row)
            new_row["dpo_v2_policy"] = "reweighted_numeric_first_beta005"
            selected.append(new_row)
            selected_ids.add(row_id)

    rng.shuffle(selected)
    write_jsonl(args.output, selected)

    report = {
        "input": str(args.input),
        "output": str(args.output),
        "seed": args.seed,
        "rows": len(selected),
        "quota_hits": quota_hits,
        "source": dict(Counter(str(row.get("source")) for row in selected)),
        "reject_type": dict(Counter(str(row.get("reject_type")) for row in selected)),
        "difficulty": dict(Counter(str(row.get("difficulty")) for row in selected)),
        "answerability_type": dict(Counter(str(row.get("answerability_type")) for row in selected)),
        "notes": [
            "Reduced synthetic wrong-citation pairs to avoid over-optimizing citation style.",
            "Kept all numeric corruption and model-mined calculation-error pairs available in v1.",
            "Designed to be trained with beta=0.05 and about 100 steps as a conservative DPO v2.",
        ],
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
