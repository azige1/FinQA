#!/usr/bin/env python
"""Offline rule-reward and group-relative advantage analysis."""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finground_qa.io_utils import read_jsonl, write_json, write_jsonl
from finground_qa.reward import (
    grounded_reward,
    reward_summary_finalize,
    reward_summary_init,
    reward_summary_update,
)


def parse_prediction_arg(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("prediction must be NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("prediction must be NAME=PATH")
    return name, path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-file", default="data/eval/eval.jsonl")
    parser.add_argument(
        "--prediction",
        action="append",
        type=parse_prediction_arg,
        required=True,
        help="Prediction file as NAME=PATH. Can be repeated.",
    )
    parser.add_argument("--output", default="results/rl_offline_reward_scored.jsonl")
    parser.add_argument("--report", default="reports/rl_offline_reward_report.json")
    args = parser.parse_args()

    eval_rows = {row["id"]: row for row in read_jsonl(args.eval_file)}
    scored_rows: List[Dict[str, Any]] = []
    by_model: Dict[str, Dict[str, Any]] = defaultdict(reward_summary_init)
    by_id: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for model_name, pred_path in args.prediction:
        for pred in read_jsonl(pred_path):
            row_id = pred.get("id") or pred.get("row_id")
            if row_id not in eval_rows:
                continue
            prediction = pred.get("prediction") or pred.get("output") or pred.get("response") or ""
            scored = grounded_reward(eval_rows[row_id], prediction)
            item = {
                "id": row_id,
                "model": model_name,
                "prediction": prediction,
                "reward": scored["reward"],
                "raw_reward": scored["raw_reward"],
                "features": scored["features"],
                "answer": scored["answer"],
                "gold_answer": scored["gold_answer"],
            }
            scored_rows.append(item)
            by_id[row_id].append(item)
            reward_summary_update(by_model[model_name], scored)

    for group in by_id.values():
        rewards = [float(item["reward"]) for item in group]
        mean = sum(rewards) / max(1, len(rewards))
        variance = sum((value - mean) ** 2 for value in rewards) / max(1, len(rewards))
        std = math.sqrt(variance)
        for item in group:
            item["group_reward_mean"] = mean
            item["group_reward_std"] = std
            item["group_relative_advantage"] = (
                (float(item["reward"]) - mean) / std if std > 1e-8 else 0.0
            )

    model_names = [name for name, _ in args.prediction]
    win_matrix: Dict[str, Dict[str, float]] = {name: {} for name in model_names}
    by_model_id = {(item["model"], item["id"]): item for item in scored_rows}
    eval_ids = sorted(by_id)
    for left in model_names:
        for right in model_names:
            if left == right:
                continue
            comparable = 0
            wins = 0
            ties = 0
            for row_id in eval_ids:
                l_item = by_model_id.get((left, row_id))
                r_item = by_model_id.get((right, row_id))
                if not l_item or not r_item:
                    continue
                comparable += 1
                l_reward = float(l_item["reward"])
                r_reward = float(r_item["reward"])
                if abs(l_reward - r_reward) < 1e-8:
                    ties += 1
                elif l_reward > r_reward:
                    wins += 1
            win_matrix[left][right] = {
                "comparable": comparable,
                "win_rate": wins / max(1, comparable),
                "tie_rate": ties / max(1, comparable),
            }

    report = {
        "eval_file": args.eval_file,
        "num_scored": len(scored_rows),
        "num_prompts": len(by_id),
        "models": {
            name: reward_summary_finalize(summary)
            for name, summary in sorted(by_model.items())
        },
        "pairwise_reward_win_matrix": win_matrix,
        "notes": [
            "Rewards are transparent rule rewards, not learned human preference rewards.",
            "group_relative_advantage can be used as a GRPO-style diagnostic over multiple model responses per prompt.",
            "Use this report to inspect reward hacking before running online PPO/GRPO training.",
        ],
    }
    write_jsonl(args.output, scored_rows)
    write_json(args.report, report)
    print(f"wrote {args.output}")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
