"""Parse DPO Trainer logs and export training curves."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


METRIC_RE = re.compile(r"\{[^{}]*'loss':[^{}]*\}")
STEP_RE = re.compile(r"(\d+)/(\d+)")


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def parse_metrics(log_path: Path) -> list[dict[str, Any]]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    rows: list[dict[str, Any]] = []
    seen_steps: set[int] = set()
    for match in METRIC_RE.finditer(text):
        try:
            payload = ast.literal_eval(match.group(0))
        except (SyntaxError, ValueError):
            continue
        if not isinstance(payload, dict) or "loss" not in payload:
            continue
        prefix = text[max(0, match.start() - 1200) : match.start()]
        step_matches = STEP_RE.findall(prefix)
        if step_matches:
            step, total_steps = map(int, step_matches[-1])
        else:
            step, total_steps = len(rows) + 1, None
        if step in seen_steps:
            continue
        seen_steps.add(step)
        row: dict[str, Any] = {"step": step}
        if total_steps is not None:
            row["total_steps"] = total_steps
        for key, value in payload.items():
            converted = to_float(value)
            row[key] = converted if converted is not None else value
        rows.append(row)
    rows.sort(key=lambda item: int(item["step"]))
    return rows


def write_csv(rows: list[dict[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows: list[dict[str, Any]], summary_path: Path) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    last = rows[-1] if rows else {}
    summary = {
        "num_points": len(rows),
        "first_step": rows[0]["step"] if rows else None,
        "last_step": last.get("step"),
        "last_loss": last.get("loss"),
        "last_reward_accuracy": last.get("rewards/accuracies"),
        "last_reward_margin": last.get("rewards/margins"),
        "last_learning_rate": last.get("learning_rate"),
        "last_epoch": last.get("epoch"),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def plot(rows: list[dict[str, Any]], png_path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    if not rows:
        return False
    png_path.parent.mkdir(parents=True, exist_ok=True)
    steps = [int(row["step"]) for row in rows]
    series = [
        ("loss", "Loss"),
        ("rewards/accuracies", "Reward Accuracy"),
        ("rewards/margins", "Reward Margin"),
        ("learning_rate", "Learning Rate"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    fig.suptitle("FinGround-QA DPO Training", fontsize=15)
    for ax, (key, title) in zip(axes.ravel(), series):
        values = [to_float(row.get(key)) for row in rows]
        points = [(step, value) for step, value in zip(steps, values) if value is not None]
        if points:
            xs, ys = zip(*points)
            ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=3.5)
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.25)
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    return True


def replay_to_wandb(
    rows: list[dict[str, Any]],
    project: str,
    run_name: str,
    png_path: Path | None,
    offline: bool,
) -> bool:
    try:
        import wandb
    except ImportError:
        print("wandb is not installed; skipped W&B replay", file=sys.stderr)
        return False

    mode = "offline" if offline else None
    run = wandb.init(project=project, name=run_name, mode=mode, reinit=True)
    try:
        for row in rows:
            step = int(row["step"])
            payload = {k: v for k, v in row.items() if k not in {"step", "total_steps"}}
            wandb.log(payload, step=step)
        if png_path and png_path.exists():
            wandb.log({"dpo_training_curve": wandb.Image(str(png_path))})
    finally:
        run.finish()
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-log", type=Path, default=Path("logs/dpo_train.log"))
    parser.add_argument("--csv", type=Path, default=Path("results/dpo_training_metrics.csv"))
    parser.add_argument("--png", type=Path, default=Path("reports/dpo_training_curve.png"))
    parser.add_argument("--summary", type=Path, default=Path("reports/dpo_training_curve_summary.json"))
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--wandb-run-name", default="finground-dpo-qwen25-7b-formal-replay")
    parser.add_argument("--wandb-offline", action="store_true")
    args = parser.parse_args()

    if not args.train_log.exists():
        print(f"missing train log: {args.train_log}", file=sys.stderr)
        return 2
    rows = parse_metrics(args.train_log)
    if not rows:
        print(f"no metric rows parsed from {args.train_log}", file=sys.stderr)
        return 3
    write_csv(rows, args.csv)
    write_summary(rows, args.summary)
    png_ok = plot(rows, args.png)
    print(f"parsed {len(rows)} metric rows")
    print(f"wrote {args.csv}")
    print(f"wrote {args.summary}")
    print(f"wrote {args.png}" if png_ok else "matplotlib unavailable; skipped png")
    if args.wandb_project:
        replay_to_wandb(rows, args.wandb_project, args.wandb_run_name, args.png, args.wandb_offline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
