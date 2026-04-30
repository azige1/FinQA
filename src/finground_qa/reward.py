"""Rule reward utilities for RL-style grounded QA post-training."""

from __future__ import annotations

from typing import Any, Dict

from .checker import check_output
from .eval import extract_answer_from_prediction
from .text_utils import answer_equivalent


DEFAULT_REWARD_WEIGHTS: Dict[str, float] = {
    "json_valid": 0.6,
    "schema_pass": 0.6,
    "exact_match": 2.0,
    "numeric_exact_match": 0.8,
    "faithfulness_proxy": 1.0,
    "citation_precision": 1.0,
    "citation_consistency_score": 1.0,
    "wrong_citation": -1.2,
    "unsupported_claim": -1.2,
    "fabricated_number": -1.0,
    "calculation_error": -1.0,
    "over_refusal": -0.8,
    "forced_answer": -1.0,
    "format_error": -1.0,
    "schema_error": -0.8,
    "generic_answer": -0.5,
}


def grounded_reward(
    row: Dict[str, Any],
    prediction: str,
    weights: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    """Score one response with a bounded, decomposed rule reward.

    The reward intentionally mirrors the project evaluation proxies. It is not
    a human preference reward model; it is a transparent rule reward for
    RL-style experiments and reward hacking analysis.
    """

    active_weights = dict(DEFAULT_REWARD_WEIGHTS)
    if weights:
        active_weights.update(weights)

    check = check_output(row, prediction)
    answer = extract_answer_from_prediction(prediction)
    gold_answer = row.get("answer", row.get("gold_answer", ""))
    exact_match = bool(gold_answer) and answer_equivalent(answer, gold_answer)
    numeric_exact_match = row.get("answer_type") == "number" and exact_match

    features: Dict[str, float] = {
        "json_valid": float(bool(check.get("json_valid"))),
        "schema_pass": float(bool(check.get("schema_pass"))),
        "exact_match": float(exact_match),
        "numeric_exact_match": float(numeric_exact_match),
        "faithfulness_proxy": float(check.get("faithfulness_proxy", 0.0)),
        "citation_precision": float(check.get("citation_precision", 0.0)),
        "citation_consistency_score": float(check.get("citation_consistency_score", 0.0)),
        "wrong_citation": float(bool(check.get("wrong_citation"))),
        "unsupported_claim": float(bool(check.get("unsupported_claim"))),
        "fabricated_number": float(bool(check.get("fabricated_number"))),
        "calculation_error": float(bool(check.get("calculation_error"))),
        "over_refusal": float(bool(check.get("over_refusal"))),
        "forced_answer": float(bool(check.get("forced_answer"))),
        "format_error": float(bool(check.get("format_error"))),
        "schema_error": float(bool(check.get("schema_error"))),
        "generic_answer": float(bool(check.get("generic_answer"))),
    }
    contributions = {
        key: active_weights[key] * value
        for key, value in features.items()
        if key in active_weights
    }
    raw_reward = sum(contributions.values())
    # Keep the reward in a PPO/GRPO-friendly range while preserving ordering.
    normalized_reward = max(-1.0, min(1.0, raw_reward / 6.0))
    return {
        "reward": normalized_reward,
        "raw_reward": raw_reward,
        "features": features,
        "contributions": contributions,
        "eval": check,
        "answer": answer,
        "gold_answer": gold_answer,
    }


def reward_summary_init() -> Dict[str, Any]:
    return {
        "count": 0,
        "reward_sum": 0.0,
        "raw_reward_sum": 0.0,
        "feature_sums": {key: 0.0 for key in DEFAULT_REWARD_WEIGHTS},
    }


def reward_summary_update(summary: Dict[str, Any], scored: Dict[str, Any]) -> None:
    summary["count"] += 1
    summary["reward_sum"] += float(scored["reward"])
    summary["raw_reward_sum"] += float(scored["raw_reward"])
    for key, value in scored["features"].items():
        summary["feature_sums"][key] = summary["feature_sums"].get(key, 0.0) + float(value)


def reward_summary_finalize(summary: Dict[str, Any]) -> Dict[str, Any]:
    count = max(1, int(summary["count"]))
    return {
        "count": summary["count"],
        "reward_mean": summary["reward_sum"] / count,
        "raw_reward_mean": summary["raw_reward_sum"] / count,
        "features": {
            key: value / count
            for key, value in sorted(summary["feature_sums"].items())
        },
    }
