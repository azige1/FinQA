"""vLLM batch generation for eval and model-mined rejected."""

from __future__ import annotations

import argparse
from typing import Any, Dict, List

from .io_utils import iter_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=8192)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = list(iter_jsonl(args.input))
    if args.limit > 0:
        rows = rows[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts: List[str] = []
    for row in rows:
        prompt = row.get("prompt") or row.get("instruction")
        messages = [{"role": "user", "content": prompt}]
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )
    outputs = llm.generate(prompts, sampling)

    out_rows: List[Dict[str, Any]] = []
    for row, output in zip(rows, outputs):
        new_row = dict(row)
        new_row["prediction"] = output.outputs[0].text
        out_rows.append(new_row)
    write_jsonl(args.output, out_rows)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
