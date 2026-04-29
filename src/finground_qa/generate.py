"""Lightweight generation script for A100 eval/mining."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("PYTORCH_JIT", "0")

from .io_utils import ensure_dir, iter_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--tokenizer")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    rows = list(iter_jsonl(args.input))
    if args.limit > 0:
        rows = rows[: args.limit]

    tokenizer_path = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    quant = None
    if args.load_in_4bit:
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=quant,
    )

    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)

    model.eval()

    ensure_dir(os.path.dirname(args.output) or ".")
    progress = make_progress(rows, args.progress_every)
    start = time.time()

    with open(args.output, "w", encoding="utf-8", newline="\n") as handle:
        for idx, row in enumerate(rows, start=1):
            prompt = row.get("prompt") or row.get("instruction")
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.temperature > 0,
                    temperature=max(args.temperature, 1e-5),
                    top_p=args.top_p,
                    pad_token_id=tokenizer.eos_token_id,
                )

            gen = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
            new_row = dict(row)
            new_row["prediction"] = gen

            handle.write(json.dumps(new_row, ensure_ascii=False) + "\n")
            handle.flush()

            progress.update(idx, row.get("id", ""))

    progress.close()
    elapsed = time.time() - start
    print(f"wrote {args.output} ({len(rows)} rows, {elapsed:.1f}s)")


class SimpleProgress:
    def __init__(self, total: int, every: int) -> None:
        self.total = total
        self.every = max(1, every)
        self.start = time.time()

    def update(self, idx: int, row_id: str) -> None:
        if idx == 1 or idx == self.total or idx % self.every == 0:
            elapsed = time.time() - self.start
            per_item = elapsed / max(1, idx)
            eta = per_item * max(0, self.total - idx)
            print(
                f"[generate] {idx}/{self.total} elapsed={elapsed:.1f}s eta={eta:.1f}s id={row_id}",
                file=sys.stderr,
                flush=True,
            )

    def close(self) -> None:
        pass


class TqdmProgress:
    def __init__(self, total: int) -> None:
        from tqdm.auto import tqdm
        self.bar = tqdm(total=total, desc="generate", dynamic_ncols=True)

    def update(self, idx: int, row_id: str) -> None:
        self.bar.update(1)
        self.bar.set_postfix_str(str(row_id)[:32])

    def close(self) -> None:
        self.bar.close()


def make_progress(rows: List[Dict[str, Any]], every: int) -> SimpleProgress | TqdmProgress:
    if every <= 0:
        return SimpleProgress(len(rows), max(1, len(rows) + 1))
    try:
        return TqdmProgress(len(rows))
    except Exception:
        return SimpleProgress(len(rows), every)


if __name__ == "__main__":
    main()
