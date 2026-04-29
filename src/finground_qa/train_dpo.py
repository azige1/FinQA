"""QLoRA DPO entrypoint for FinGround-QA."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import torch
from datasets import Dataset

from .io_utils import read_jsonl


def load_pairs(path: str, limit: int = -1) -> Dataset:
    rows = read_jsonl(path)
    if limit > 0:
        rows = rows[:limit]
    items: List[Dict[str, str]] = []
    for row in rows:
        items.append(
            {
                "prompt": row.get("prompt", ""),
                "chosen": row.get("chosen", ""),
                "rejected": row.get("rejected", ""),
            }
        )
    return Dataset.from_list(items)


def make_training_args(cls: Any, eval_enabled: bool, **kwargs: Any) -> Any:
    strategy = "steps" if eval_enabled else "no"
    try:
        return cls(evaluation_strategy=strategy, **kwargs)
    except TypeError:
        return cls(eval_strategy=strategy, **kwargs)


def report_to_list(value: str) -> List[str]:
    if value.lower() in {"", "none", "off", "false", "disabled"}:
        return []
    return [value]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Base model path.")
    parser.add_argument("--sft-adapter", help="Optional SFT LoRA adapter to initialize from.")
    parser.add_argument("--train-file", default="data/dpo/dpo_balanced_v1.jsonl")
    parser.add_argument("--val-file", default="")
    parser.add_argument("--output-dir", default="results/dpo/qwen25_7b_finground_dpo")
    parser.add_argument("--max-source-length", type=int, default=1024)
    parser.add_argument("--max-target-length", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--run-name", default="finground-dpo")
    parser.add_argument(
        "--attn-implementation",
        default=os.environ.get("ATTN_IMPLEMENTATION", "eager"),
        help="Transformers attention backend. Use eager to avoid cuDNN SDPA plan failures.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("GRADIENT_CHECKPOINTING", "1") == "1",
        help="Enable gradient checkpointing. Default on because full-length DPO needs it on A800 40GB.",
    )
    args = parser.parse_args()

    from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
    from trl import DPOTrainer

    try:
        from trl import DPOConfig
    except ImportError:
        DPOConfig = None

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.eos_token is None and tokenizer.pad_token is not None:
        tokenizer.eos_token = tokenizer.pad_token
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token
    if tokenizer.bos_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.bos_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token_id is None or tokenizer.bos_token_id is None:
        raise ValueError(
            "DPO training requires non-null pad_token_id and bos_token_id; "
            f"got pad={tokenizer.pad_token_id}, bos={tokenizer.bos_token_id}."
        )
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=quant,
        attn_implementation=args.attn_implementation,
    )
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    try:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    except TypeError:
        model = prepare_model_for_kbit_training(model)
    if args.sft_adapter:
        model = PeftModel.from_pretrained(model, args.sft_adapter, is_trainable=True)
        peft_config = None
    else:
        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if args.gradient_checkpointing:
        input_embeddings = model.get_input_embeddings()

        def make_inputs_require_grad(_module: Any, _inputs: Any, output: Any) -> None:
            output.requires_grad_(True)

        input_embeddings.register_forward_hook(make_inputs_require_grad)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    train_dataset = load_pairs(args.train_file, args.limit)
    eval_dataset = load_pairs(args.val_file) if args.val_file else None
    max_length = args.max_source_length + args.max_target_length
    training_kwargs: Dict[str, Any] = {
        "output_dir": args.output_dir,
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "bf16": True,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "save_strategy": "steps",
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine",
        "gradient_checkpointing": args.gradient_checkpointing,
        "optim": "paged_adamw_8bit",
        "report_to": report_to_list(args.report_to),
        "run_name": args.run_name,
        "seed": args.seed,
        "remove_unused_columns": False,
    }
    trainer_extra: Dict[str, Any] = {}
    if DPOConfig is not None:
        training_cls = DPOConfig
        training_kwargs.update(
            {
                "beta": args.beta,
                "max_prompt_length": args.max_source_length,
                "max_target_length": args.max_target_length,
                "max_length": max_length,
            }
        )
    else:
        training_cls = TrainingArguments
        trainer_extra.update(
            {
                "beta": args.beta,
                "max_prompt_length": args.max_source_length,
                "max_length": max_length,
            }
        )
    training_args = make_training_args(
        training_cls,
        eval_enabled=eval_dataset is not None,
        **training_kwargs,
    )
    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "ref_model": None,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "tokenizer": tokenizer,
    }
    trainer_kwargs.update(trainer_extra)
    if peft_config is not None:
        trainer_kwargs["peft_config"] = peft_config
    try:
        trainer = DPOTrainer(**trainer_kwargs)
    except TypeError:
        trainer_kwargs.pop("tokenizer", None)
        trainer_kwargs["processing_class"] = tokenizer
        trainer = DPOTrainer(**trainer_kwargs)

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    with open(f"{args.output_dir}/training_args_summary.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
