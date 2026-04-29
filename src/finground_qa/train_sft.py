"""QLoRA SFT entrypoint for FinGround-QA."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .io_utils import iter_jsonl


class SftJsonlDataset(Dataset):
    def __init__(self, path: str, tokenizer: Any, max_length: int) -> None:
        self.rows = list(iter_jsonl(path))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]
        prompt = row.get("instruction", "")
        target = row.get("output", "")
        messages = [{"role": "user", "content": prompt}]
        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full_text = prompt_text + target + self.tokenizer.eos_token
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full = self.tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=self.max_length)
        input_ids = torch.tensor(full["input_ids"], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        labels[: min(len(prompt_ids), labels.numel())] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


@dataclass
class CausalCollator:
    pad_token_id: int

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = pad_sequence([f["input_ids"] for f in features], batch_first=True, padding_value=self.pad_token_id)
        attention_mask = pad_sequence([f["attention_mask"] for f in features], batch_first=True, padding_value=0)
        labels = pad_sequence([f["labels"] for f in features], batch_first=True, padding_value=-100)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def make_training_args(cls: Any, **kwargs: Any) -> Any:
    try:
        return cls(evaluation_strategy="steps", **kwargs)
    except TypeError:
        return cls(eval_strategy="steps", **kwargs)


def report_to_list(value: str) -> List[str]:
    if value.lower() in {"", "none", "off", "false", "disabled"}:
        return []
    return [value]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-file", default="data/sft/sft_train.jsonl")
    parser.add_argument("--val-file", default="data/sft/sft_val.jsonl")
    parser.add_argument("--output-dir", default="results/sft/qwen25_7b_finground_sft")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--run-name", default="finground-sft")
    args = parser.parse_args()

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
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
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_config)

    train_dataset = SftJsonlDataset(args.train_file, tokenizer, args.max_length)
    eval_dataset = SftJsonlDataset(args.val_file, tokenizer, args.max_length)
    training_args = make_training_args(
        TrainingArguments,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_strategy="steps",
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to=report_to_list(args.report_to),
        run_name=args.run_name,
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=CausalCollator(tokenizer.pad_token_id),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    with open(f"{args.output_dir}/training_args_summary.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
