import argparse
import inspect
from pathlib import Path

import unsloth
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer


TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune an instruct model with Unsloth SFT.")
    parser.add_argument("--model-name", required=True, help="Base model to fine-tune.")
    parser.add_argument("--train-file", default="data/sft_train.jsonl", help="Training JSONL path.")
    parser.add_argument("--val-file", default="data/sft_val.jsonl", help="Validation JSONL path.")
    parser.add_argument("--output-dir", default="runs/kernel-sft", help="Checkpoint/output directory.")
    parser.add_argument("--max-seq-length", type=int, default=2048, help="Maximum sequence length.")
    parser.add_argument("--load-in-4bit", action="store_true", default=True, help="Load the base model in 4-bit.")
    parser.add_argument("--no-4bit", dest="load_in_4bit", action="store_false", help="Disable 4-bit loading.")
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha.")
    parser.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA dropout.")
    parser.add_argument("--per-device-batch-size", type=int, default=1, help="Per-device train batch size.")
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
        help="Gradient accumulation steps.",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--warmup-ratio", type=float, default=0.03, help="Warmup ratio.")
    parser.add_argument("--num-train-epochs", type=float, default=3.0, help="Number of training epochs.")
    parser.add_argument("--max-steps", type=int, default=-1, help="Override training steps. Use -1 for epochs.")
    parser.add_argument("--logging-steps", type=int, default=1, help="Logging frequency in steps.")
    parser.add_argument("--save-steps", type=int, default=50, help="Checkpoint save frequency in steps.")
    parser.add_argument("--eval-steps", type=int, default=50, help="Evaluation frequency in steps.")
    parser.add_argument("--save-total-limit", type=int, default=3, help="How many checkpoints to retain.")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint under --output-dir if one exists.",
    )
    parser.add_argument(
        "--save-adapter-dir",
        default=None,
        help="Optional explicit directory for the final LoRA adapter. Defaults to <output-dir>/final_adapter.",
    )
    return parser.parse_args()


def latest_checkpoint(output_dir: Path) -> str | None:
    checkpoints = []
    for child in output_dir.iterdir():
        if not child.is_dir() or not child.name.startswith("checkpoint-"):
            continue
        try:
            step = int(child.name.split("-", 1)[1])
        except (IndexError, ValueError):
            continue
        checkpoints.append((step, child))
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda item: item[0])
    return str(checkpoints[-1][1])


def build_datasets(train_file: str, val_file: str):
    train_dataset = load_dataset("json", data_files=train_file, split="train")
    val_path = Path(val_file)
    eval_dataset = None
    if val_path.exists() and val_path.stat().st_size > 0:
        eval_dataset = load_dataset("json", data_files=val_file, split="train")
    return train_dataset, eval_dataset


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, eval_dataset = build_datasets(args.train_file, args.val_file)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=TARGET_MODULES,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        max_seq_length=args.max_seq_length,
        use_rslora=False,
        loftq_config=None,
    )

    eval_strategy = "steps" if eval_dataset is not None else "no"

    sft_config_kwargs = {
        "dataset_text_field": "text",
        "per_device_train_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "warmup_ratio": args.warmup_ratio,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_strategy": eval_strategy,
        "eval_steps": args.eval_steps,
        "save_total_limit": args.save_total_limit,
        "optim": "adamw_8bit",
        "lr_scheduler_type": "cosine",
        "seed": args.seed,
        "output_dir": str(output_dir),
        "report_to": "none",
        "packing": False,
        "fp16": not is_bfloat16_supported(),
        "bf16": is_bfloat16_supported(),
    }
    sft_signature = inspect.signature(SFTConfig.__init__).parameters
    if "max_seq_length" in sft_signature:
        sft_config_kwargs["max_seq_length"] = args.max_seq_length
    elif "max_length" in sft_signature:
        sft_config_kwargs["max_length"] = args.max_seq_length

    trainer_kwargs = {
        "model": model,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "args": SFTConfig(**sft_config_kwargs),
    }
    trainer_signature = inspect.signature(SFTTrainer.__init__).parameters
    if "tokenizer" in trainer_signature:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_signature:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = SFTTrainer(**trainer_kwargs)

    resume_from_checkpoint = latest_checkpoint(output_dir) if args.resume else None
    if resume_from_checkpoint:
        print(f"Resuming from checkpoint: {resume_from_checkpoint}")

    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    print(train_result)

    adapter_dir = (
        Path(args.save_adapter_dir)
        if args.save_adapter_dir is not None
        else output_dir / "final_adapter"
    )
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    trainer.save_state()
    print(f"Saved final adapter to {adapter_dir}")


if __name__ == "__main__":
    main()
