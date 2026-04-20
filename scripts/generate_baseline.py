import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from unsloth import FastLanguageModel


RESPONSE_MARKER = "\n\n### Response:\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate baseline outputs from an unfine-tuned model on the SFT dataset."
    )
    parser.add_argument("--model-name", required=True, help="Base model to run for baseline inference.")
    parser.add_argument("--input-file", default="data/sft_val.jsonl", help="Input JSONL with SFT examples.")
    parser.add_argument(
        "--output-file",
        default="data/baseline_val_outputs.jsonl",
        help="Destination JSONL for generated outputs.",
    )
    parser.add_argument("--max-seq-length", type=int, default=2048, help="Model max sequence length.")
    parser.add_argument("--max-new-tokens", type=int, default=768, help="Maximum tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature. Use 0 for greedy.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p for sampling when temperature > 0.")
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of examples to run.")
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        default=True,
        help="Load the base model in 4-bit.",
    )
    parser.add_argument("--no-4bit", dest="load_in_4bit", action="store_false", help="Disable 4-bit loading.")
    return parser.parse_args()


def split_prompt_and_reference(text: str) -> tuple[str, str]:
    if RESPONSE_MARKER not in text:
        raise ValueError("Could not find response marker in example text.")
    prompt, reference = text.split(RESPONSE_MARKER, 1)
    return prompt + RESPONSE_MARKER, reference.strip()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("json", data_files=args.input_file, split="train")
    if args.limit > 0:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )
    FastLanguageModel.for_inference(model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    records = []
    for example in dataset:
        prompt_text, reference_text = split_prompt_and_reference(example["text"])
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        generate_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "use_cache": True,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if args.temperature > 0:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = args.temperature
            generate_kwargs["top_p"] = args.top_p
        else:
            generate_kwargs["do_sample"] = False

        with torch.no_grad():
            outputs = model.generate(**inputs, **generate_kwargs)

        generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        records.append(
            {
                "id": example.get("id"),
                "benchmark": example.get("benchmark"),
                "variant": example.get("variant"),
                "generator_model": args.model_name,
                "prompt": prompt_text,
                "reference_response": reference_text,
                "generated_response": generated_text,
            }
        )

    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    print(f"Wrote {len(records)} baseline generations to {output_path}")


if __name__ == "__main__":
    main()
