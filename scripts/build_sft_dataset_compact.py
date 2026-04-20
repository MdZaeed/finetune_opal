import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_INSTRUCTION = (
    "Optimize the following GPU kernel using the provided performance guidance. "
    "Preserve correctness and external behavior."
)

RESPONSE_CODE_BLOCK_RE = re.compile(r"```(?:[A-Za-z0-9_+-]+)?\n(.*?)```", re.DOTALL)
def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def clean_text(text: str | None) -> str | None:
    if text is None:
        return None
    text = text.strip()
    return text if text else None


def extract_named_block(prompt: str, name: str) -> str | None:
    begin_marker = f"# Begin {name}\n"
    end_marker = f"\n# End {name}"
    start = prompt.find(begin_marker)
    if start == -1:
        return None
    start += len(begin_marker)
    end = prompt.find(end_marker, start)
    if end == -1:
        return None
    return clean_text(prompt[start:end])


def extract_source_code(prompt: str) -> str | None:
    return extract_named_block(prompt, "Source Code")


def extract_performance_guidance(prompt: str) -> str | None:
    guidance_parts = []

    for key in ("Performance Analysis", "H100 Optimization Guidance"):
        value = extract_named_block(prompt, key)
        if value:
            guidance_parts.append(value)

    if guidance_parts:
        return "\n\n".join(guidance_parts)
    return None


def extract_code_only(response: str) -> str | None:
    matches = RESPONSE_CODE_BLOCK_RE.findall(response)
    if matches:
        return matches[0].strip()
    text = response.strip()
    marker = "\noptimizations list ="
    if marker in text:
        text = text.split(marker, 1)[0].strip()
    marker = "\nsuggested_but_not_applied list ="
    if marker in text:
        text = text.split(marker, 1)[0].strip()
    return text or None


def build_example(record: dict[str, Any]) -> dict[str, Any] | None:
    prompt = clean_text(record.get("prompt"))
    response = clean_text(record.get("response"))
    if not prompt or not response:
        return None

    source_code = extract_source_code(prompt)
    performance_guidance = extract_performance_guidance(prompt)
    optimized_code = extract_code_only(response)

    if not source_code or not performance_guidance or not optimized_code:
        return None

    text = (
        f"### Instruction:\n{DEFAULT_INSTRUCTION}\n\n"
        f"### Input:\n"
        f"### Benchmark\n{record['benchmark']}\n\n"
        f"### Variant\n{record['variant']}\n\n"
        f"### Target\nNVIDIA H100 / CUDA\n\n"
        f"### Source Code\n{source_code}\n\n"
        f"### Performance Guidance\n{performance_guidance}\n\n"
        f"### Response:\n{optimized_code}"
    )

    return {
        "id": record["id"],
        "text": text,
        "benchmark": record["benchmark"],
        "model_source": record["model_source"],
        "generator_model": record["generator_model"],
        "variant": record["variant"],
        "iteration": record["iteration"],
        "is_repair": record["is_repair"],
    }


def split_examples(
    examples: list[dict[str, Any]], val_fraction: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shuffled = examples[:]
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * val_fraction)) if shuffled else 0
    val_examples = shuffled[:val_count]
    train_examples = shuffled[val_count:]
    return train_examples, val_examples


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    normalized = {}
    for key, value in payload.items():
        normalized[key] = dict(value) if isinstance(value, Counter) else value
    path.write_text(json.dumps(normalized, indent=2, sort_keys=True), encoding="utf-8")


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(records),
        "benchmarks": Counter(record["benchmark"] for record in records),
        "generator_models": Counter(record["generator_model"] for record in records),
        "repair_records": sum(1 for record in records if record["is_repair"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a compact SFT dataset from parsed raw records.")
    parser.add_argument(
        "--input",
        default="data/parsed_raw_records.jsonl",
        help="Parsed JSONL records from parse_raw_data.py",
    )
    parser.add_argument(
        "--train-output",
        default="data/sft_train_compact.jsonl",
        help="Destination JSONL for the compact training split.",
    )
    parser.add_argument(
        "--val-output",
        default="data/sft_val_compact.jsonl",
        help="Destination JSONL for the compact validation split.",
    )
    parser.add_argument(
        "--manifest",
        default="data/sft_dataset_compact_manifest.json",
        help="Destination manifest summarizing the compact SFT dataset.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Validation fraction for the random train/val split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the train/val split.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    train_output = Path(args.train_output)
    val_output = Path(args.val_output)
    manifest_output = Path(args.manifest)

    raw_records = load_records(input_path)
    examples = []
    skipped = Counter()

    for record in raw_records:
        example = build_example(record)
        if example is None:
            skipped["missing_required_compact_fields"] += 1
            continue
        examples.append(example)

    train_examples, val_examples = split_examples(
        examples, val_fraction=args.val_fraction, seed=args.seed
    )
    write_jsonl(train_output, train_examples)
    write_jsonl(val_output, val_examples)

    manifest = {
        "input_records": len(raw_records),
        "usable_examples": len(examples),
        "skipped": skipped,
        "train": summarize(train_examples),
        "val": summarize(val_examples),
        "val_fraction": args.val_fraction,
        "seed": args.seed,
    }
    write_manifest(manifest_output, manifest)

    print(f"Wrote {len(train_examples)} compact train examples to {train_output}")
    print(f"Wrote {len(val_examples)} compact val examples to {val_output}")
    print(f"Manifest written to {manifest_output}")


if __name__ == "__main__":
    main()
