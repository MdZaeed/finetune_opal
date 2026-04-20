import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_INSTRUCTION = (
    "Optimize the following GPU kernel or related accelerator code. "
    "Preserve correctness, keep the external behavior intact, and use any "
    "available build or runtime feedback to improve the implementation."
)


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def clean_text(text: str | None) -> str | None:
    if text is None:
        return None
    text = text.strip()
    return text if text else None


def build_input_sections(record: dict[str, Any], include_feedback: bool) -> str:
    sections = [
        f"### Benchmark\n{record['benchmark']}",
        f"### Source\n{record['model_source']}",
        f"### Variant\n{record['variant']}",
        f"### Iteration\n{record['iteration']}",
    ]

    prompt = clean_text(record.get("prompt"))
    if prompt:
        sections.append(f"### Prompt\n{prompt}")

    if include_feedback:
        build_err = clean_text(record.get("build_err"))
        build_errors = clean_text(record.get("build_errors"))
        build_out = clean_text(record.get("build_out"))
        runtime_extract_err = clean_text(record.get("runtime_extract_err"))

        feedback_parts = []
        if build_err:
            feedback_parts.append(f"#### Build stderr\n{build_err}")
        if build_errors:
            feedback_parts.append(f"#### Build errors\n{build_errors}")
        if build_out:
            feedback_parts.append(f"#### Build output\n{build_out}")
        if record.get("runtime_ms") is not None:
            feedback_parts.append(f"#### Runtime (ms)\n{record['runtime_ms']}")
        if record.get("average_runtime_ms") is not None:
            feedback_parts.append(
                f"#### Average runtime (ms)\n{record['average_runtime_ms']}"
            )
        if runtime_extract_err:
            feedback_parts.append(f"#### Runtime extraction error\n{runtime_extract_err}")

        if feedback_parts:
            sections.append("### Feedback\n" + "\n\n".join(feedback_parts))

    return "\n\n".join(sections)


def build_example(record: dict[str, Any], include_feedback: bool) -> dict[str, Any] | None:
    prompt = clean_text(record.get("prompt"))
    response = clean_text(record.get("response"))
    if not prompt or not response:
        return None

    input_context = build_input_sections(record, include_feedback=include_feedback)
    text = (
        f"### Instruction:\n{DEFAULT_INSTRUCTION}\n\n"
        f"### Input:\n{input_context}\n\n"
        f"### Response:\n{response}"
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
        "runtime_ms": record.get("runtime_ms"),
        "average_runtime_ms": record.get("average_runtime_ms"),
        "has_build_err": bool(clean_text(record.get("build_err"))),
        "has_build_errors": bool(clean_text(record.get("build_errors"))),
        "has_build_out": bool(clean_text(record.get("build_out"))),
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
        "with_runtime_ms": sum(1 for record in records if record["runtime_ms"] is not None),
        "with_average_runtime_ms": sum(
            1 for record in records if record["average_runtime_ms"] is not None
        ),
        "with_build_err": sum(1 for record in records if record["has_build_err"]),
        "with_build_errors": sum(1 for record in records if record["has_build_errors"]),
        "with_build_out": sum(1 for record in records if record["has_build_out"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an SFT dataset from parsed raw records.")
    parser.add_argument(
        "--input",
        default="data/parsed_raw_records.jsonl",
        help="Parsed JSONL records from parse_raw_data.py",
    )
    parser.add_argument(
        "--train-output",
        default="data/sft_train.jsonl",
        help="Destination JSONL for the training split.",
    )
    parser.add_argument(
        "--val-output",
        default="data/sft_val.jsonl",
        help="Destination JSONL for the validation split.",
    )
    parser.add_argument(
        "--manifest",
        default="data/sft_dataset_manifest.json",
        help="Destination manifest summarizing the generated SFT dataset.",
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
    parser.add_argument(
        "--no-feedback",
        action="store_true",
        help="Exclude build/runtime feedback from the input section.",
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
        example = build_example(record, include_feedback=not args.no_feedback)
        if example is None:
            skipped["missing_prompt_or_response"] += 1
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
        "include_feedback": not args.no_feedback,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
    }
    write_manifest(manifest_output, manifest)

    print(f"Wrote {len(train_examples)} train examples to {train_output}")
    print(f"Wrote {len(val_examples)} val examples to {val_output}")
    print(f"Manifest written to {manifest_output}")


if __name__ == "__main__":
    main()
