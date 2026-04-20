import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ARTIFACT_PATTERNS = [
    re.compile(r"^(?P<variant>.+)_(?P<model>gpt5|gemini25pro)_repair_prompt_(?P<iter>\d+)$"),
    re.compile(r"^(?P<variant>.+)_(?P<model>gpt5|gemini25pro)_prompt(?:_(?P<iter>\d+))?$"),
    re.compile(r"^(?P<variant>.+)_(?P<model>gpt5|gemini25pro)_response_(?P<iter>\d+)$"),
    re.compile(r"^(?P<variant>.+)_(?P<model>gpt5|gemini25pro)_build_err_iter_(?P<iter>\d+)$"),
    re.compile(r"^(?P<variant>.+)_(?P<model>gpt5|gemini25pro)_build_errors_iter_(?P<iter>\d+)$"),
    re.compile(r"^(?P<variant>.+)_(?P<model>gpt5|gemini25pro)_build_out_iter_(?P<iter>\d+)$"),
    re.compile(r"^(?P<variant>.+)_(?P<model>gpt5|gemini25pro)_runtime_iter_(?P<iter>\d+)$"),
    re.compile(r"^(?P<variant>.+)_(?P<model>gpt5|gemini25pro)_average_runtime_iter_(?P<iter>\d+)$"),
    re.compile(r"^(?P<variant>.+)_(?P<model>gpt5|gemini25pro)_runtime_extract_err_iter_(?P<iter>\d+)$"),
]

NUMERIC_VALUE_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def artifact_key(stem: str) -> tuple[str, dict[str, Any]] | None:
    for pattern in ARTIFACT_PATTERNS:
        match = pattern.match(stem)
        if not match:
            continue
        data = match.groupdict()
        if "_repair_prompt_" in stem:
            return "repair_prompt", data
        if "_prompt" in stem and "_repair_prompt_" not in stem:
            return "prompt", data
        if "_response_" in stem:
            return "response", data
        if "_average_runtime_iter_" in stem:
            return "average_runtime", data
        if "_build_err_iter_" in stem:
            return "build_err", data
        if "_build_errors_iter_" in stem:
            return "build_errors", data
        if "_build_out_iter_" in stem:
            return "build_out", data
        if "_runtime_iter_" in stem:
            return "runtime", data
        if "_runtime_extract_err_iter_" in stem:
            return "runtime_extract_err", data
    return None


def parse_benchmark(folder_name: str) -> tuple[str, str]:
    benchmark, _, source = folder_name.partition("-")
    return benchmark, source or folder_name


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").strip()


def maybe_parse_numeric(text: str) -> float | None:
    match = NUMERIC_VALUE_RE.search(text)
    return float(match.group(0)) if match else None


def build_records(data_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, int], dict[str, Any]] = {}
    skipped = Counter()
    auxiliaries = Counter()

    for path in sorted(data_root.rglob("*.txt")):
        rel_path = path.relative_to(data_root)
        if len(rel_path.parts) < 2:
            skipped["unexpected_path_depth"] += 1
            continue

        folder = rel_path.parts[0]
        match = artifact_key(path.stem)
        if match is None:
            auxiliaries[path.suffix] += 1
            skipped["unparsed_txt"] += 1
            continue

        artifact_type, data = match
        benchmark, model_source = parse_benchmark(folder)
        variant = data["variant"]
        model = data["model"]
        iteration = int(data.get("iter") or 1)

        key = (folder, benchmark, model_source, variant, iteration)
        record = grouped.setdefault(
            key,
            {
                "id": f"{folder}:{variant}:iter{iteration}",
                "folder": folder,
                "benchmark": benchmark,
                "model_source": model_source,
                "generator_model": model,
                "variant": variant,
                "iteration": iteration,
                "is_repair": False,
                "prompt_path": None,
                "response_path": None,
                "build_err_path": None,
                "build_errors_path": None,
                "build_out_path": None,
                "runtime_path": None,
                "average_runtime_path": None,
                "runtime_extract_err_path": None,
                "prompt": None,
                "response": None,
                "build_err": None,
                "build_errors": None,
                "build_out": None,
                "runtime_raw": None,
                "runtime_ms": None,
                "average_runtime_raw": None,
                "average_runtime_ms": None,
                "runtime_extract_err": None,
            },
        )

        text = read_text(path)
        path_str = str(rel_path)

        if artifact_type == "prompt":
            record["prompt_path"] = path_str
            record["prompt"] = text
        elif artifact_type == "repair_prompt":
            record["prompt_path"] = path_str
            record["prompt"] = text
            record["is_repair"] = True
        elif artifact_type == "response":
            record["response_path"] = path_str
            record["response"] = text
        elif artifact_type == "build_err":
            record["build_err_path"] = path_str
            record["build_err"] = text
        elif artifact_type == "build_errors":
            record["build_errors_path"] = path_str
            record["build_errors"] = text
        elif artifact_type == "build_out":
            record["build_out_path"] = path_str
            record["build_out"] = text
        elif artifact_type == "runtime":
            record["runtime_path"] = path_str
            record["runtime_raw"] = text
            record["runtime_ms"] = maybe_parse_numeric(text)
        elif artifact_type == "average_runtime":
            record["average_runtime_path"] = path_str
            record["average_runtime_raw"] = text
            record["average_runtime_ms"] = maybe_parse_numeric(text)
        elif artifact_type == "runtime_extract_err":
            record["runtime_extract_err_path"] = path_str
            record["runtime_extract_err"] = text

    records = sorted(
        grouped.values(),
        key=lambda record: (
            record["folder"],
            record["variant"],
            record["iteration"],
        ),
    )

    summary = {
        "total_records": len(records),
        "folders": Counter(record["folder"] for record in records),
        "benchmarks": Counter(record["benchmark"] for record in records),
        "generator_models": Counter(record["generator_model"] for record in records),
        "repair_records": sum(1 for record in records if record["is_repair"]),
        "records_with_prompt": sum(1 for record in records if record["prompt"]),
        "records_with_response": sum(1 for record in records if record["response"]),
        "records_with_runtime": sum(1 for record in records if record["runtime_ms"] is not None),
        "records_with_average_runtime": sum(
            1 for record in records if record["average_runtime_ms"] is not None
        ),
        "records_with_build_err": sum(1 for record in records if record["build_err"]),
        "records_with_build_errors": sum(1 for record in records if record["build_errors"]),
        "records_with_build_out": sum(1 for record in records if record["build_out"]),
        "skipped": skipped,
        "auxiliaries": auxiliaries,
    }
    return records, summary


def write_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def write_manifest(summary: dict[str, Any], output_path: Path) -> None:
    normalized = {}
    for key, value in summary.items():
        normalized[key] = dict(value) if isinstance(value, Counter) else value
    output_path.write_text(json.dumps(normalized, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse raw optimization artifacts into JSONL.")
    parser.add_argument("--data-root", default="data", help="Root directory containing raw data folders.")
    parser.add_argument(
        "--output",
        default="data/parsed_raw_records.jsonl",
        help="Destination JSONL path for parsed records.",
    )
    parser.add_argument(
        "--manifest",
        default="data/parsed_raw_records_manifest.json",
        help="Destination JSON manifest path for parse summary.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_path = Path(args.output)
    manifest_path = Path(args.manifest)

    records, summary = build_records(data_root)
    write_jsonl(records, output_path)
    write_manifest(summary, manifest_path)

    print(f"Wrote {len(records)} records to {output_path}")
    print(f"Summary manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
