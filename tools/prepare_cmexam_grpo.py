# -*- coding: utf-8 -*-
"""Convert CMExam-style multiple-choice data to GRPO jsonl format."""

import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset


CHOICE_LETTERS = "ABCDEFG"


def normalize_answer(text):
    if text is None:
        return ""
    letters = re.findall(r"[A-G]", str(text).upper())
    if not letters:
        return ""
    return "".join(letter for letter in CHOICE_LETTERS if letter in set(letters))


def get_field(example, *names):
    lower_map = {str(k).lower(): v for k, v in example.items()}
    for name in names:
        if name in example and example[name] is not None:
            return example[name]
        value = lower_map.get(name.lower())
        if value is not None:
            return value
    return None


def parse_options(options):
    if isinstance(options, dict):
        items = sorted(options.items(), key=lambda kv: str(kv[0]))
        return [(str(k).strip().upper(), str(v).strip()) for k, v in items]
    if isinstance(options, list):
        return [(CHOICE_LETTERS[i], str(v).strip()) for i, v in enumerate(options) if i < len(CHOICE_LETTERS)]
    if isinstance(options, str):
        options = options.strip()
        try:
            parsed = json.loads(options)
            return parse_options(parsed)
        except Exception:
            lines = [line.strip() for line in options.splitlines() if line.strip()]
            parsed_lines = []
            for line in lines:
                match = re.match(r"^([A-Ga-g])[\.\):\s]+(.+)$", line)
                if match:
                    parsed_lines.append((match.group(1).upper(), match.group(2).strip()))
            if parsed_lines:
                return parsed_lines
            return [("", options)]
    return []


def format_options(options):
    parsed = parse_options(options)
    if not parsed:
        return ""
    lines = []
    for letter, value in parsed:
        lines.append(f"{letter}. {value}" if letter else value)
    return "\n".join(lines)


def build_grpo_question(example):
    question = get_field(example, "Question", "question", "stem")
    options = get_field(example, "Options", "options")
    options_text = format_options(options)
    if not question or not options_text:
        return None
    return (
        "Answer the following Chinese medical multiple-choice question.\n"
        "First provide brief reasoning inside <think>...</think>, then write only the answer letters inside "
        "<answer>...</answer>.\n"
        "If there are multiple correct answers, output the letters in alphabetical order, for example "
        "<answer>ABD</answer>.\n\n"
        f"Question: {str(question).strip()}\n\n"
        f"Options:\n{options_text}"
    )


def find_local_files(data_dir, split):
    data_dir = Path(data_dir)
    candidates = []
    for suffix in ("*.parquet", "*.jsonl", "*.json", "*.csv"):
        candidates.extend(data_dir.rglob(suffix))
    split = split.lower()
    matched = [p for p in candidates if split in str(p).lower()]
    files = matched or candidates
    if not files:
        raise FileNotFoundError(f"No dataset files found under {data_dir}")
    by_suffix = {}
    for path in files:
        by_suffix.setdefault(path.suffix.lower(), []).append(str(path))
    for suffix, dataset_type in ((".parquet", "parquet"), (".jsonl", "json"), (".json", "json"), (".csv", "csv")):
        if suffix in by_suffix:
            return dataset_type, sorted(by_suffix[suffix])
    raise FileNotFoundError(f"No supported dataset files found under {data_dir}")


def load_cmexam(args):
    if args.data_dir:
        dataset_type, files = find_local_files(args.data_dir, args.split)
        return load_dataset(dataset_type, data_files=files, split="train", cache_dir=args.cache_dir)
    return load_dataset(args.dataset_name, split=args.split, cache_dir=args.cache_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="fzkuji/CMExam")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_file", default="data/cmexam_grpo/train.jsonl")
    args = parser.parse_args()

    dataset = load_cmexam(args).shuffle(seed=args.seed)
    if args.max_samples and args.max_samples > 0:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for example in dataset:
            question = build_grpo_question(example)
            answer = normalize_answer(get_field(example, "Answer", "answer"))
            if not question or not answer:
                continue
            json.dump({"question": question, "answer": answer}, f, ensure_ascii=False)
            f.write("\n")
            count += 1

    print(f"Saved {count} GRPO samples to {output_path}")


if __name__ == "__main__":
    main()
