# -*- coding: utf-8 -*-
"""Build a fixed Huatuo-style evaluation prompt set from local ShareGPT json/jsonl files."""

import argparse
import json
import random
import re
from pathlib import Path


SENSITIVE_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"\b1[3-9]\d{9}\b"),
    re.compile(r"\b\d{17}[\dXx]\b"),
]


def iter_json_records(path):
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            yield from data
        elif isinstance(data, dict):
            yield data


def first_qa_pair(record):
    conversations = record.get("conversations")
    if not isinstance(conversations, list):
        return None

    question = None
    for turn in conversations:
        role = turn.get("from") or turn.get("role")
        value = (turn.get("value") or turn.get("content") or "").strip()
        if role in {"human", "user"} and value:
            question = value
        elif role in {"gpt", "assistant"} and question and value:
            return question, value
    return None


def chinese_ratio(text):
    if not text:
        return 0.0
    zh = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return zh / max(len(text), 1)


def has_sensitive_text(text):
    return any(pattern.search(text) for pattern in SENSITIVE_PATTERNS)


def keep_pair(question, answer, min_q, max_q, min_a, max_a):
    q_len = len(question)
    a_len = len(answer)
    if not (min_q <= q_len <= max_q and min_a <= a_len <= max_a):
        return False
    if chinese_ratio(question + answer) < 0.45:
        return False
    if has_sensitive_text(question) or has_sensitive_text(answer):
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Directory containing ShareGPT json/jsonl files.")
    parser.add_argument("--output_prompts", default="eval/huatuo_baseline/prompts.txt")
    parser.add_argument("--output_refs", default="eval/huatuo_baseline/refs.jsonl")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_question_chars", type=int, default=10)
    parser.add_argument("--max_question_chars", type=int, default=300)
    parser.add_argument("--min_answer_chars", type=int, default=50)
    parser.add_argument("--max_answer_chars", type=int, default=1000)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir does not exist: {data_dir}")

    pairs = []
    seen_questions = set()
    files = sorted(list(data_dir.rglob("*.jsonl")) + list(data_dir.rglob("*.json")))
    if not files:
        raise FileNotFoundError(f"No json/jsonl files found under: {data_dir}")

    for path in files:
        for record in iter_json_records(path):
            pair = first_qa_pair(record)
            if pair is None:
                continue
            question, answer = pair
            norm_question = re.sub(r"\s+", " ", question.strip())
            if norm_question in seen_questions:
                continue
            if not keep_pair(
                norm_question,
                answer,
                args.min_question_chars,
                args.max_question_chars,
                args.min_answer_chars,
                args.max_answer_chars,
            ):
                continue
            seen_questions.add(norm_question)
            pairs.append({"question": norm_question, "reference": answer.strip(), "source": str(path)})

    if len(pairs) < args.num_samples:
        raise ValueError(f"Only found {len(pairs)} valid pairs, need {args.num_samples}.")

    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    selected = pairs[: args.num_samples]

    output_prompts = Path(args.output_prompts)
    output_refs = Path(args.output_refs)
    output_prompts.parent.mkdir(parents=True, exist_ok=True)
    output_refs.parent.mkdir(parents=True, exist_ok=True)

    with output_prompts.open("w", encoding="utf-8") as f:
        for item in selected:
            f.write(item["question"].replace("\n", " ").strip() + "\n")

    with output_refs.open("w", encoding="utf-8") as f:
        for item in selected:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

    print(f"Saved prompts: {output_prompts} ({len(selected)})")
    print(f"Saved refs: {output_refs} ({len(selected)})")


if __name__ == "__main__":
    main()
