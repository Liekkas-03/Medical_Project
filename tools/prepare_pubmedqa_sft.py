# -*- coding: utf-8 -*-
"""Convert PubMedQA to MedicalGPT ShareGPT SFT jsonl format."""

import argparse
import json
from pathlib import Path

from datasets import load_dataset


LABELS = {"yes", "no", "maybe"}


def normalize_label(text):
    text = str(text or "").strip().lower()
    return text if text in LABELS else ""


def context_to_text(context, max_context_chars):
    if isinstance(context, dict):
        contexts = context.get("contexts") or context.get("context") or []
        labels = context.get("labels") or []
        if isinstance(contexts, str):
            contexts = [contexts]
        if isinstance(labels, str):
            labels = [labels]
        lines = []
        for idx, item in enumerate(contexts):
            prefix = ""
            if idx < len(labels) and labels[idx]:
                prefix = f"{str(labels[idx]).strip()}: "
            lines.append(prefix + str(item).strip())
        text = "\n".join(line for line in lines if line.strip())
    elif isinstance(context, list):
        text = "\n".join(str(item).strip() for item in context if str(item).strip())
    else:
        text = str(context or "").strip()

    if max_context_chars > 0 and len(text) > max_context_chars:
        text = text[:max_context_chars].rsplit(" ", 1)[0]
    return text


def build_prompt(example, max_context_chars):
    question = str(example.get("question", "")).strip()
    context = context_to_text(example.get("context"), max_context_chars)
    if not question or not context:
        return ""
    return (
        "Answer the biomedical research question using only the provided PubMed abstract context.\n"
        "Give a brief evidence-based explanation, then end with exactly one line: Final decision: yes/no/maybe.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}"
    )


def build_response(example):
    label = normalize_label(example.get("final_decision"))
    if not label:
        return ""
    long_answer = str(example.get("long_answer") or "").strip()
    if not long_answer:
        long_answer = "The final decision follows from the evidence in the provided context."
    return f"{long_answer}\nFinal decision: {label}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="qiaojin/PubMedQA")
    parser.add_argument("--dataset_config", default="pqa_artificial")
    parser.add_argument("--split", default="train")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--max_samples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_context_chars", type=int, default=2400)
    parser.add_argument("--output_file", default="data/pubmedqa_sft/train/train.jsonl")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.split, cache_dir=args.cache_dir)
    dataset = dataset.shuffle(seed=args.seed)
    if args.max_samples and args.max_samples > 0:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for example in dataset:
            prompt = build_prompt(example, args.max_context_chars)
            response = build_response(example)
            if not prompt or not response:
                continue
            json.dump(
                {
                    "conversations": [
                        {"from": "human", "value": prompt},
                        {"from": "gpt", "value": response},
                    ]
                },
                f,
                ensure_ascii=False,
            )
            f.write("\n")
            count += 1

    print(f"Saved {count} SFT samples to {output_path}")


if __name__ == "__main__":
    main()
