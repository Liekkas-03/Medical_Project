# -*- coding: utf-8 -*-
"""Evaluate a causal LM on PubMedQA final decision prediction."""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List

import torch
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


LABELS = ("yes", "no", "maybe")


def normalize_label(text: str) -> str:
    if text is None:
        return ""
    text = str(text).strip().lower()
    return text if text in LABELS else ""


def extract_direct_label(text: str) -> str:
    """Parse direct-mode output, where the whole response should be one label."""
    if text is None:
        return ""
    text = str(text).strip().lower()
    match = re.fullmatch(r"\s*(yes|no|maybe)\s*[\.。!！]?\s*", text)
    return match.group(1) if match else ""


def extract_final_decision_label(text: str) -> str:
    """Parse rationale-mode output from the explicit final decision line only."""
    if text is None:
        return ""
    text = str(text).strip().lower()
    matches = re.findall(
        r"\bfinal\s*decision\s*[:：\-]\s*(yes|no|maybe)\b",
        text,
        flags=re.MULTILINE,
    )
    return matches[-1] if matches else ""


def extract_prediction_label(text: str, prompt_style: str) -> str:
    if prompt_style == "rationale":
        return extract_final_decision_label(text)
    return extract_direct_label(text)


def context_to_text(context, max_context_chars: int) -> str:
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


def build_prompt(example: Dict, max_context_chars: int, prompt_style: str) -> str:
    question = str(example.get("question", "")).strip()
    context = context_to_text(example.get("context"), max_context_chars)
    if not question or not context:
        return ""
    if prompt_style == "rationale":
        return (
            "Answer the biomedical research question using only the provided PubMed abstract context.\n"
            "Give a brief evidence-based explanation, then end with exactly one line: "
            "Final decision: yes/no/maybe.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question}"
        )
    return (
        "Answer the biomedical research question using only the provided PubMed abstract context.\n"
        "Choose exactly one final decision from: yes, no, maybe.\n"
        "Output only the final decision word.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Final decision:"
    )


def apply_chat_template(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def macro_f1(golds: Iterable[str], preds: Iterable[str]) -> float:
    golds = list(golds)
    preds = list(preds)
    scores = []
    for label in LABELS:
        tp = sum(1 for gold, pred in zip(golds, preds) if gold == label and pred == label)
        fp = sum(1 for gold, pred in zip(golds, preds) if gold != label and pred == label)
        fn = sum(1 for gold, pred in zip(golds, preds) if gold == label and pred != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        scores.append(score)
    return sum(scores) / len(scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--lora_model", default=None)
    parser.add_argument("--dataset_name", default="qiaojin/PubMedQA")
    parser.add_argument("--dataset_config", default="pqa_labeled")
    parser.add_argument("--split", default="train")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_context_chars", type=int, default=2400)
    parser.add_argument("--prompt_style", choices=("direct", "rationale"), default="direct")
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--output_file", default="outputs/eval/pubmedqa_eval.jsonl")
    args = parser.parse_args()
    max_new_tokens = args.max_new_tokens
    if max_new_tokens is None:
        max_new_tokens = 128 if args.prompt_style == "rationale" else 8

    dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.split, cache_dir=args.cache_dir)
    dataset = dataset.shuffle(seed=args.seed)
    if args.max_samples and args.max_samples > 0:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    if args.lora_model:
        model = PeftModel.from_pretrained(model, args.lora_model)
    model.eval()

    rows: List[Dict] = []
    prompts = []
    for example in dataset:
        prompt = build_prompt(example, args.max_context_chars, args.prompt_style)
        gold = normalize_label(example.get("final_decision"))
        if prompt and gold:
            rows.append({
                "question": example.get("question"),
                "gold": gold,
                "prompt": prompt,
            })
            prompts.append(apply_chat_template(tokenizer, prompt))

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    preds = []
    golds = []
    invalid = 0
    with output_path.open("w", encoding="utf-8") as f:
        for start in tqdm(range(0, len(prompts), args.batch_size), desc="Evaluating PubMedQA"):
            batch_prompts = prompts[start:start + args.batch_size]
            batch_rows = rows[start:start + args.batch_size]
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            prompt_len = inputs["input_ids"].shape[1]
            decoded = tokenizer.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)

            for row, raw_output in zip(batch_rows, decoded):
                pred = extract_prediction_label(raw_output, args.prompt_style)
                invalid += int(pred == "")
                golds.append(row["gold"])
                preds.append(pred)
                json.dump(
                    {
                        "question": row["question"],
                        "gold": row["gold"],
                        "pred": pred,
                        "correct": pred == row["gold"],
                        "raw_output": raw_output.strip(),
                        "prompt": row["prompt"],
                    },
                    f,
                    ensure_ascii=False,
                )
                f.write("\n")

    total = len(golds)
    correct = sum(1 for gold, pred in zip(golds, preds) if gold == pred)
    result = {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "macro_f1": macro_f1(golds, preds) if total else 0.0,
        "invalid": invalid,
        "invalid_rate": invalid / total if total else 0.0,
        "prompt_style": args.prompt_style,
        "max_new_tokens": max_new_tokens,
        "seed": args.seed,
        "output_file": str(output_path),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
