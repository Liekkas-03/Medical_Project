# -*- coding: utf-8 -*-
"""Evaluate a causal LM on CMExam-style medical multiple-choice questions."""

import argparse
import json
import re
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


CHOICE_LETTERS = "ABCDEFG"


def normalize_answer(text):
    if text is None:
        return ""
    text = str(text).upper()
    match = re.search(r"<answer>\s*([A-G,\s]+)\s*</answer>", text, flags=re.DOTALL)
    if match:
        text = match.group(1)
    letters = re.findall(r"[A-G]", text)
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


def build_prompt(example):
    question = get_field(example, "Question", "question", "stem")
    options = get_field(example, "Options", "options")
    options_text = format_options(options)
    if not question or not options_text:
        return None
    return (
        "Answer the following Chinese medical multiple-choice question. "
        "Output only the answer letters, with no explanation.\n"
        "If there are multiple correct answers, output the letters in alphabetical order, for example ABD.\n\n"
        f"Question: {str(question).strip()}\n\n"
        f"Options:\n{options_text}\n\n"
        "Answer:"
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


def apply_chat_template(tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--lora_model", default=None)
    parser.add_argument("--dataset_name", default="fzkuji/CMExam")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--output_file", default="outputs/eval/cmexam_eval.jsonl")
    args = parser.parse_args()

    dataset = load_cmexam(args).shuffle(seed=args.seed)
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

    rows = []
    prompts = []
    for example in dataset:
        prompt = build_prompt(example)
        gold = normalize_answer(get_field(example, "Answer", "answer"))
        if prompt and gold:
            rows.append({"example": example, "prompt": prompt, "gold": gold})
            prompts.append(apply_chat_template(tokenizer, prompt))

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    correct = 0
    invalid = 0
    with output_path.open("w", encoding="utf-8") as f:
        for start in tqdm(range(0, len(prompts), args.batch_size), desc="Evaluating CMExam"):
            batch_prompts = prompts[start:start + args.batch_size]
            batch_rows = rows[start:start + args.batch_size]
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            prompt_len = inputs["input_ids"].shape[1]
            decoded = tokenizer.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)
            for row, raw_output in zip(batch_rows, decoded):
                pred = normalize_answer(raw_output)
                is_correct = pred == row["gold"]
                total += 1
                correct += int(is_correct)
                invalid += int(pred == "")
                json.dump(
                    {
                        "gold": row["gold"],
                        "pred": pred,
                        "correct": is_correct,
                        "raw_output": raw_output.strip(),
                        "prompt": row["prompt"],
                    },
                    f,
                    ensure_ascii=False,
                )
                f.write("\n")

    accuracy = correct / total if total else 0.0
    invalid_rate = invalid / total if total else 0.0
    print(json.dumps({
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "invalid": invalid,
        "invalid_rate": invalid_rate,
        "output_file": str(output_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
