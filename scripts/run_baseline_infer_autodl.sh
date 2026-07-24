#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/root/autodl-tmp/hf-cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/root/autodl-tmp/hf-cache/datasets}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/models/Qwen2.5-3B-Instruct}"
DATA_DIR="${DATA_DIR:-/root/autodl-tmp/datasets/huatuo_medical_qa_sharegpt}"
OUT_DIR="${OUT_DIR:-eval/huatuo_baseline}"
NUM_SAMPLES="${NUM_SAMPLES:-50}"
SEED="${SEED:-42}"

mkdir -p "${OUT_DIR}"

python tools/make_huatuo_eval_prompts.py \
  --data_dir "${DATA_DIR}" \
  --num_samples "${NUM_SAMPLES}" \
  --seed "${SEED}" \
  --output_prompts "${OUT_DIR}/prompts.txt" \
  --output_refs "${OUT_DIR}/refs.jsonl"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python demo/inference.py \
  --base_model "${BASE_MODEL}" \
  --data_file "${OUT_DIR}/prompts.txt" \
  --output_file "${OUT_DIR}/base_outputs.jsonl" \
  --eval_batch_size 4 \
  --max_new_tokens 256 \
  --temperature 0
