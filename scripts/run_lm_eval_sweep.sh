#!/usr/bin/env bash
set -euo pipefail

: "${MODEL:?Set MODEL to a Hugging Face model ID or local checkpoint path}"

TASKS="${TASKS:-mmlu gsm8k ruler_cwe ruler_qa_hotpot}"
QUANTIZATION="${QUANTIZATION:-none}"
KEEP_RATIOS="${KEEP_RATIOS:-1.0 0.8 0.5}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_LENGTH="${MAX_LENGTH:-16384}"
RULER_LENGTHS="${RULER_LENGTHS:-4096 8192 16384}"
OUTPUT_DIR="${OUTPUT_DIR:-results/lm_eval}"
LIMIT="${LIMIT:-}"

command=(
  python scripts/evaluate_lm_eval.py
  --model "${MODEL}"
  --tasks ${TASKS}
  --quantization ${QUANTIZATION}
  --keep-ratios ${KEEP_RATIOS}
  --batch-size "${BATCH_SIZE}"
  --max-length "${MAX_LENGTH}"
  --ruler-lengths ${RULER_LENGTHS}
  --output-dir "${OUTPUT_DIR}"
)

if [[ -n "${LIMIT}" ]]; then
  command+=(--limit "${LIMIT}")
fi

"${command[@]}"

