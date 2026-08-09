#!/usr/bin/env bash
set -euo pipefail

: "${MODEL:?Set MODEL to a Hugging Face model ID or local checkpoint path}"

QUANTIZATION="${QUANTIZATION:-none bnb-int8 bnb-nf4}"
KEEP_RATIOS="${KEEP_RATIOS:-1.0 0.8 0.5}"
TASKS="${TASKS:-copa piqa arc_easy arc_challenge commonsense_qa}"
OUTPUT_DIR="${OUTPUT_DIR:-results}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

command=(
  python scripts/evaluate_quant.py
  --model "${MODEL}"
  --quantization ${QUANTIZATION}
  --keep-ratios ${KEEP_RATIOS}
  --tasks ${TASKS}
  --output-dir "${OUTPUT_DIR}"
)

if [[ -n "${MAX_SAMPLES}" ]]; then
  command+=(--max-samples "${MAX_SAMPLES}")
fi

"${command[@]}"

