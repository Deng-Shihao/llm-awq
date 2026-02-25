#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/qwen25_1p5b_tinychat.sh
# Optional env overrides:
#   MODEL_PATH, W_BIT, Q_GROUP_SIZE, AWQ_CACHE_DIR, QUANT_CACHE_DIR

MODEL_PATH="${MODEL_PATH:-$HOME/models/Qwen2.5-1.5B}"
MODEL_NAME="qwen2.5-1.5b"
W_BIT="${W_BIT:-4}"
Q_GROUP_SIZE="${Q_GROUP_SIZE:-128}"
AWQ_CACHE_DIR="${AWQ_CACHE_DIR:-awq_cache}"
QUANT_CACHE_DIR="${QUANT_CACHE_DIR:-quant_cache}"

AWQ_FILE="${AWQ_CACHE_DIR}/${MODEL_NAME}-w${W_BIT}-g${Q_GROUP_SIZE}.pt"
QUANT_FILE_BASE="${QUANT_CACHE_DIR}/${MODEL_NAME}-w${W_BIT}-g${Q_GROUP_SIZE}-awq.pt"
QUANT_FILE_V2="${QUANT_CACHE_DIR}/${MODEL_NAME}-w${W_BIT}-g${Q_GROUP_SIZE}-awq-v2.pt"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Model path not found: ${MODEL_PATH}" >&2
  exit 1
fi

mkdir -p "${AWQ_CACHE_DIR}" "${QUANT_CACHE_DIR}"

echo "[1/3] Running AWQ search -> ${AWQ_FILE}"
python -m awq.entry \
  --model_path "${MODEL_PATH}" \
  --w_bit "${W_BIT}" --q_group_size "${Q_GROUP_SIZE}" \
  --run_awq --dump_awq "${AWQ_FILE}"

echo "[2/3] Building real INT4 weights -> ${QUANT_FILE_BASE}"
python -m awq.entry \
  --model_path "${MODEL_PATH}" \
  --w_bit "${W_BIT}" --q_group_size "${Q_GROUP_SIZE}" \
  --load_awq "${AWQ_FILE}" \
  --q_backend real --dump_quant "${QUANT_FILE_BASE}"

if [[ ! -f "${QUANT_FILE_V2}" ]]; then
  echo "Expected quantized file not found: ${QUANT_FILE_V2}" >&2
  echo "Note: awq.entry auto-renames output to *-v2.pt" >&2
  exit 1
fi

echo "[3/3] Launching TinyChat for Qwen"
python -m tinychat.demo \
  --model_type qwen \
  --model_path "${MODEL_PATH}" \
  --q_group_size "${Q_GROUP_SIZE}" \
  --load_quant "${QUANT_FILE_V2}" \
  --precision W4A16 \
  --flash_attn --chunk_prefilling
