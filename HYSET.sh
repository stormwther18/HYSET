#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"

: "${HYSET_ENCODER_PATH:?Set HYSET_ENCODER_PATH to the frozen BERT or Qwen checkpoint}"
: "${HYSET_REWARD_CACHE:?Set HYSET_REWARD_CACHE to the refreshed execution-reward JSON}"

ENCODER_TYPE="${HYSET_ENCODER_TYPE:-bert}"
if [[ "$ENCODER_TYPE" == "qwen" ]]; then
  D_Z="1536"
else
  D_Z="768"
fi

read -r -a SEEDS <<< "${HYSET_SEEDS:-13 42 87}"
for SEED in "${SEEDS[@]}"; do
  CHECKPOINT_DIR="$ROOT/checkpoints/hyset_${ENCODER_TYPE}_seed${SEED}"
  OUTPUT_DIR="$ROOT/results/hyset_${ENCODER_TYPE}_seed${SEED}"

  "$PYTHON" "$ROOT/src/train_hyset.py" \
    --encoder "$HYSET_ENCODER_PATH" \
    --encoder_type "$ENCODER_TYPE" \
    --d_z "$D_Z" \
    --corpus "$ROOT/data/hyset_corpus.json" \
    --instruction_dir "$ROOT/data/instruction" \
    --test_id_dir "$ROOT/data/test_query_ids" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --reward_cache "$HYSET_REWARD_CACHE" \
    --seed "$SEED"

  "$PYTHON" "$ROOT/src/evaluate_hyset.py" \
    --checkpoint "$CHECKPOINT_DIR/best.pt" \
    --encoder "$HYSET_ENCODER_PATH" \
    --instruction_dir "$ROOT/data/instruction" \
    --test_id_dir "$ROOT/data/test_query_ids" \
    --output_dir "$OUTPUT_DIR"
done

"$PYTHON" "$ROOT/src/summarize_results.py" \
  --inputs "$ROOT"/results/hyset_"${ENCODER_TYPE}"_seed*/summary.json \
  --output "$ROOT/results/hyset_${ENCODER_TYPE}_three_seed_summary.json"
