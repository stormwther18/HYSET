#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"

exec "$PYTHON" "$ROOT/src/train_hyset.py" \
  --corpus "$ROOT/data/hyset_corpus.json" \
  --instruction_dir "$ROOT/data/instruction" \
  --test_id_dir "$ROOT/data/test_query_ids" \
  "$@"
