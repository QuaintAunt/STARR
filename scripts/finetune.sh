#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-python3}"

if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [base|large|all]" >&2
    exit 2
fi
case "${1:-base}" in
    base|large) sizes=("${1:-base}") ;;
    all) sizes=(base large) ;;
    -h|--help) echo "Usage: $0 [base|large|all] (default: base)"; exit 0 ;;
    *) echo "Usage: $0 [base|large|all]" >&2; exit 2 ;;
esac

"$PYTHON" dataset/finetune/prepare_data.py
for size in "${sizes[@]}"; do
    STARR_MODEL_NAME="BAAI/bge-${size}-en-v1.5" \
        STARR_TRAIN_OUTPUT="checkpoints/bge-${size}" \
        "$PYTHON" dataset/finetune/train.py
done
