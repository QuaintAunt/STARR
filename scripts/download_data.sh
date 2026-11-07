#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-python3}"

if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [MsMarco|Wikipedia_dpr|text|cc3m|all]" >&2
    exit 2
fi
case "${1:-text}" in
    MsMarco|Wikipedia_dpr|text|cc3m|all) mode="${1:-text}" ;;
    -h|--help) echo "Usage: $0 [MsMarco|Wikipedia_dpr|text|cc3m|all] (default: text)"; exit 0 ;;
    *) echo "Usage: $0 [MsMarco|Wikipedia_dpr|text|cc3m|all]" >&2; exit 2 ;;
esac

if [[ "$mode" == MsMarco || "$mode" == text || "$mode" == all ]]; then
    mkdir -p data/corpora/MsMarco
    (cd data/corpora/MsMarco && "$PYTHON" ../../../dataset/MsMarco/download.py)
fi
if [[ "$mode" == Wikipedia_dpr || "$mode" == text || "$mode" == all ]]; then
    mkdir -p data/corpora/Wikipedia_dpr
    (cd data/corpora/Wikipedia_dpr && "$PYTHON" ../../../dataset/Wikipedia_dpr/download.py)
fi
if [[ "$mode" == cc3m || "$mode" == all ]]; then
    "$PYTHON" dataset/CC3M/download.py
fi
