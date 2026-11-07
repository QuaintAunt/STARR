#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-python3}"

if [[ $# -gt 0 ]]; then
    echo "Usage: $0"
    [[ $# -eq 1 && ( "$1" == -h || "$1" == --help ) ]] && exit 0
    exit 2
fi

mkdir -p data/CC3M
for encoder in vits16 vitb16 resnet50; do
    STARR_MODEL_NAME="dino_${encoder}" \
        STARR_DINO_CKPT_PATH="checkpoints/dino_${encoder}_pretrain.pth" \
        STARR_OUTPUT_FILE="data/CC3M/cc3m_dinov1-${encoder}.npy" \
        STARR_CKPT_FILE="data/CC3M/cc3m_dinov1-${encoder}_ckpt.npy" \
        "$PYTHON" dataset/CC3M/embed_dinov1.py
done
STARR_MODEL_NAME=facebook/dinov2-base \
    STARR_OUTPUT_FILE=data/CC3M/cc3m_dinov2-base.npy \
    STARR_CKPT_FILE=data/CC3M/dinov2-base_ckpt.npy \
    STARR_CKPT_IDX_FILE=data/CC3M/dinov2-base_ckpt_idx.npy \
    "$PYTHON" dataset/CC3M/embed_dinov2.py
STARR_MODEL_NAME=openai/clip-vit-large-patch14 \
    STARR_OUTPUT_FILE=data/CC3M/cc3m_clip_vit_large_patch14.npy \
    STARR_CKPT_FILE=data/CC3M/clip_ckpt.npy \
    "$PYTHON" dataset/CC3M/embed_clip_vit.py
STARR_MODEL_NAME=google/siglip-base-patch16-224 \
    STARR_OUTPUT_FILE=data/CC3M/cc3m_siglip_base_patch16_224.npy \
    STARR_CKPT_FILE=data/CC3M/siglip_ckpt.npy \
    "$PYTHON" dataset/CC3M/embed_siglip.py
for name in dinov1-vits16 dinov1-vitb16 dinov1-resnet50 dinov2-base \
            clip_vit_large_patch14 siglip_base_patch16_224; do
    "$PYTHON" dataset/npy_to_bin.py "data/CC3M/cc3m_${name}.npy" "data/CC3M/cc3m_${name}.bin"
done
