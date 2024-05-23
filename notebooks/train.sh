#!/bin/bash
cd "$(dirname "$0")"
source ../.venv/bin/activate
export TORCH_CUDNN_V8_API_ENABLED=1
export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync

# ViT-SO400M-14-SigLIP  webli
# ViT-H-14-quickgelu    dfn5b
# --lock-image \
# --lock-image-unlocked-groups 2 \
# --lock-image-freeze-bn-stats \
# --lock-text \
# --lock-text-unlocked-layers 2 \
# --lock-text-freeze-layer-norm \
# --force-patch-dropout 0.0 \
python -m training.main \
    --train-data /workspaces/til24-cv-trainer/data/til24id/train/labels.csv \
    --val-data /workspaces/til24-cv-trainer/data/til24id/val/labels.csv \
    --workers 8 \
    --batch-size 8 \
    --accum-freq 64 \
    --epochs 32 \
    --aug-cfg scale='(0.9, 1.1)' color_jitter='(0.2, 0.4, 0.2, 0.0)' color_jitter_prob=0.8 use_extra=True \
    --lr 1e-6 \
    --wd 0.0005 \
    --warmup 1000 \
    --save-frequency -1 \
    --save-most-recent \
    --precision amp \
    --force-image-size 224 \
    --lock-image \
    --lock-image-unlocked-groups 999 \
    --lock-image-freeze-bn-stats \
    --lock-text \
    --lock-text-unlocked-layers 999 \
    --lock-text-freeze-layer-norm \
    --model ViT-H-14-quickgelu \
    --pretrained dfn5b \
    --image-interpolation bicubic \
    --image-resize-mode longest \
    --seed 42
    # --torchcompile
