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
# --save-most-recent \
exec python -m training.main \
    --val-data /workspaces/til24-cv-trainer/data/til24id/val/labels.csv \
    --imagenet-val /workspaces/til24-cv-trainer/data/imagenette2-160/val \
    --workers 8 \
    --batch-size 8 \
    --accum-freq 128 \
    --epochs 32 \
    --lr 0 \
    --wd 100 \
    --warmup 100000 \
    --zeroshot-frequency 1 \
    --save-frequency -1 \
    --precision amp \
    --force-image-size 224 \
    --lock-image \
    --lock-image-freeze-bn-stats \
    --lock-text \
    --lock-text-freeze-layer-norm \
    --force-patch-dropout 0 \
    --log-every-n-steps 1 \
    --model ViT-H-14-quickgelu \
    --pretrained /workspaces/til24-cv-trainer/notebooks/archive/artifacts/dfn5b.bin \
    --image-interpolation bicubic \
    --image-resize-mode longest \
    --seed 42
    # --torchcompile
