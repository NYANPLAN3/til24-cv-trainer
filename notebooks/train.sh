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
# --model ViT-H-14-quickgelu \
# --pretrained dfn5b \
# --lr 7e-7 \
# --pretrained /workspaces/til24-cv-trainer/notebooks/archive/2024_05_23-15_34_38-model_ViT-H-14-quickgelu-lr_7e-07-b_8-j_8-p_amp/checkpoints/epoch_32.pt \
# --wd 0.005 \
exec python -m training.main \
    --train-data /workspaces/til24-cv-trainer/data/til24id/train/labels.csv \
    --val-data /workspaces/til24-cv-trainer/data/til24id/val/labels.csv \
    --imagenet-val /workspaces/til24-cv-trainer/data/imagenette2-160/val \
    --workers 16 \
    --batch-size 8 \
    --accum-freq 128 \
    --epochs 32 \
    --aug-cfg scale='(1.0, 1.0)' color_jitter='(0.2, 0.4, 0.2, 0.0)' color_jitter_prob=0.8 use_extra=True \
    --lr 1e-6 \
    --wd 0.002 \
    --warmup 150 \
    --zeroshot-frequency 1 \
    --save-frequency 2 \
    --precision amp \
    --force-image-size 224 \
    --lock-image \
    --lock-image-unlocked-groups 999 \
    --lock-image-freeze-bn-stats \
    --lock-text \
    --lock-text-unlocked-layers 999 \
    --lock-text-freeze-layer-norm \
    --force-patch-dropout 0.1 \
    --log-every-n-steps 1 \
    --report-to tensorboard \
    --model ViT-H-14-quickgelu \
    --pretrained dfn5b \
    --image-interpolation bicubic \
    --image-resize-mode longest \
    --seed 42
    # --torchcompile
