#!/bin/bash
cd "$(dirname "$0")"
source ../.venv/bin/activate
export TORCH_CUDNN_V8_API_ENABLED=1
export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync

exec python -m training.main \
    --train-data /workspaces/til24-cv-trainer/data/til24id/train/labels.csv \
    --val-data /workspaces/til24-cv-trainer/data/til24id/val/labels.csv \
    --imagenet-val /workspaces/til24-cv-trainer/data/imagenette2-160/val \
    --workers 16 \
    --batch-size 32 \
    --accum-freq 32 \
    --grad-checkpointing \
    --local-loss \
    --gather-with-grad \
    --epochs 32 \
    --aug-cfg scale='(0.55, 1.0)' color_jitter='(0.2, 0.4, 0.2, 0.0)' color_jitter_prob=0.8 use_extra=True save_samples=0 \
    --lr 1e-5 \
    --wd 0.01 \
    --warmup 100 \
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
    --force-patch-dropout 0.0 \
    --log-every-n-steps 1 \
    --report-to tensorboard \
    --model ViT-H-14-quickgelu \
    --pretrained /workspaces/til24-cv-trainer/notebooks/archive/2024_06_01-03_46_26-model_ViT-H-14-quickgelu-lr_1e-06-b_32-j_16-p_amp/checkpoints/epoch_64.pt \
    --image-interpolation bicubic \
    --image-resize-mode longest \
    --seed 42
    # --torchcompile
