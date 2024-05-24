#!/bin/bash
# copy the private key from host
# eval $(ssh-agent)
# ssh-add ~/.ssh/id_rsa
# Modify code to set to private & give access token

python -m open_clip.push_to_hf_hub \
    --model ViT-H-14-quickgelu \
    --pretrained /workspaces/til24-cv-trainer/notebooks/archive/2024_05_23-15_34_38-model_ViT-H-14-quickgelu-lr_7e-07-b_8-j_8-p_amp/checkpoints/epoch_28.pt \
    --repo-id Interpause/ViT-H-14-quickgelu-dfn5b-til24id \
    --precision fp16 \
    --image-interpolation bicubic \
    --image-resize-mode longest \
    --hf-tokenizer-self
