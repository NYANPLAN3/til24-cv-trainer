import pickle
from pathlib import Path

import torch
from open_clip import create_model_from_pretrained, get_model_config, get_tokenizer
from open_clip.tokenizer import HFTokenizer

MODEL_ARCH = "ViT-H-14-quickgelu"
FT_PATH = "/workspaces/til24-cv-trainer/notebooks/archive/2024_05_23-15_34_38-model_ViT-H-14-quickgelu-lr_7e-07-b_8-j_8-p_amp/checkpoints/epoch_8.pt"
PRECISION = "fp16"  # Experiments with/without AMP show no diff between fp16 and fp32
IMAGE_INTERPOLATION = "bicubic"
IMAGE_RESIZE_MODE = "longest"
OUT_DIR_BASE = "/workspaces/til24-cv-trainer/notebooks/archive/artifacts"
ARTIFACT_NAME = "v1_e8_fp16"


def main():
    model, _ = create_model_from_pretrained(
        MODEL_ARCH,
        pretrained=FT_PATH,
        precision=PRECISION,
        image_interpolation=IMAGE_INTERPOLATION,
        image_resize_mode=IMAGE_RESIZE_MODE,
    )
    model_config = get_model_config(MODEL_ARCH)
    assert model_config

    tokenizer = get_tokenizer(MODEL_ARCH)
    # NOTE: This is most definitely wrong, so client side must always use tokenizer
    # associated with MODEL_ARCH.
    if not isinstance(tokenizer, HFTokenizer):
        # FIXME this makes it awkward to push models with new tokenizers, come up with better soln.
        # default CLIP tokenizers use https://huggingface.co/openai/clip-vit-large-patch14
        tokenizer = HFTokenizer("openai/clip-vit-large-patch14")

    folder = Path(OUT_DIR_BASE).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    tensors = model.state_dict()
    torch.save(
        tensors,
        folder / f"{ARTIFACT_NAME}.bin",
        pickle_protocol=pickle.HIGHEST_PROTOCOL,
    )


if __name__ == "__main__":
    main()
