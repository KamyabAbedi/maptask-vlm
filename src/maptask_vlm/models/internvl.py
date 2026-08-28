"""InternVL2.5-8B model wrapper.

InternVL requires custom image tiling/preprocessing (dividing the image
into up-to-12 448x448 tiles for dynamic resolution) and uses model.chat(),
not a chat-template + generate() call -- another genuinely different API
from Qwen and Molmo, so this stays in its own module.
"""

from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

INTERNVL_MODEL_ID = "OpenGVLab/InternVL2_5-8B"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_transform(input_size: int) -> T.Compose:
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif (
            ratio_diff == best_ratio_diff
            and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]
        ):
            best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(
    image, min_num=1, max_num=12, image_size=448, use_thumbnail=True
):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = sorted(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        },
        key=lambda x: x[0] * x[1],
    )

    target_aspect_ratio = _find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))

    return processed_images


def _load_image(
    image_path: str | Path, input_size: int = 448, max_num: int = 12
) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    transform = _build_transform(input_size)
    tiles = _dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(tile) for tile in tiles]
    return torch.stack(pixel_values)


class InternVLModel:
    """Loads InternVL2.5-8B once, then answers (image, prompt) pairs."""

    def __init__(self, model_id: str = INTERNVL_MODEL_ID, cache_dir: str | None = None):
        self.model = (
            AutoModel.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                cache_dir=cache_dir,
            )
            .eval()
            .cuda()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True, use_fast=False, cache_dir=cache_dir
        )

    def answer(
        self, image_path: str | Path, prompt: str, max_new_tokens: int = 128
    ) -> str:
        """Run one (image, prompt) pair through the model and return the
        generated text response.
        """
        pixel_values = _load_image(image_path).to(torch.bfloat16).cuda()
        generation_config = {"max_new_tokens": max_new_tokens, "do_sample": False}

        question = f"<image>\n{prompt}"
        with torch.no_grad():
            response = self.model.chat(
                tokenizer=self.tokenizer,
                pixel_values=pixel_values,
                question=question,
                generation_config=generation_config,
            )

        return response
