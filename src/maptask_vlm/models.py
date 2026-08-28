"""Thin wrappers around VLM APIs/SDKs used in the experiments.

Currently supports Qwen2.5-VL-7B-Instruct via Hugging Face transformers.
Requires a GPU; not runnable/testable on a laptop without one.
"""

from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"


class QwenVLModel:
    """Loads Qwen2.5-VL-7B-Instruct once, then answers (image, prompt) pairs."""

    def __init__(self, model_id: str = QWEN_MODEL_ID, cache_dir: str | None = None):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
            cache_dir=cache_dir,
        )
        self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)

    def answer(
        self, image_path: str | Path, prompt: str, max_new_tokens: int = 128
    ) -> str:
        """Run one (image, prompt) pair through the model and return the
        generated text response (with the input prompt stripped off).
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=True)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0]
