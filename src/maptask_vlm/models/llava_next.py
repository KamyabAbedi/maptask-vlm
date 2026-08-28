"""LLaVA-NeXT (llava-v1.6-mistral-7b) model wrapper.

Natively supported by transformers (no trust_remote_code needed),
using a chat-template API similar to Qwen's -- chosen as the third
model after InternVL2.5-8B's tokenizer loading proved broken with
this environment's pinned transformers/trust_remote_code combination.
"""

from pathlib import Path

import torch
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

LLAVA_NEXT_MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"


class LlavaNextModel:
    """Loads LLaVA-NeXT-Mistral-7B once, then answers (image, prompt) pairs."""

    def __init__(
        self, model_id: str = LLAVA_NEXT_MODEL_ID, cache_dir: str | None = None
    ):
        self.processor = LlavaNextProcessor.from_pretrained(
            model_id, cache_dir=cache_dir
        )
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
            cache_dir=cache_dir,
        )

    def answer(
        self, image_path: str | Path, prompt: str, max_new_tokens: int = 128
    ) -> str:
        """Run one (image, prompt) pair through the model and return the
        generated text response (with the input prompt stripped off).
        """
        from PIL import Image

        image = Image.open(image_path)

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        chat_prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        inputs = self.processor(image, chat_prompt, return_tensors="pt").to(
            self.model.device
        )

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        generated_ids_trimmed = output_ids[0, inputs["input_ids"].shape[1] :]
        return self.processor.decode(generated_ids_trimmed, skip_special_tokens=True)
