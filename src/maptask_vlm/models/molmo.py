"""Molmo-7B-D model wrapper.

Molmo uses a different processing API than Qwen (processor.process +
model.generate_from_batch, not a chat template), so this does not
share code with qwen.py -- but exposes the same .answer() interface
so the runner script can treat both models identically.
"""

from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

MOLMO_MODEL_ID = "allenai/Molmo-7B-D-0924"


class MolmoModel:
    """Loads Molmo-7B-D once, then answers (image, prompt) pairs."""

    def __init__(self, model_id: str = MOLMO_MODEL_ID, cache_dir: str | None = None):
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto",
            cache_dir=cache_dir,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto",
            cache_dir=cache_dir,
        )

    def answer(
        self, image_path: str | Path, prompt: str, max_new_tokens: int = 128
    ) -> str:
        """Run one (image, prompt) pair through the model and return the
        generated text response.
        """
        image = Image.open(image_path)

        inputs = self.processor.process(images=[image], text=prompt)
        inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}

        with torch.no_grad():
            output = self.model.generate_from_batch(
                inputs,
                GenerationConfig(
                    max_new_tokens=max_new_tokens, stop_strings="<|endoftext|>"
                ),
                tokenizer=self.processor.tokenizer,
            )

        generated_tokens = output[0, inputs["input_ids"].size(1) :]
        return self.processor.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        )
