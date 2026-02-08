"""VLM-based OCR using Qwen2.5-VL for text extraction + figure description."""

from __future__ import annotations

import base64
import os
from pathlib import Path

from openai import OpenAI

from src.ocr.base import BaseOCR


class VLMExtractor(BaseOCR):
    """Uses a Vision-Language Model (GPT-4o or similar) for OCR.

    This approach handles:
    - Text extraction
    - Layout detection
    - Figure/diagram description

    All in a single pass.
    """
    name = "vlm"

    def __init__(
        self,
        model_name: str = "gpt-4o",  
        **kwargs,
    ):
        self.model_name = model_name
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def extract_page(self, image_path: str | Path) -> str:
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        suffix = Path(image_path).suffix.lower()
        mime_type = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
        }.get(suffix, "image/png")

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an OCR assistant. Extract ALL text from this scanned book page. "
                        "Preserve paragraph structure. If there are figures or diagrams, "
                        "describe them in [FIGURE: description] format. "
                        "Output only the extracted text, no commentary."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_data}",
                            },
                        },
                        {
                            "type": "text",
                            "text": "Extract all text from this page.",
                        },
                    ],
                },
            ],
            max_tokens=4096,
            temperature=0.0,
        )

        return response.choices[0].message.content.strip()


class Qwen2VLExtractor(BaseOCR):
    """Uses local Qwen2.5-VL-7B model for OCR.

    Requires: pip install transformers accelerate qwen-vl-utils
    And sufficient GPU memory (~16GB for 7B model).
    """
    name = "qwen2vl"

    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct", **kwargs):
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        import torch

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_name)

    def extract_page(self, image_path: str | Path) -> str:
        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {
                        "type": "text",
                        "text": (
                            "Extract ALL text from this scanned book page. "
                            "Preserve paragraph structure. If there are figures or diagrams, "
                            "describe them in [FIGURE: description] format. "
                            "Output only the extracted text."
                        ),
                    },
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

        generated_ids = self.model.generate(**inputs, max_new_tokens=4096)
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0].strip()
