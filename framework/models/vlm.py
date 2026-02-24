"""
vlm.py — SmolVLM-500M-Instruct wrapper (Vision-Language Model).

  - HuggingFaceTB/SmolVLM-500M-Instruct via HuggingFace transformers
  - Uses AutoModelForImageTextToText (transformers 5.x) with fallback to
    AutoModelForVision2Seq (transformers 4.x)
  - SigLIP backbone: mean=0.5, std=0.5 normalization → maps [0,1] to [-1,1]
  - 512×512 inputs → [B, seq_len, vocab_size] logits
  - A fixed prompt ("Describe what you see.") is pre-tokenized once and
    stored as registered buffers so .to(device) works transparently
  - Frozen (eval mode, no gradients)

pixel_values shape: (B, num_images, 3, H, W)  — SmolVLM convention (one image per sample)
pixel_attention_mask shape: (B, num_images, H, W)  — in PIXEL space, NOT patch space.
  The outer Idefics3 model downsamples to patch space internally (patch_size=16).
  SmolVLM-500M SigLIP: patch_size=16, 512//16=32 → 32×32=1024 patches per image.

If runtime raises a stride error on expand(), call .contiguous() on the buffers.
"""
from __future__ import annotations

from typing import Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
# transformers 5.x renamed AutoModelForVision2Seq → AutoModelForImageTextToText
try:
    from transformers import AutoModelForImageTextToText as _AutoVLMCls
except ImportError:
    from transformers import AutoModelForVision2Seq as _AutoVLMCls  # type: ignore[assignment]
from transformers import AutoProcessor


class SmolVLMWrapper(nn.Module):
    """Frozen SmolVLM-500M-Instruct (HuggingFaceTB/SmolVLM-500M-Instruct)."""

    def __init__(self):
        super().__init__()
        model_id = 'HuggingFaceTB/SmolVLM-500M-Instruct'

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = _AutoVLMCls.from_pretrained(
            model_id,
            torch_dtype=torch.float32,  # bfloat16 for GPU, float32 for CPU compat
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Pre-tokenize a fixed prompt once; store as buffers for device portability
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe what you see."},
        ]}]
        prompt_text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        tok = self.processor.tokenizer(
            prompt_text, return_tensors='pt', add_special_tokens=False
        )
        self.register_buffer('_input_ids',      tok['input_ids'])       # [1, seq_len]
        self.register_buffer('_attention_mask', tok['attention_mask'])  # [1, seq_len]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # SigLIP normalization: [0,1] → [-1,1]
        x_norm = (x - 0.5) / 0.5   # [B, 3, 512, 512]

        # SmolVLM expects pixel_values: (B, num_images, 3, H, W)
        pixel_values = x_norm.unsqueeze(1)   # [B, 1, 3, 512, 512]

        # pixel_attention_mask must be in PIXEL space (B, num_images, H, W).
        # The outer Idefics3/SmolVLM model downsamples to patch space internally
        # using its vision_config.patch_size (=16 for SmolVLM-500M).
        # Passing patch-space dims here causes a double-downsampling shape mismatch.
        pixel_attention_mask = torch.ones(
            B, 1, x.shape[2], x.shape[3], device=x.device, dtype=torch.bool
        )

        outputs = self.model(
            input_ids=self._input_ids.expand(B, -1).contiguous(),
            attention_mask=self._attention_mask.expand(B, -1).contiguous(),
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
        )
        return outputs.logits   # [B, seq_len, vocab_size]

    @staticmethod
    def get_preprocess_fn() -> Callable:
        def preprocess(x: torch.Tensor) -> torch.Tensor:
            # Resize only; normalization is applied inside forward()
            x = F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=False)
            return x.clamp(0.0, 1.0)
        return preprocess

    @staticmethod
    def input_size() -> Tuple[int, int]:
        return (512, 512)
