"""
vlm.py — SmolVLM-500M-Instruct wrapper (Vision-Language Model).

  - HuggingFaceTB/SmolVLM-500M-Instruct via HuggingFace transformers
  - SigLIP backbone: mean=0.5, std=0.5 normalization → maps [0,1] to [-1,1]
  - 512×512 inputs → [B, seq_len, vocab_size] logits
  - A fixed prompt ("Describe what you see.") is pre-tokenized once and
    stored as registered buffers so .to(device) works transparently
  - Frozen (eval mode, no gradients)

pixel_values shape: (B, num_images, 3, H, W)  — SmolVLM convention (one image per sample)
pixel_attention_mask shape: (B, num_images, H//14, W//14)  — SigLIP patch_size=14

If runtime raises a stride error on expand(), call .contiguous() on the buffers.
If pixel_attention_mask is rejected, fall back to passing None (model handles it).
"""
from __future__ import annotations

from typing import Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModelForVision2Seq


class SmolVLMWrapper(nn.Module):
    """Frozen SmolVLM-500M-Instruct (HuggingFaceTB/SmolVLM-500M-Instruct)."""

    def __init__(self):
        super().__init__()
        model_id = 'HuggingFaceTB/SmolVLM-500M-Instruct'

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForVision2Seq.from_pretrained(
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

        # pixel_attention_mask: (B, num_images, H//patch_size, W//patch_size)
        # SigLIP patch_size = 14; 512 // 14 = 36
        h_p = x.shape[2] // 14
        w_p = x.shape[3] // 14
        pixel_attention_mask = torch.ones(
            B, 1, h_p, w_p, device=x.device, dtype=torch.bool
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
