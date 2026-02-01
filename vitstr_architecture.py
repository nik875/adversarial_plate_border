"""
Standalone ViTSTR architecture (no doctr dependency at load time).

Extracted from doctr to allow loading pretrained weights without doctr import.
For profiling only - training functionality minimal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Callable


def _bf16_to_float32(x: torch.Tensor) -> torch.Tensor:
    """Convert bfloat16 to float32 if needed."""
    if x.dtype == torch.bfloat16:
        return x.float()
    return x


class _ViTSTRPostProcessor:
    """Base class for ViTSTR postprocessor."""

    def __init__(self, vocab: str):
        self.vocab = vocab
        self._embedding = list(vocab) + ["<eos>"]


class ViTSTRPostProcessor(_ViTSTRPostProcessor):
    """Post processor for ViTSTR architecture"""

    def __call__(self, logits: torch.Tensor) -> list[tuple[str, float]]:
        """Decode logits to text predictions."""
        # compute pred with argmax for attention models
        out_idxs = logits.argmax(-1)
        preds_prob = torch.softmax(logits, -1).max(dim=-1)[0]

        # Manual decoding
        word_values = [
            "".join(self._embedding[idx] for idx in encoded_seq).split("<eos>")[0]
            for encoded_seq in out_idxs.cpu().numpy()
        ]
        # compute probabilities for each word up to the EOS token
        probs = [
            preds_prob[i, : len(word)].clip(0, 1).mean().item() if word else 0.0
            for i, word in enumerate(word_values)
        ]

        return list(zip(word_values, probs))


class _ViTSTR:
    """Base class for ViTSTR with utility methods."""

    vocab: str
    max_length: int

    def build_target(self, gts: list[str]) -> tuple:
        """Encode ground truth labels (for training, not used in profiling)."""
        # Simplified version - full implementation would use doctr.sequences.encode_sequences
        raise NotImplementedError("build_target not needed for profiling")


class ViTSTR(_ViTSTR, nn.Module):
    """ViTSTR architecture for text recognition.

    Implements "Vision Transformer for Fast and Efficient Scene Text Recognition"
    (https://arxiv.org/pdf/2105.08582.pdf)
    """

    def __init__(
        self,
        feature_extractor: nn.Module,
        vocab: str,
        embedding_units: int,
        max_length: int = 32,
        input_shape: tuple = (3, 32, 128),
        exportable: bool = False,
        cfg: dict | None = None,
    ) -> None:
        super().__init__()
        self.vocab = vocab
        self.exportable = exportable
        self.cfg = cfg
        self.max_length = max_length + 2  # +2 for SOS and EOS

        self.feat_extractor = feature_extractor
        self.head = nn.Linear(embedding_units, len(self.vocab) + 1)  # +1 for EOS
        self.postprocessor = ViTSTRPostProcessor(vocab=self.vocab)

    def forward(
        self,
        x: torch.Tensor,
        target: list[str] | None = None,
        return_model_output: bool = False,
        return_preds: bool = False,
    ) -> dict[str, Any]:
        """Forward pass.

        Args:
            x: Input image tensor
            target: Optional ground truth labels (for training)
            return_model_output: Whether to return intermediate features
            return_preds: Whether to return decoded predictions

        Returns:
            Dictionary with logits, predictions, and optional loss
        """
        features = self.feat_extractor(x)["features"]  # (batch, seqlen, d_model)

        if target is not None:
            if self.training:
                _gt, _seq_len = self.build_target(target)
                gt = torch.from_numpy(_gt).to(dtype=torch.long)
                seq_len = torch.tensor(_seq_len)
                gt, seq_len = gt.to(x.device), seq_len.to(x.device)
            else:
                raise ValueError("Target labels only used during training")

        if self.training and target is None:
            raise ValueError("Need to provide labels during training")

        # Trim sequence to max length
        features = features[:, : self.max_length]  # (batch, max_length, d_model)
        B, N, E = features.size()
        features = features.reshape(B * N, E)
        logits = self.head(features).view(B, N, len(self.vocab) + 1)
        decoded_features = _bf16_to_float32(logits[:, 1:])  # remove cls_token

        out: dict[str, Any] = {}

        # For ONNX export or tensor-only output
        if self.exportable:
            out["logits"] = decoded_features
            return out

        if return_model_output:
            out["out_map"] = decoded_features

        if target is None or return_preds:
            out["preds"] = self.postprocessor(decoded_features)

        if target is not None:
            out["loss"] = self.compute_loss(decoded_features, gt, seq_len)

        return out

    @staticmethod
    def compute_loss(
        model_output: torch.Tensor,
        gt: torch.Tensor,
        seq_len: torch.Tensor,
    ) -> torch.Tensor:
        """Compute categorical cross-entropy loss."""
        input_len = model_output.shape[1]
        seq_len = seq_len + 1
        cce = F.cross_entropy(model_output.permute(0, 2, 1), gt[:, 1:], reduction="none")
        mask_2d = torch.arange(input_len, device=model_output.device)[None, :] >= seq_len[:, None]
        cce[mask_2d] = 0
        ce_loss = cce.sum(1) / seq_len.to(dtype=model_output.dtype)
        return ce_loss.mean()
