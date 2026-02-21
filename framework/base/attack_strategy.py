"""
AttackStrategy ABC and three concrete strategies.

An AttackStrategy encapsulates *how* a patch is composited onto a clean image:
  - BorderStrategy    : patch surrounds the subject (like OCR license-plate attack)
  - StickerStrategy   : patch pasted at a fixed bbox, subject unmasked
  - PerturbationStrategy: additive L∞/L2-bounded delta, no distinct patch region
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


class AttackStrategy(ABC):
    """
    Abstract base class for patch compositing strategies.

    Every strategy must implement:
      - apply()         : composite patch onto image, return (composited, visibility_mask)
      - apply_neutral() : place a neutral (grey) placeholder instead of adversarial patch
    """

    @abstractmethod
    def apply(
        self,
        image: Tensor,
        patch: Tensor,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        """
        Composite the adversarial patch onto the image.

        Args:
            image: [B, 3, H, W] float in [0, 1]
            patch: [3, pH, pW] float in [0, 1]
            **kwargs: strategy-specific parameters (e.g., bbox, scale)

        Returns:
            composited: [B, 3, out_H, out_W] composited image
            visibility_mask: [1, 1, pH, pW] float mask (1 = visible patch pixel, 0 = hidden)
        """

    @abstractmethod
    def apply_neutral(
        self,
        image: Tensor,
        **kwargs,
    ) -> Tensor:
        """
        Composite a neutral (non-adversarial) placeholder onto the image.

        Used for baseline computation — spatial layout must be identical to apply().

        Args:
            image: [B, 3, H, W] float in [0, 1]
            **kwargs: same keyword arguments as apply()

        Returns:
            neutral_composited: [B, 3, out_H, out_W]
        """


# ---------------------------------------------------------------------------
# Concrete strategy 1: Border (surrounds subject — default for license plates)
# ---------------------------------------------------------------------------

class BorderStrategy(AttackStrategy):
    """
    Patch fills the entire output canvas; the subject is resized and placed
    in the centre, overwriting the adversarial patch in that region.

    Extracted from ProgressivePatchTrainer.apply_patch_ocr_mode() /
    apply_neutral_border_ocr_mode().  Parameterised by center_ratio and
    an explicit output_size so it generalises to any domain.
    """

    def __init__(
        self,
        center_ratio: float = 0.6,
        output_size: Optional[Tuple[int, int]] = None,
        neutral_color: float = 0.5,
    ):
        """
        Args:
            center_ratio: Fraction of output canvas occupied by the subject (default 0.6).
            output_size: (H, W) of the output canvas.  If None, matches patch size.
            neutral_color: Grey value [0,1] used in apply_neutral (default 0.5).
        """
        self.center_ratio = center_ratio
        self.output_size = output_size
        self.neutral_color = neutral_color

    # ------------------------------------------------------------------
    def apply(
        self,
        image: Tensor,
        patch: Tensor,
        center_ratio: Optional[float] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        """
        Composite patch as border around the subject.

        Args:
            image: [B, 3, H, W]
            patch: [3, pH, pW]
            center_ratio: override self.center_ratio if provided

        Returns:
            result: [B, 3, pH, pW]
            visibility_mask: [1, 1, pH, pW]  (1 = border visible, 0 = centre cut-out)
        """
        ratio = center_ratio if center_ratio is not None else self.center_ratio
        device = image.device
        batch_size = image.shape[0]
        patch_h, patch_w = patch.shape[1], patch.shape[2]

        # Patch canvas: tile patch for each batch item
        canvas = patch.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # [B, 3, pH, pW]

        center_h = int(patch_h * ratio)
        center_w = int(patch_w * ratio)
        pad_h = (patch_h - center_h) // 2
        pad_w = (patch_w - center_w) // 2

        # Resize subject into centre slot
        subject = F.interpolate(image, size=(center_h, center_w),
                                mode='bilinear', align_corners=False)

        result = canvas.clone()
        result[:, :, pad_h:pad_h + center_h, pad_w:pad_w + center_w] = subject
        result = torch.clamp(result, 0.0, 1.0)

        # Visibility mask: whole patch visible except centre region (covered by subject)
        mask = torch.ones(1, 1, patch_h, patch_w, device=device)
        mask[:, :, pad_h:pad_h + center_h, pad_w:pad_w + center_w] = 0.0

        return result, mask

    # ------------------------------------------------------------------
    def apply_neutral(
        self,
        image: Tensor,
        center_ratio: Optional[float] = None,
        output_size: Optional[Tuple[int, int]] = None,
        **kwargs,
    ) -> Tensor:
        """
        Place subject in the centre of a neutral grey canvas.

        Args:
            image: [B, 3, H, W]
            center_ratio: override self.center_ratio if provided
            output_size: (H, W) override; if None uses self.output_size or image size

        Returns:
            result: [B, 3, out_H, out_W]
        """
        ratio = center_ratio if center_ratio is not None else self.center_ratio
        device = image.device
        batch_size = image.shape[0]

        out_size = output_size or self.output_size
        if out_size is None:
            out_h, out_w = image.shape[2], image.shape[3]
        else:
            out_h, out_w = out_size

        result = torch.full((batch_size, 3, out_h, out_w), self.neutral_color,
                            dtype=torch.float32, device=device)

        center_h = int(out_h * ratio)
        center_w = int(out_w * ratio)
        pad_h = (out_h - center_h) // 2
        pad_w = (out_w - center_w) // 2

        subject = F.interpolate(image, size=(center_h, center_w),
                                mode='bilinear', align_corners=False)
        result[:, :, pad_h:pad_h + center_h, pad_w:pad_w + center_w] = subject
        result = torch.clamp(result, 0.0, 1.0)

        return result


# ---------------------------------------------------------------------------
# Concrete strategy 2: Sticker (patch pasted at fixed bbox, subject unmasked)
# ---------------------------------------------------------------------------

class StickerStrategy(AttackStrategy):
    """
    Paste the (resized) patch at a fixed bounding box on the image.
    The rest of the image remains unchanged.

    bbox should be (x_min, y_min, x_max, y_max) in pixel coordinates relative
    to the image spatial dimensions.  If not provided at construction time,
    it must be supplied as a kwarg to apply().
    """

    def __init__(
        self,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        neutral_color: float = 0.5,
    ):
        """
        Args:
            bbox: (x_min, y_min, x_max, y_max) default paste region.
            neutral_color: grey value for apply_neutral.
        """
        self.bbox = bbox
        self.neutral_color = neutral_color

    def _resolve_bbox(self, image: Tensor, bbox) -> Tuple[int, int, int, int]:
        if bbox is not None:
            return bbox
        if self.bbox is not None:
            return self.bbox
        # Default: full image
        H, W = image.shape[2], image.shape[3]
        return (0, 0, W, H)

    def apply(
        self,
        image: Tensor,
        patch: Tensor,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        """
        Paste resized patch into bbox region.

        Returns:
            composited: [B, 3, H, W]  (same spatial size as input image)
            visibility_mask: [1, 1, pH, pW]  (all ones — full patch is 'visible')
        """
        x0, y0, x1, y1 = self._resolve_bbox(image, bbox)
        region_h, region_w = y1 - y0, x1 - x0

        patch_resized = F.interpolate(patch.unsqueeze(0), size=(region_h, region_w),
                                      mode='bilinear', align_corners=False).squeeze(0)

        result = image.clone()
        result[:, :, y0:y1, x0:x1] = patch_resized.unsqueeze(0)
        result = torch.clamp(result, 0.0, 1.0)

        # Visibility mask: entire patch is visible
        mask = torch.ones(1, 1, patch.shape[1], patch.shape[2], device=image.device)

        return result, mask

    def apply_neutral(
        self,
        image: Tensor,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        **kwargs,
    ) -> Tensor:
        """Fill bbox with neutral grey, rest of image unchanged."""
        x0, y0, x1, y1 = self._resolve_bbox(image, bbox)
        result = image.clone()
        result[:, :, y0:y1, x0:x1] = self.neutral_color
        return torch.clamp(result, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Concrete strategy 3: Perturbation (additive delta, L∞ budget)
# ---------------------------------------------------------------------------

class PerturbationStrategy(AttackStrategy):
    """
    Additive perturbation strategy: composite = clamp(image + delta, 0, 1).

    The 'patch' tensor is treated as a delta in [−budget, +budget].
    The visibility mask is all-ones (every pixel is perturbed).
    """

    def __init__(self, budget: float = 0.05, norm: str = 'linf'):
        """
        Args:
            budget: Maximum perturbation magnitude (L∞ or L2).
            norm: 'linf' or 'l2'.
        """
        self.budget = budget
        self.norm = norm

    def _clip_patch(self, patch: Tensor) -> Tensor:
        if self.norm == 'linf':
            return torch.clamp(patch, -self.budget, self.budget)
        elif self.norm == 'l2':
            norm_val = patch.reshape(patch.shape[0], -1).norm(dim=1, keepdim=True)
            norm_val = norm_val.view(patch.shape[0], 1, 1, 1)
            scale = torch.clamp(norm_val, min=1.0) / self.budget
            return patch / scale
        else:
            raise ValueError(f"Unknown norm: {self.norm}")

    def apply(
        self,
        image: Tensor,
        patch: Tensor,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        """
        Add clipped delta to every pixel of the image.

        patch is expected to be in [0, 1]; we map it to [-budget, +budget].

        Returns:
            composited: [B, 3, H, W]
            visibility_mask: [1, 1, pH, pW]  (all ones)
        """
        # Map patch from [0,1] to [-budget, +budget]
        delta = (patch - 0.5) * 2.0 * self.budget   # [3, pH, pW]

        # Resize delta to match image if needed
        if delta.shape[1] != image.shape[2] or delta.shape[2] != image.shape[3]:
            delta = F.interpolate(delta.unsqueeze(0), size=(image.shape[2], image.shape[3]),
                                  mode='bilinear', align_corners=False).squeeze(0)

        result = torch.clamp(image + delta.unsqueeze(0), 0.0, 1.0)
        mask = torch.ones(1, 1, patch.shape[1], patch.shape[2], device=image.device)

        return result, mask

    def apply_neutral(
        self,
        image: Tensor,
        **kwargs,
    ) -> Tensor:
        """Return image unchanged (zero perturbation baseline)."""
        return image.clone()
