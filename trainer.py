"""
trainer.py

AdversarialPatchTrainer — trains an adversarial border patch against a
pluggable detector + OCR backend pair.

Patch parameterization
----------------------
The patch is produced by a trainable ConvTranspose2d decoder that grows a
compact seed tensor from [C, 4, 8] to [3, 256, 512] through six stride-2
layers.  Both the seed and the decoder weights are optimised jointly.
This replaces the direct pixel-level nn.Parameter used in earlier versions.

The inductive bias imposed by the decoder:
  * Early layers capture global structure (the seed operates on a 4×8 grid
    that sees the entire patch simultaneously).
  * Later layers add finer spatial detail at progressively higher resolution.
  * The convolutional weight sharing acts as a natural regulariser, making TV
    loss unnecessary.
  * The parameterisation is strictly more expressive than raw pixels while
    using far fewer "free" degrees of freedom in the seed (~4 K vs 393 K).

Other design decisions
----------------------
* Full-resolution HEIC images loaded from preproc_labels.csv.
* Detector-specific preprocessing runs inside DataLoader workers (parallel).
* Patch applied to preprocessed image; OCR crop from same space.
* Detection loss target: corners_to_bbox (not expanded border).
* Best detection selected by max(IoU × confidence).
* LR schedule: 5-epoch linear warmup (1e-4→5e-4) + CosineAnnealingLR (→1e-4), 100 epochs, no early stopping.
* validate_pipeline() sanity-checks before training.
* save_debug_images() writes 20 annotated images to run_dir/debug/.
"""

from __future__ import annotations

import csv
import random
import re
import time
import warnings
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torchvision.transforms as T
import kornia
import kornia.geometry as K
from tqdm import tqdm

from detector_backends import DetectorBackend, Detection, build_backend
from ocr_backends import OCRBackend, OCRResult, build_ocr_backend, _diff_char_positions
from dataset import (create_dataloaders,
                     make_letterbox_prep, make_resize_prep, make_passthrough_prep)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Differentiable detector preprocessing
# ---------------------------------------------------------------------------

def _diff_letterbox(
    img_chw: torch.Tensor, target_size: int
) -> Tuple[torch.Tensor, float, float, float]:
    """Bilinear resize + grey pad — matches the cv2 letterbox exactly."""
    C, H, W = img_chw.shape
    r      = min(target_size / H, target_size / W)
    new_h  = int(round(H * r))
    new_w  = int(round(W * r))
    img    = F.interpolate(img_chw.unsqueeze(0), size=(new_h, new_w),
                           mode="bilinear", align_corners=False).squeeze(0)
    dw     = (target_size - new_w) / 2
    dh     = (target_size - new_h) / 2
    top    = int(round(dh - 0.1)); bottom = int(round(dh + 0.1))
    left   = int(round(dw - 0.1)); right  = int(round(dw + 0.1))
    img    = F.pad(img, (left, right, top, bottom), value=114.0 / 255.0)
    return img, r, dw, dh


def _diff_resize(
    img_chw: torch.Tensor, target_w: int, target_h: int
) -> Tuple[torch.Tensor, float, float]:
    """Bilinear hard-resize — matches cv2.resize exactly."""
    C, H, W = img_chw.shape
    img     = F.interpolate(img_chw.unsqueeze(0), size=(target_h, target_w),
                            mode="bilinear", align_corners=False).squeeze(0)
    return img, target_w / W, target_h / H


def _corners_letterbox(corners: np.ndarray, r: float, dw: float, dh: float) -> np.ndarray:
    c = corners.astype(np.float32).copy()
    c[:, 0] = c[:, 0] * r + dw
    c[:, 1] = c[:, 1] * r + dh
    return c


def _corners_resize(corners: np.ndarray, sx: float, sy: float) -> np.ndarray:
    c = corners.astype(np.float32).copy()
    c[:, 0] *= sx
    c[:, 1] *= sy
    return c


PATCH_WIDTH  = 512
PATCH_HEIGHT = 256

# Probability that each augmentation transform is applied on a given call.
AUG_PROB = 1 / 3


def augment_plate(image: torch.Tensor, device: str) -> torch.Tensor:
    """
    Differentiable photometric augmentation for the canonical plate+border region.

    Each transform is applied independently with probability AUG_PROB so that
    the number and type of transforms varies per sample.  All ops are pure
    tensor arithmetic — the autograd graph from loss to patch decoder is fully
    preserved.

    Transforms (each applied with probability AUG_PROB):
      1. Brightness       — U(0.5, 1.5) multiplicative scale
      2. Contrast         — U(0.7, 1.3) scale around channel mean
      3. Saturation       — U(0.5, 1.5) via kornia
      4. Color temperature — U(-0.2, 0.2) shift; warm (+) boosts R/reduces B,
                             cool (-) does the opposite
      5. Directional shadow — angle U(0°, 360°), intensity U(0.1, 0.4),
                              linear gradient mask

    Parameters
    ----------
    image  : [1, C, H, W] float32 in [0, 1]
    device : torch device string (must match image.device)
    """
    if random.random() < AUG_PROB:
        factor = random.uniform(0.5, 1.5)
        image  = image * factor

    if random.random() < AUG_PROB:
        factor = random.uniform(0.7, 1.3)
        mean   = image.mean()
        image  = (image - mean) * factor + mean

    if random.random() < AUG_PROB:
        factor = random.uniform(0.5, 1.5)
        image  = kornia.enhance.adjust_saturation(image, factor)

    if random.random() < AUG_PROB:
        shift      = random.uniform(-0.2, 0.2)
        temp_scale = torch.tensor(
            [1.0 + shift * 0.3, 1.0, 1.0 - shift * 0.3],
            dtype=image.dtype, device=device,
        ).view(1, 3, 1, 1)
        image = image * temp_scale

    if random.random() < AUG_PROB:
        angle_deg = random.uniform(0.0, 360.0)
        intensity = random.uniform(0.1, 0.4)
        H, W      = image.shape[-2], image.shape[-1]
        xs        = torch.linspace(0.0, 1.0, W, device=device)
        ys        = torch.linspace(0.0, 1.0, H, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        cos_a     = float(np.cos(np.radians(angle_deg)))
        sin_a     = float(np.sin(np.radians(angle_deg)))
        gradient  = grid_x * cos_a + grid_y * sin_a
        gradient  = (gradient - gradient.min()) / (gradient.max() - gradient.min() + 1e-6)
        shadow    = 1.0 - gradient * intensity
        image     = image * shadow.unsqueeze(0).unsqueeze(0)

    return torch.clamp(image, 0.0, 1.0)


def _bbox_ocr_crop_diff(
    img: torch.Tensor,                      # [1, C, H, W]
    box: torch.Tensor,                      # [x1, y1, x2, y2], may carry autograd grads
    target_size: Tuple[int, Optional[int]], # (h, w) — w=None preserves aspect ratio
) -> torch.Tensor:                          # [1, C, target_h, target_w]
    """
    Differentiable bbox crop via F.grid_sample.
    Gradients flow through both the image pixels and the box coordinates.
    """
    H, W = img.shape[-2], img.shape[-1]
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    th = target_size[0]
    tw = target_size[1]
    if tw is None:
        with torch.no_grad():
            bw = (x2 - x1).clamp(min=1)
            bh = (y2 - y1).clamp(min=1)
            tw = max(1, int((bw / bh * th).item()))
    # Normalise box corners to [-1, 1] for grid_sample
    x1n = x1 / W * 2 - 1
    y1n = y1 / H * 2 - 1
    x2n = x2 / W * 2 - 1
    y2n = y2 / H * 2 - 1
    # Build [1, th, tw, 2] sampling grid; differentiable w.r.t. box
    xs = torch.linspace(0, 1, tw, device=img.device, dtype=img.dtype)
    ys = torch.linspace(0, 1, th, device=img.device, dtype=img.dtype)
    gx = x1n + xs * (x2n - x1n)          # [tw]
    gy = y1n + ys * (y2n - y1n)          # [th]
    grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")  # [th, tw]
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
    return F.grid_sample(img, grid, mode="bilinear", align_corners=True,
                         padding_mode="zeros")


def _bbox_ocr_crop(
    img: torch.Tensor,                      # [1, C, H, W]
    corners: torch.Tensor,                  # [4, 2]  (x, y) in image coords
    target_size: Tuple[int, Optional[int]], # (h, w) — w=None preserves aspect ratio
) -> torch.Tensor:                          # [1, C, target_h, target_w]
    """
    Rectangular bbox crop + bilinear resize for OCR.
    No perspective correction — just clips the axis-aligned bounding box
    of the plate corners out of the image and resizes it.
    If target_size[1] is None, only the height is fixed and width is scaled
    to preserve the crop's natural aspect ratio (avoids character distortion).
    """
    H, W = img.shape[-2], img.shape[-1]
    x1 = int(corners[:, 0].min().clamp(0, W).item())
    y1 = int(corners[:, 1].min().clamp(0, H).item())
    x2 = int(corners[:, 0].max().clamp(0, W).item())
    y2 = int(corners[:, 1].max().clamp(0, H).item())
    crop = img[..., y1:y2, x1:x2]
    th, tw = target_size
    if tw is None:
        crop_h, crop_w = crop.shape[-2], crop.shape[-1]
        tw = max(1, int(crop_w * th / max(crop_h, 1)))
    return F.interpolate(crop, size=(th, tw), mode="bilinear", align_corners=False)




# ---------------------------------------------------------------------------
# Patch decoder
# ---------------------------------------------------------------------------

class PatchDecoder(nn.Module):
    """
    Residual-stream decoder that maps [1, seed_channels, 4, 8] → [1, 3, 256, 512].

    Six stages, each doubling spatial size:
        4×8 → 8×16 → 16×32 → 32×64 → 64×128 → 128×256 → 256×512

    All intermediate feature maps are kept at seed_channels (128) throughout,
    forming a flat residual stream. Each stage contributes two additive deltas:
      1. ConvTranspose2d: nearest-neighbour upsample of stream + deconv delta
      2. Conv2d 7×7:      stream + conv delta (same spatial size)

    This gives every layer — including those close to the seed — a short gradient
    path back to the loss, since gradients flow through the addition operations
    without passing through earlier transposed convs.

    A final 1×1 conv projects the 128-channel stream to 3 channels (RGB).
    Output is passed through tanh and scaled to [0, 1].
    """

    def __init__(self, seed_channels: int = 128):
        super().__init__()
        C = seed_channels
        self.deconvs = nn.ModuleList([
            nn.ConvTranspose2d(C, C, 4, stride=2, padding=1) for _ in range(6)
        ])
        self.convs = nn.ModuleList([
            nn.Conv2d(C, C, 7, padding=3) for _ in range(6)
        ])
        self.final = nn.Conv2d(C, 3, 1)

    def forward(self, seed: torch.Tensor) -> torch.Tensor:
        """seed: [1, C, 4, 8]  →  patch: [1, 3, 256, 512] in [0, 1]"""
        x = seed
        for deconv, conv in zip(self.deconvs, self.convs):
            x = F.interpolate(x, scale_factor=2, mode='nearest') + F.leaky_relu(deconv(x), 0.2)
            x = x + F.leaky_relu(conv(x), 0.2)
        return torch.tanh(self.final(x)) * 0.5 + 0.5


# ---------------------------------------------------------------------------
# m-SAM optimizer wrapper
# ---------------------------------------------------------------------------

class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization wrapper.

    Implements the two-step SAM update (Foret et al. 2021).  Used by
    AdversarialPatchTrainer when sam_m > 0: the ascent step runs on a random
    mini-batch of size sam_m while the descent step uses the full window.
    """

    def __init__(self, params, base_optimizer_cls, rho: float = 0.025, **base_kwargs):
        defaults = dict(rho=rho)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **base_kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        """Perturb weights toward the local sharpness maximum."""
        grad_norm = torch.norm(torch.stack([
            p.grad.norm(p=2)
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]), p=2)
        scale = self.param_groups[0]["rho"] / (grad_norm + 1e-12)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        """Restore original weights, then apply the base optimizer step."""
        for group in self.param_groups:
            for p in group["params"]:
                if "e_w" in self.state[p]:
                    p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    # Required by SequentialLR / other schedulers that call optimizer.step()
    def step(self, closure=None):
        raise RuntimeError("Call first_step / second_step explicitly.")


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class AdversarialPatchTrainer:
    def __init__(
        self,
        detector:             DetectorBackend,
        ocr:                  OCRBackend,
        csv_path:             str            = "preproc_labels.csv",
        seed_channels:        int            = 128,
        preload_images:       bool           = False,
        gpu_preload:          bool           = False,
        num_workers:          int            = 0,
        pin_memory:           bool           = False,
        limit:                int            = 0,
        use_all_for_train:    bool           = False,
        grad_accumulate:      Optional[int]  = None,
        impersonation_target: Optional[str]  = None,
        expected_plate_text:  str            = "VRJ7774",
        print_blur:           float          = 0.0,
        training:             bool           = False,
        train_detector:       bool           = False,
        run_name:             Optional[str]  = None,
        tv_weight:            float          = 10.0,
        ocr_loss_scale:       float          = 1.0,
        det_loss_scale:       float          = 1.0,
        disable_disruption:   bool           = False,
        eval_batch_size:      int            = 1,
        sam_m:                Optional[int]  = None,
        sam_rho:              float          = 0.025,
        skip_sanity:          bool           = False,
        augment:              bool           = False,
        top_extend:           bool           = False,
    ):
        self.training             = training
        self.tv_weight            = tv_weight
        self.ocr_loss_scale       = ocr_loss_scale
        self.det_loss_scale       = det_loss_scale
        self.disable_disruption   = disable_disruption
        self.eval_batch_size      = eval_batch_size
        self.sam_m                = sam_m
        self.sam_rho              = sam_rho
        self.print_blur           = print_blur
        self.augment              = augment
        self.top_extend           = top_extend
        self.grad_accumulate      = grad_accumulate
        self.impersonation_target = impersonation_target
        self.expected_plate_text  = expected_plate_text
        self.train_detector       = train_detector

        # ── Detector ───────────────────────────────────────────────────
        self.detector = detector
        self.device   = detector.device
        if self.train_detector:
            self.detector.train_mode()
        else:
            self.detector.eval()
            self.detector.freeze()
        self.diff_prep    = self._make_differentiable_prep()
        self._cv2_prep    = self._make_cv2_prep()

        # ── OCR ────────────────────────────────────────────────────────
        self.ocr = ocr
        if not self.ocr.is_trainable:
            self.ocr.eval()
            self.ocr.freeze()
        else:
            self.ocr.eval()

        # ── Run output directory ───────────────────────────────────────
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix    = run_name or timestamp
        self.run_dir = Path("runs") / f"{detector.name}_{ocr.name}_{suffix}"
        (self.run_dir / "patches").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "debug").mkdir(parents=True, exist_ok=True)
        print(f"  Run directory : {self.run_dir}")

        # ── Patch decoder (seed + decoder jointly optimised) ───────────
        self.patch_width   = PATCH_WIDTH
        self.patch_height  = PATCH_HEIGHT
        self.seed_channels = seed_channels
        # Small initialisation → decoder output ≈ 0.5 (neutral grey patch)
        self.seed    = nn.Parameter(
            torch.randn(1, seed_channels, 4, 8, device=self.device) * 0.1
        )
        self.decoder = PatchDecoder(seed_channels).to(self.device)

        # ── Image transform ────────────────────────────────────────────
        self.transform = T.Compose([T.ToTensor()])

        # ── Sanity check (before any preloading) ───────────────────────
        if not skip_sanity:
            _sanity_accum = grad_accumulate if grad_accumulate is not None else 4
            _sanity_limit = eval_batch_size * _sanity_accum
            _sanity_loader, _ = create_dataloaders(
                csv_path,
                transform=self.transform,
                preload=False,
                gpu_device=None,
                batch_size=1,
                n_jobs=0,
                pin_memory=False,
                limit=_sanity_limit,
                use_all_for_train=True,
                use_original=True,
            )
            self.validate_pipeline(_sanity_loader)
            del _sanity_loader

        # ── DataLoaders ────────────────────────────────────────────────
        self.train_loader, self.val_loader = create_dataloaders(
            csv_path,
            transform=self.transform,
            preload=preload_images,
            gpu_device=self.device if gpu_preload else None,
            batch_size=1,
            n_jobs=0 if (gpu_preload or preload_images) else num_workers,
            pin_memory=False if (gpu_preload or preload_images) else pin_memory,
            limit=limit,
            use_all_for_train=use_all_for_train,
            use_original=True,
        )

        self.epoch_stats: list = []

    # ====================================================================
    # Patch generation
    # ====================================================================

    def _trainable_params(self):
        """All parameters that the optimiser should update."""
        params = [self.seed] + list(self.decoder.parameters())
        if self.train_detector:
            params.extend(list(self.detector.parameters()))
        return params

    def generate_patch(self, training_aug: bool = False) -> torch.Tensor:
        """
        Run the decoder forward to produce the patch.

        Returns
        -------
        torch.Tensor
            Shape [3, H, W], values in [0, 1], on self.device.
            Gradient graph is intact (suitable for loss.backward()).
        """
        patch = self.decoder(self.seed).squeeze(0)   # [3, 256, 512]

        if self.print_blur > 0:
            patch = kornia.filters.gaussian_blur2d(
                patch.unsqueeze(0), (3, 3),
                (self.print_blur, self.print_blur),
            ).squeeze(0)

        return patch   # [3, 256, 512]

    # ====================================================================
    # Detector preprocessing selection
    # ====================================================================

    def _make_differentiable_prep(self):
        """
        Return a callable  (img_chw: Tensor, corners_np: ndarray)
                        ->  (img_chw_prep: Tensor, new_corners_np: ndarray)
        that applies detector-specific preprocessing differentiably on the GPU.
        """
        name = self.detector.name
        if name in ("yolov8", "yolov11"):
            imgsz = 640
            if hasattr(self.detector, "_yolo") and self.detector._yolo is not None:
                raw = self.detector._yolo.overrides.get("imgsz", 640)
                imgsz = int(raw[0] if hasattr(raw, "__len__") else raw)
            def fn(img_chw, corners_np):
                img, r, dw, dh = _diff_letterbox(img_chw, imgsz)
                return img, _corners_letterbox(corners_np, r, dw, dh)
        elif name == "rtdetr":
            def fn(img_chw, corners_np):
                img, sx, sy = _diff_resize(img_chw, 640, 640)
                return img, _corners_resize(corners_np, sx, sy)
        elif name == "fasterrcnn":
            def fn(img_chw, corners_np):
                return img_chw, corners_np.astype(np.float32).copy()
        elif name == "yolo-v9-384":
            def fn(img_chw, corners_np):
                img, r, dw, dh = _diff_letterbox(img_chw, 384)
                return img, _corners_letterbox(corners_np, r, dw, dh)
        else:
            def fn(img_chw, corners_np):
                img, r, dw, dh = _diff_letterbox(img_chw, 384)
                return img, _corners_letterbox(corners_np, r, dw, dh)
        return fn

    def _make_cv2_prep(self):
        """Return the equivalent cv2-based prep fn (for debug comparison only)."""
        name = self.detector.name
        if name in ("yolov8", "yolov11"):
            imgsz = 640
            if hasattr(self.detector, "_yolo") and self.detector._yolo is not None:
                raw = self.detector._yolo.overrides.get("imgsz", 640)
                imgsz = int(raw[0] if hasattr(raw, "__len__") else raw)
            return make_letterbox_prep(imgsz)
        elif name == "rtdetr":
            return make_resize_prep(640, 640)
        elif name == "fasterrcnn":
            return make_passthrough_prep()
        elif name == "yolo-v9-384":
            return make_letterbox_prep(384)
        else:
            return make_letterbox_prep(384)

    # ====================================================================
    # Geometry helpers
    # ====================================================================

    def bbox_to_corners(self, bbox: torch.Tensor, device=None) -> torch.Tensor:
        x1, y1, x2, y2 = bbox
        return torch.tensor([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]],
                             device=device or self.device)

    def corners_to_bbox(self, corners: torch.Tensor) -> torch.Tensor:
        return torch.stack([corners[:, 0].min(), corners[:, 1].min(),
                            corners[:, 0].max(), corners[:, 1].max()])

    @staticmethod
    def _plate_text_matches(text: str, expected: str) -> bool:
        """
        Fuzzy match for sanity-check categorisation only (not the training objective).
        Strips all non-alphanumeric characters from both strings and compares
        case-insensitively, so "VRJ-7774", "VRJ 7774", and "VRJ7774" all match.
        """
        normalise = lambda s: re.sub(r"[^A-Za-z0-9]", "", s).upper()
        return normalise(text) == normalise(expected)

    @staticmethod
    def _boxes_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        a1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        a2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
        b1, b2 = boxes1.unsqueeze(1), boxes2.unsqueeze(0)
        iw = torch.clamp(torch.min(b1[..., 2], b2[..., 2]) -
                         torch.max(b1[..., 0], b2[..., 0]), min=0)
        ih = torch.clamp(torch.min(b1[..., 3], b2[..., 3]) -
                         torch.max(b1[..., 1], b2[..., 1]), min=0)
        inter = iw * ih
        return inter / (a1 + a2 - inter + 1e-8)

    @staticmethod
    def _rim_bbox(corners: torch.Tensor, border_scale: float = 1.4) -> torch.Tensor:
        """Bounding box of the outer rim (plate corners scaled by border_scale).

        corners : [4, 2] plate corners in detector space.
        Returns [x1, y1, x2, y2].
        """
        ctr = corners.mean(dim=0)
        rim = ctr.unsqueeze(0) + (corners - ctr.unsqueeze(0)) * border_scale
        return torch.stack([rim[:, 0].min(), rim[:, 1].min(),
                            rim[:, 0].max(), rim[:, 1].max()])

    @staticmethod
    def _top_extend_region_corners(corners: torch.Tensor) -> torch.Tensor:
        """[4, 2] perspective-correct corners of the top target region.

        The target is a plate-sized region in the extra attacker-controlled
        block, parameterized by the plate's own column vectors so that the
        target quad matches the perspective tilt of the plate.

        corners[0]=TL, [1]=TR, [2]=BR, [3]=BL.
        col_left  = TL - BL  (upward direction along left edge)
        col_right = TR - BR  (upward direction along right edge)

        Target spans 0.4–1.4 plate-heights above the plate top edge, which
        centers it in the 1.4*ph extra block above the plate.
        Ordering: TL, TR, BR, BL (matches plate corners ordering).
        """
        col_left  = corners[0] - corners[3]   # TL - BL
        col_right = corners[1] - corners[2]   # TR - BR
        return torch.stack([
            corners[0] + 1.4 * col_left,   # TL of target
            corners[1] + 1.4 * col_right,  # TR of target
            corners[1] + 0.4 * col_right,  # BR of target
            corners[0] + 0.4 * col_left,   # BL of target
        ])

    @staticmethod
    def _top_extend_region_bbox(corners: torch.Tensor) -> torch.Tensor:
        """Axis-aligned bbox of the perspective-correct top target region.

        Computes the 4 perspective-correct target corners (via column vectors),
        then returns their axis-aligned bounding box for use in IoU computation
        against axis-aligned detector output boxes.
        """
        quad = AdversarialPatchTrainer._top_extend_region_corners(corners)
        return torch.stack([quad[:, 0].min(), quad[:, 1].min(),
                            quad[:, 0].max(), quad[:, 1].max()])

    # ====================================================================
    # Patch application
    # ====================================================================

    def apply_patch_to_image(
        self,
        image:          torch.Tensor,
        corners:        torch.Tensor,
        patch_norm:     Optional[torch.Tensor] = None,
        border_scale:   float = 1.4,
        augment:        bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Warp the patch onto the image using the plate corners.

        Directly warps the patch to the border quadrilateral (M_border), warps a
        ones mask to the plate region (M_plate), and composites as:
            final_mask = clamp(border_mask - plate_mask, 0, 1)  # ring only
            result     = image * (1 - final_mask) + warped_patch * final_mask
        """
        if patch_norm is None:
            patch_norm = self.generate_patch(training_aug=self.training)

        B    = image.shape[0]
        H, W = image.shape[2], image.shape[3]

        plate  = corners[0]   # [4, 2]: TL, TR, BR, BL
        cx, cy = plate[:, 0].mean(), plate[:, 1].mean()
        center = torch.tensor([cx, cy], device=self.device)

        ph, pw = self.patch_height, self.patch_width
        src    = torch.tensor([[0, 0], [pw, 0], [pw, ph], [0, ph]],
                               dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, 4, 2]

        if self.top_extend:
            # Perspective-correct asymmetric border.
            # Bottom corners: normal 1.4× scale from center.
            # Top corners: normal scale + 1.4 × column vector (TL-BL / TR-BR),
            # giving total column height = 2.8 × plate_h, tilt fully preserved.
            p1_b = center + (plate[0] - center) * border_scale
            p2_b = center + (plate[1] - center) * border_scale
            p3_b = center + (plate[2] - center) * border_scale
            p4_b = center + (plate[3] - center) * border_scale
            col_left  = plate[0] - plate[3]   # TL - BL
            col_right = plate[1] - plate[2]   # TR - BR
            border = torch.stack([p1_b + 1.4 * col_left,
                                   p2_b + 1.4 * col_right,
                                   p3_b, p4_b]).unsqueeze(0)   # [1, 4, 2]
        else:
            border = (center.unsqueeze(0) +
                      (plate - center.unsqueeze(0)) * border_scale).unsqueeze(0)  # [1, 4, 2]

        # Patch space → border quad in image space
        M_border = K.get_perspective_transform(src, border)
        # Patch space → plate quad in image space (for cut-out)
        M_plate  = K.get_perspective_transform(src, plate.unsqueeze(0))

        patch_batch = patch_norm.unsqueeze(0).repeat(B, 1, 1, 1)
        if augment:
            patch_batch = self._augment_image(patch_batch)

        # Brightness correction: scale patch to match the plate region's mean
        # brightness, so it doesn't stand out against different lighting conditions.
        # During training, skipped with probability 0.2 so the model doesn't
        # learn to rely on it as a secondary activation.
        if not (self.training and torch.rand(1, device=self.device).item() < 0.2):
            with torch.no_grad():
                px1 = int(plate[:, 0].min().clamp(0, W - 1).item())
                px2 = int(plate[:, 0].max().clamp(0, W).item())
                py1 = int(plate[:, 1].min().clamp(0, H - 1).item())
                py2 = int(plate[:, 1].max().clamp(0, H).item())
                if px2 > px1 and py2 > py1:
                    plate_brightness = image[0, :, py1:py2, px1:px2].mean().clamp(min=1e-6)
                    patch_brightness = patch_batch.mean().clamp(min=1e-6)
                    brightness_scale = (plate_brightness / patch_brightness).clamp(0.2, 5.0)
                    patch_batch = patch_batch * brightness_scale

        ones = torch.ones(B, 1, ph, pw, device=self.device)

        warped_patch       = K.warp_perspective(patch_batch, M_border, (H, W),
                                                mode="bilinear", padding_mode="zeros",
                                                align_corners=True)
        warped_border_mask = K.warp_perspective(ones, M_border, (H, W),
                                                mode="bilinear", padding_mode="zeros",
                                                align_corners=True)
        warped_plate_mask  = K.warp_perspective(ones, M_plate,  (H, W),
                                                mode="bilinear", padding_mode="zeros",
                                                align_corners=True)

        final_mask = torch.clamp(warped_border_mask - warped_plate_mask, 0, 1).expand(-1, 3, -1, -1)
        result = image * (1 - final_mask) + warped_patch * final_mask
        return torch.clamp(result, 0, 1), final_mask

    # ====================================================================
    # Loss
    # ====================================================================

    def total_variation_loss(self, patch: torch.Tensor) -> torch.Tensor:
        """Isotropic L2 total variation loss on [C, H, W] or [1, C, H, W] patch,
        normalized by number of pixel comparisons."""
        if patch.dim() == 4:
            patch = patch.squeeze(0)
        C, H, W = patch.shape
        tv_h = (patch[:, :, 1:] - patch[:, :, :-1]).pow(2).sum()
        tv_v = (patch[:, 1:, :] - patch[:, :-1, :]).pow(2).sum()
        num_comparisons = C * (H * (W - 1) + (H - 1) * W)
        return (tv_h + tv_v) / num_comparisons

    def _augment_image(self, image: torch.Tensor) -> torch.Tensor:
        return augment_plate(image, self.device)

    def _prepare_one(self, batch_item: dict, patch_norm: torch.Tensor,
                     augment: Optional[bool] = None) -> dict:
        """Fast per-image ops: patch application + preprocessing. No model calls."""
        orig_tensor     = batch_item["orig_image"].to(self.device)      # [C, H, W]
        orig_corners_np = batch_item["orig_corners"].cpu().numpy()
        orig_corners    = batch_item["orig_corners"].to(self.device)    # [4, 2]

        # Halve images that are too large to keep GPU memory manageable
        _, H, W = orig_tensor.shape
        if max(H, W) > 2000:
            orig_tensor     = F.interpolate(orig_tensor.unsqueeze(0), scale_factor=0.5,
                                            mode="bilinear", align_corners=False).squeeze(0)
            orig_corners_np = orig_corners_np * 0.5
            orig_corners    = orig_corners * 0.5

        patched_orig, _ = self.apply_patch_to_image(
            orig_tensor.unsqueeze(0), orig_corners.unsqueeze(0),
            patch_norm=patch_norm,
            augment=self.augment if augment is None else augment)

        patched_prep_chw, new_corners_np = self.diff_prep(
            patched_orig.squeeze(0), orig_corners_np)
        new_corners = torch.from_numpy(new_corners_np).to(self.device)
        target_box  = self.corners_to_bbox(new_corners)
        rim_box     = self._rim_bbox(new_corners)
        ocr_crop    = _bbox_ocr_crop(patched_orig, orig_corners, self.ocr.ocr_crop_size)

        if self.top_extend:
            top_region_box = self._top_extend_region_bbox(new_corners)
            # OCR crop for the top target region, using perspective-correct corners
            # in original full-res image coords (same column-vector parameterization).
            top_corners_orig = self._top_extend_region_corners(orig_corners)
            top_ocr_crop = _bbox_ocr_crop(patched_orig, top_corners_orig,
                                          self.ocr.ocr_crop_size)
        else:
            top_region_box = None
            top_ocr_crop   = None

        return {
            "patched_prep":  patched_prep_chw,   # [C, H_p, W_p]
            "target_box":    target_box,          # [4]
            "rim_box":       rim_box,             # [4]
            "ocr_crop":      ocr_crop,            # [1, 3, H_c, W_c]
            "top_region_box": top_region_box,     # [4] or None
            "top_ocr_crop":   top_ocr_crop,       # [1, 3, H_c, W_c] or None
        }

    def compute_loss_batch(self, items: list) -> tuple:
        """Batch the slow model-eval calls; average losses over B items.

        Dual-flow top-extend impersonation:
          Flow 1 (real plate, suppress): minimize IoU when detected, else
            minimize conf score — gradient always flows.
          Flow 2 (top region, attract): maximize IoU when detected, else
            maximize conf score — gradient always flows.

        Mutual-exclusion in the no-overlap case:
          When exactly one target has an overlapping predicted box, the other
          target's proximity-weighted confidence (used as the no-overlap
          gradient signal) excludes the boxes that overlap the detected target.
          This prevents the already-claimed box from dominating the proximity
          weighting for the non-detected target.

        OCR: differentiable crop from predicted box when available; GT crop
          fallback only when --no-disruption; otherwise that flow is skipped.
        """
        patch_norm = items[0]["_patch_norm"]
        preps = [x["patched_prep"] for x in items]
        if all(p.shape == preps[0].shape for p in preps):
            batched_prep = torch.stack(preps)
        else:
            max_h = max(p.shape[1] for p in preps)
            max_w = max(p.shape[2] for p in preps)
            batched_prep = torch.stack([
                F.pad(p, (0, max_w - p.shape[2], 0, max_h - p.shape[1]))
                for p in preps
            ])
        target_boxes     = [x["target_box"]     for x in items]
        top_region_boxes = [x["top_region_box"] for x in items]
        tv_l             = self.total_variation_loss(patch_norm)

        # Positional OCR weights: diff chars get 4× emphasis over same chars.
        if self.expected_plate_text:
            diff_pos = _diff_char_positions(self.impersonation_target,
                                            self.expected_plate_text)
            n        = max(len(self.impersonation_target), len(self.expected_plate_text))
            same_pos = [i for i in range(n) if i not in diff_pos]
        else:
            diff_pos, same_pos = None, None

        two_target_results = self.detector.differentiable_predict_box_batch_two_targets(
            batched_prep, target_boxes, top_region_boxes)

        image_losses, det_l_list, ocr_l_list = [], [], []
        for i, ((conf_real, pred_box), (conf_top, top_pred_box)) in enumerate(two_target_results):
            # ── Detection ─────────────────────────────────────────────────
            # Real plate (suppress): if detected use IoU, else use conf.
            # Both are in [0, det_loss_scale]; gradient flows either way.
            if pred_box is not None:
                iou_real = self._boxes_iou(
                    pred_box.unsqueeze(0),
                    items[i]["target_box"].to(self.device).unsqueeze(0)).squeeze()
                det_real = iou_real * self.det_loss_scale
            else:
                det_real = conf_real * self.det_loss_scale

            # Top region (attract): negate so minimizing loss maximizes detection.
            if top_pred_box is not None:
                iou_top = self._boxes_iou(
                    top_pred_box.unsqueeze(0),
                    items[i]["top_region_box"].to(self.device).unsqueeze(0)).squeeze()
                det_top = -iou_top * self.det_loss_scale
            else:
                det_top = -conf_top * self.det_loss_scale

            det_i = det_real + det_top

            # ── OCR ───────────────────────────────────────────────────────
            # Use differentiable crop from predicted box when available.
            # With --no-disruption, fall back to GT crop so OCR loss still flows
            # when the detector hasn't fired yet. Otherwise skip that flow.
            if pred_box is not None:
                real_crop = _bbox_ocr_crop_diff(
                    items[i]["patched_prep"].unsqueeze(0),
                    pred_box.to(self.device), self.ocr.ocr_crop_size)
            elif self.disable_disruption:
                real_crop = items[i]["ocr_crop"]
            else:
                real_crop = None

            if top_pred_box is not None:
                top_crop = _bbox_ocr_crop_diff(
                    items[i]["patched_prep"].unsqueeze(0),
                    top_pred_box.to(self.device), self.ocr.ocr_crop_size)
            elif self.disable_disruption:
                top_crop = items[i]["top_ocr_crop"]
            else:
                top_crop = None

            ocr_parts = []
            for crop in [c for c in [real_crop, top_crop] if c is not None]:
                if diff_pos is not None:
                    loss_d = self.ocr.differentiable_loss_batch(
                        [crop], self.impersonation_target,
                        impersonation=True, diff_positions=diff_pos)[0]
                    loss_s = self.ocr.differentiable_loss_batch(
                        [crop], self.expected_plate_text,
                        impersonation=True, diff_positions=same_pos)[0]
                    ocr_parts.append((0.8 * loss_d + 0.2 * loss_s) * self.ocr_loss_scale)
                else:
                    ocr_parts.append(self.ocr.differentiable_loss_batch(
                        [crop], self.impersonation_target,
                        impersonation=True)[0] * self.ocr_loss_scale)

            ocr_i = (sum(ocr_parts) / len(ocr_parts)) if ocr_parts \
                else torch.tensor(0.0, device=self.device)

            image_losses.append((det_i + ocr_i) / 2)
            det_l_list.append(det_i.detach())
            ocr_l_list.append(ocr_i.detach())

        total = torch.stack(image_losses).mean() + self.tv_weight * tv_l
        det_l = torch.stack(det_l_list).mean()
        ocr_l = torch.stack(ocr_l_list).mean()

        return total, det_l.detach(), ocr_l.detach(), (self.tv_weight * tv_l).detach()

    def compute_loss(self, batch: dict) -> tuple:
        """Thin wrapper: B=1 case, for backward compatibility."""
        patch_norm = self.generate_patch(training_aug=self.training)
        item = self._prepare_one(
            {k: v[0] for k, v in batch.items()}, patch_norm)
        item["_patch_norm"] = patch_norm
        return self.compute_loss_batch([item])

    # ====================================================================
    # Patch persistence
    # ====================================================================

    def save_patch(self, epoch: int, subdir: str = "patches",
                   stem: Optional[str] = None) -> None:
        save_dir = self.run_dir / subdir
        save_dir.mkdir(parents=True, exist_ok=True)
        if stem is None:
            stem = f"patch_{self.detector.name}_epoch_{epoch:04d}"
        with torch.no_grad():
            patch_img = self.generate_patch()   # [3, H, W] in [0,1]
            T.ToPILImage()(patch_img.cpu()).save(str(save_dir / f"{stem}.png"))
            torch.save({
                "seed":          self.seed.detach().cpu(),
                "decoder":       self.decoder.state_dict(),
                "seed_channels": self.seed_channels,
                "epoch":         epoch,
                "backend":       self.detector.name,
                "ocr":           self.ocr.name,
                "patch_size":    (self.patch_height, self.patch_width),
                "patch":         patch_img.cpu(),   # rendered tensor (incl. print_blur)
            }, str(save_dir / f"{stem}.pt"))

    # ====================================================================
    # Pre-training sanity check
    # ====================================================================

    def validate_pipeline(self, loader=None) -> None:
        """
        Run one full gradient-accumulation cycle (B * update_every items) through
        the training forward+backward path to verify the pipeline end-to-end.
        Raises on any crash.
        """
        print("\n── Pre-training sanity check ──────────────────────────────")
        _loader = loader if loader is not None else self.train_loader
        B = self.eval_batch_size
        update_every = (self.grad_accumulate
                        if self.grad_accumulate is not None
                        else min(4, len(_loader)))
        need = B * update_every

        items_raw = []
        for batch in _loader:
            items_raw.append({k: v[0] for k, v in batch.items()})
            if len(items_raw) >= need:
                break
        if not items_raw:
            raise RuntimeError("No training data found for sanity check.")

        # Models must be in training mode for cuDNN RNN backward to work
        self.detector.train_mode()
        self.ocr.train()
        try:
            patch_with_graph = self.generate_patch(training_aug=self.training)
            patch_leaf = patch_with_graph.detach().requires_grad_(True)
            buffer: list = []
            step = 0
            total_loss = 0.0
            for raw_item in items_raw:
                item = self._prepare_one(raw_item, patch_leaf)
                item["_patch_norm"] = patch_leaf
                buffer.append(item)
                if len(buffer) < B:
                    continue
                loss, det_l, ocr_l, tv_l = self.compute_loss_batch(buffer)
                buffer = []
                step += 1
                scaled = loss / update_every
                scaled.backward()  # frees item graph; accumulates into patch_leaf.grad
                total_loss += loss.item()
                del loss, scaled
            # Propagate through generator so trainable params get gradients
            if patch_leaf.grad is not None:
                patch_with_graph.backward(patch_leaf.grad)

            # Zero grads so the optimizer starts clean
            for p in self._trainable_params():
                if p.grad is not None:
                    p.grad.zero_()
        finally:
            self.detector.eval()
            self.detector.freeze()
            self.ocr.eval()
            self.ocr.freeze()

        avg = total_loss / max(step, 1)
        msg = f"Sanity check passed: {step} accumulation step(s), avg loss {avg:.4f}"
        print(f"  {msg}\n")
        (self.run_dir / "sanity_check.txt").write_text(msg + "\n")

    # ====================================================================
    # Debug images
    # ====================================================================

    @staticmethod
    def _shrink_for_save(img_hwc: np.ndarray, max_dim: int = 1280) -> np.ndarray:
        """Downscale an HWC uint8 image so its longest side ≤ max_dim."""
        h, w = img_hwc.shape[:2]
        scale = min(max_dim / max(h, w), 1.0)
        if scale < 1.0:
            img_hwc = cv2.resize(img_hwc, (int(w * scale), int(h * scale)),
                                 interpolation=cv2.INTER_AREA)
        return img_hwc

    def save_debug_images(self, n: int = 20) -> None:
        from torch.utils.data import Subset, DataLoader as DL

        debug_dir = self.run_dir / "debug"

        # Pick n random indices across the combined train+val dataset,
        # then build a tiny DataLoader that only loads those images.
        train_ds  = self.train_loader.dataset
        val_ds    = self.val_loader.dataset
        n_train   = len(train_ds)
        total     = n_train + len(val_ds)
        np.random.seed(42)
        chosen    = np.random.choice(total, min(n, total), replace=False)

        train_idx = [int(i)           for i in chosen if i <  n_train]
        val_idx   = [int(i) - n_train for i in chosen if i >= n_train]

        subsets = []
        if train_idx:
            subsets.append(Subset(train_ds, train_idx))
        if val_idx:
            subsets.append(Subset(val_ds, val_idx))

        sample_loader = DL(
            torch.utils.data.ConcatDataset(subsets),
            batch_size=1, shuffle=False,
            num_workers=self.train_loader.num_workers,
        )
        sample_items = list(sample_loader)  # only n images loaded

        print(f"  Saving {len(sample_items)} debug images → {debug_dir}")
        summary_rows = []

        for img_idx, batch in enumerate(sample_items):
            fn              = (batch["filename"][0]
                               if isinstance(batch["filename"], (list, tuple))
                               else batch["filename"])
            orig_tensor     = batch["orig_image"][0].to(self.device)
            orig_corners_np = batch["orig_corners"][0].cpu().numpy()
            orig_corners    = batch["orig_corners"][0].to(self.device)

            with torch.no_grad():
                prep_tensor, new_corners_np = self.diff_prep(orig_tensor, orig_corners_np)
            new_corners = torch.from_numpy(new_corners_np).to(self.device)
            new_c       = new_corners_np
            target_box  = self.corners_to_bbox(new_corners)

            with torch.no_grad():
                detections = self.detector.predict(prep_tensor)
                crop       = _bbox_ocr_crop(orig_tensor.unsqueeze(0), orig_corners,
                                            self.ocr.ocr_crop_size)
                ocr_result = self.ocr.predict(crop.squeeze(0))

            text       = ocr_result.text or ""
            conf       = ocr_result.confidence
            is_correct = self._plate_text_matches(text, self.expected_plate_text)

            vis = (prep_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8).copy()

            # (a) preprocessed image + detected bbox (red) + OCR label
            if detections:
                best_det = max(
                    detections,
                    key=lambda d: (
                        self._boxes_iou(d.box.to(self.device).unsqueeze(0),
                                        target_box.unsqueeze(0)).item()
                        * d.confidence
                    ),
                )
                x1, y1, x2, y2 = best_det.box.int().tolist()
                cv2.rectangle(vis, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
            cv2.putText(vis, f"{text} ({conf:.2f})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imwrite(str(debug_dir / f"{img_idx:02d}_a_preprocessed_detection.png"),
                        self._shrink_for_save(vis))

            # (b) raw OCR crop
            crop_np = (crop.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            cv2.imwrite(str(debug_dir / f"{img_idx:02d}_b_ocr_crop.png"), crop_np)

            # (c) random patch applied with geometry annotations:
            #   green  = plate quad (suppress target)
            #   yellow = border quad (full patch footprint)
            #   red    = top target quad (attract objective, top-extend only)
            # Pipeline matches _prepare_one exactly: patch applied to original
            # full-res image first, then preprocessed — not patch-on-preprocessed.
            with torch.no_grad():
                rand_seed    = torch.randn_like(self.seed)
                rand_patch   = self.decoder(rand_seed).squeeze(0)   # [3, H, W]
                patched_orig, _ = self.apply_patch_to_image(
                    orig_tensor.unsqueeze(0), orig_corners.unsqueeze(0),
                    patch_norm=rand_patch,
                )
                patched_prep, _ = self.diff_prep(patched_orig.squeeze(0), orig_corners_np)
            patch_vis = (patched_prep.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8).copy()

            def _draw_quad(img, corners_t, color):
                pts = corners_t.detach().cpu().numpy().astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)

            # Plate quad (green) — the cut-out / suppress target
            _draw_quad(patch_vis, new_corners, (0, 255, 0))
            # Border quad (yellow) — actual patch footprint, perspective-correct
            _cx = new_corners[:, 0].mean(); _cy = new_corners[:, 1].mean()
            _ctr = torch.tensor([_cx, _cy], device=self.device)
            if self.top_extend:
                _p1b = _ctr + (new_corners[0] - _ctr) * 1.4
                _p2b = _ctr + (new_corners[1] - _ctr) * 1.4
                _p3b = _ctr + (new_corners[2] - _ctr) * 1.4
                _p4b = _ctr + (new_corners[3] - _ctr) * 1.4
                _cl = new_corners[0] - new_corners[3]
                _cr = new_corners[1] - new_corners[2]
                _border_corners = torch.stack([_p1b + 1.4 * _cl, _p2b + 1.4 * _cr, _p3b, _p4b])
            else:
                _border_corners = _ctr.unsqueeze(0) + (new_corners - _ctr.unsqueeze(0)) * 1.4
            _draw_quad(patch_vis, _border_corners, (0, 255, 255))
            # Top target quad (red) — attract objective region, perspective-correct
            if self.top_extend:
                _draw_quad(patch_vis, self._top_extend_region_corners(new_corners), (0, 0, 255))
            cv2.imwrite(str(debug_dir / f"{img_idx:02d}_c_random_patch.png"),
                        self._shrink_for_save(patch_vis))

            # (d) cv2 preprocessing vs differentiable preprocessing comparison
            with torch.no_grad():
                img_hwc_uint8 = (orig_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                cv2_prep_t, _ = self._cv2_prep(img_hwc_uint8, orig_corners_np)
                diff_prep_np  = (prep_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                cv2_prep_np   = (cv2_prep_t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                comparison = np.concatenate([cv2_prep_np, diff_prep_np], axis=1)
                cv2.imwrite(str(debug_dir / f"{img_idx:02d}_d_prep_comparison.png"),
                            self._shrink_for_save(comparison))

            row = {
                "index": img_idx, "filename": fn,
                "detected_text": text, "confidence": f"{conf:.4f}",
                "correct": is_correct,
            }

            if hasattr(self.ocr, "_sequence_log_prob"):
                target_text = self.expected_plate_text
                with torch.no_grad():
                    pixel_values = (crop - 0.5) / 0.5
                    log_prob     = self.ocr._sequence_log_prob(pixel_values, target_text)
                row["log_prob"] = f"{log_prob.item():.4f}"

            summary_rows.append(row)

        has_log_prob = hasattr(self.ocr, "_sequence_log_prob")
        fieldnames   = ["index", "filename", "detected_text", "confidence", "correct"]
        if has_log_prob:
            fieldnames.append("log_prob")

        csv_path = debug_dir / "debug_summary.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"  Debug summary → {csv_path}")

    # ====================================================================
    # Training loop
    # ====================================================================

    def _msam_step(
        self,
        optimizer: SAM,
        window_raw: list,
        B: int,
        update_every: int,
    ) -> Tuple[float, float, float, float]:
        """One m-SAM update: ascent on sam_m random items, descent on all items.

        Returns the sum-of-chunk-averages for (loss, det, ocr, tv) consistent
        with the per-step accounting in train_epoch.

        Memory design: we avoid retain_graph by detaching the patch from the
        generator and treating it as a leaf for the per-item backward calls.
        Each item's graph is freed immediately after its backward.  Once all
        item gradients are accumulated in the detached leaf's .grad, a single
        backward through the generator (using that accumulated grad as the
        incoming gradient) propagates correctly to seed/decoder params.
        """
        M = len(window_raw)
        m = min(self.sam_m, M)

        def _accumulate(indices, weight) -> Tuple[float, float, float, float]:
            """
            Generate patch, detach → leaf, accumulate item gradients one-by-one
            (no retain_graph), then backprop once through the generator.
            Returns (total_loss, det_sum, ocr_sum, tv_sum) scaled by weight.
            """
            patch_with_graph = self.generate_patch(training_aug=self.training)
            patch_leaf = patch_with_graph.detach().requires_grad_(True)

            total_loss = det_sum = ocr_sum = tv_sum = 0.0
            for i in indices:
                item = self._prepare_one(window_raw[i], patch_leaf)
                item["_patch_norm"] = patch_leaf
                loss, det_l, ocr_l, tv_l = self.compute_loss_batch([item])
                (loss * weight).backward()   # frees item graph; accumulates patch_leaf.grad
                total_loss += loss.item()
                det_sum    += det_l.item()
                ocr_sum    += ocr_l.item()
                tv_sum     += tv_l.item()
                del item

            # Propagate accumulated patch gradient through the generator graph.
            patch_with_graph.backward(patch_leaf.grad)
            return total_loss, det_sum, ocr_sum, tv_sum

        # ── ASCENT: gradients from m random items ────────────────────────
        optimizer.zero_grad()
        ascent_idx = sorted(random.sample(range(M), m))
        _accumulate(ascent_idx, 1.0 / m)

        torch.nn.utils.clip_grad_norm_(self._trainable_params(), max_norm=1.0)
        optimizer.first_step(zero_grad=True)

        # ── DESCENT: gradients from all M items, perturbed patch ─────────
        total_loss, det_sum, ocr_sum, tv_sum = _accumulate(range(M), 1.0 / M)

        torch.nn.utils.clip_grad_norm_(self._trainable_params(), max_norm=1.0)
        optimizer.second_step(zero_grad=True)

        return total_loss, det_sum, ocr_sum, tv_sum

    def train_epoch(self, optimizer, epoch: int) -> Tuple[float, float, float, float]:
        if self.ocr.is_trainable:
            self.ocr.train()

        B = self.eval_batch_size
        update_every = (len(self.train_loader)
                        if self.grad_accumulate is None
                        else self.grad_accumulate)
        total_loss = accum_loss = 0.0
        total_det  = total_ocr  = total_tv = 0.0
        step = num_updates = 0
        buffer: list = []
        use_sam = isinstance(optimizer, SAM)
        window_raw: list = []   # SAM: raw batch items for the current update window

        # Generate the first patch (with generator graph) for the accumulation window.
        # We use a detach-accumulate-backprop pattern: detach the patch into a
        # leaf tensor so each per-item backward() frees its graph immediately
        # (no retain_graph needed).  Accumulated gradients on the leaf are then
        # back-propagated through the generator in a single call.
        patch_with_graph = self.generate_patch(training_aug=self.training)
        patch_leaf = patch_with_graph.detach().requires_grad_(True)

        with tqdm(enumerate(self.train_loader),
                  desc=f"Epoch {epoch+1}",
                  total=len(self.train_loader), leave=False) as pbar:
            for idx, batch in pbar:
                raw_item = {k: v[0] for k, v in batch.items()}

                if use_sam:
                    window_raw.append(raw_item)
                    window_full = len(window_raw) == B * update_every
                    if not window_full:
                        continue

                    loss_t, det_t, ocr_t, tv_t = self._msam_step(
                        optimizer, window_raw, B, update_every)
                    window_raw = []
                    step       += update_every
                    num_updates += 1
                    total_loss += loss_t
                    total_det  += det_t
                    total_ocr  += ocr_t
                    total_tv   += tv_t
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                    elif self.device == "mps":
                        torch.mps.empty_cache()
                    pbar.set_postfix({
                        "loss": f"{total_loss/step:.4f}",
                        "det":  f"{total_det/step:.4f}",
                        "ocr":  f"{total_ocr/step:.4f}",
                        "tv":   f"{total_tv/step:.4f}",
                    })
                    patch_with_graph = self.generate_patch(training_aug=self.training)
                    patch_leaf = patch_with_graph.detach().requires_grad_(True)
                    continue

                # ── Standard (non-SAM) accumulation path ─────────────────
                # Use the detached patch_leaf so each backward frees its
                # detector/OCR graph immediately (no retain_graph).
                item = self._prepare_one(raw_item, patch_leaf)
                item["_patch_norm"] = patch_leaf
                buffer.append(item)

                if len(buffer) < B:
                    continue

                loss, det_l, ocr_l, tv_l = self.compute_loss_batch(buffer)
                buffer = []
                scaled_loss = loss / update_every
                step += 1
                scaled_loss.backward()  # frees item graph; accumulates into patch_leaf.grad

                accum_loss += loss.item()
                total_det  += det_l.item()
                total_ocr  += ocr_l.item()
                total_tv   += tv_l.item()
                del loss, scaled_loss

                if step % update_every == 0:
                    # Propagate accumulated patch gradient through generator.
                    patch_with_graph.backward(patch_leaf.grad)
                    torch.nn.utils.clip_grad_norm_(self._trainable_params(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    total_loss  += accum_loss
                    num_updates += 1
                    accum_loss   = 0.0
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                    elif self.device == "mps":
                        torch.mps.empty_cache()
                    pbar.set_postfix({
                        "loss": f"{total_loss/step:.4f}",
                        "det":  f"{total_det/step:.4f}",
                        "ocr":  f"{total_ocr/step:.4f}",
                        "tv":   f"{total_tv/step:.4f}",
                    })
                    # New patch for next accumulation window
                    patch_with_graph = self.generate_patch(training_aug=self.training)
                    patch_leaf = patch_with_graph.detach().requires_grad_(True)

            # ── Flush remainder at end of epoch ──────────────────────────
            if use_sam and len(window_raw) >= B:
                # Partial window: trim to a multiple of B, then do m-SAM update
                n_complete = (len(window_raw) // B) * B
                loss_t, det_t, ocr_t, tv_t = self._msam_step(
                    optimizer, window_raw[:n_complete], B, n_complete // B)
                step       += n_complete // B
                num_updates += 1
                total_loss += loss_t
                total_det  += det_t
                total_ocr  += ocr_t
                total_tv   += tv_t
            elif not use_sam:
                # Flush remainder buffer (< B images left at end of epoch)
                if buffer:
                    loss, det_l, ocr_l, tv_l = self.compute_loss_batch(buffer)
                    buffer = []
                    scaled_loss = loss / update_every
                    scaled_loss.backward()  # accumulates into patch_leaf.grad
                    accum_loss += loss.item()
                    total_det  += det_l.item()
                    total_ocr  += ocr_l.item()
                    total_tv   += tv_l.item()
                    step       += 1
                    del loss, scaled_loss

                if step % update_every != 0 and self.grad_accumulate is not None:
                    # Propagate remainder gradients through generator.
                    patch_with_graph.backward(patch_leaf.grad)
                    torch.nn.utils.clip_grad_norm_(self._trainable_params(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    total_loss  += accum_loss
                    num_updates += 1

        n = max(step, 1)
        return (total_loss / n, total_det / n, total_ocr / n, total_tv / n)

    def validate(self) -> float:
        if self.ocr.is_trainable:
            self.ocr.eval()
        B = self.eval_batch_size
        losses = []
        buffer: list = []
        with torch.no_grad():
            for batch in self.val_loader:
                patch_norm = self.generate_patch(training_aug=False)
                item = self._prepare_one(
                    {k: v[0] for k, v in batch.items()}, patch_norm, augment=False)
                item["_patch_norm"] = patch_norm
                buffer.append(item)

                if len(buffer) < B:
                    continue

                _, det_l, ocr_l, _ = self.compute_loss_batch(buffer)
                val_loss = ocr_l if self.disable_disruption else (det_l + ocr_l) / 2
                losses.append(val_loss.item())
                buffer = []

            if buffer:
                _, det_l, ocr_l, _ = self.compute_loss_batch(buffer)
                val_loss = ocr_l if self.disable_disruption else (det_l + ocr_l) / 2
                losses.append(val_loss.item())

        return float(np.mean(losses)) if losses else 0.0

    def train(
        self,
        num_epochs:    int   = 100,
        learning_rate: float = 5e-4,
        lr_min:        float = 1e-5,
        save_interval: int   = 10,
        dry_run:       bool  = False,
    ) -> dict:
        if dry_run:
            print("\nDry run: saving debug images...")
            self.save_debug_images()
            print("Dry run complete.")
            return {}

        warmup_epochs = 10
        eta_min       = lr_min

        # Optimizer starts at learning_rate; warmup scales from eta_min up to it
        if self.sam_m is not None:
            optimizer = SAM(
                self._trainable_params(),
                base_optimizer_cls=optim.AdamW,
                rho=self.sam_rho,
                lr=learning_rate,
                weight_decay=1e-4,
            )
            # Schedulers operate on the base optimizer's param groups (shared via SAM)
            sched_optimizer = optimizer.base_optimizer
        else:
            optimizer = optim.AdamW(
                self._trainable_params(), lr=learning_rate, weight_decay=1e-4
            )
            sched_optimizer = optimizer
        cosine_epochs = num_epochs - warmup_epochs
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            sched_optimizer,
            start_factor=eta_min / learning_rate,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            sched_optimizer, T_max=cosine_epochs, eta_min=eta_min,
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            sched_optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )

        history    = {"loss": [], "val_score": [], "learning_rate": []}
        best_loss  = float("inf")
        best_epoch = -1

        n_params = sum(p.numel() for p in self._trainable_params())
        print(f"\n{'='*60}")
        print(f"  Adversarial Patch Training")
        print(f"  Detector  : {self.detector.name}   OCR: {self.ocr.name}")
        print(f"  Patch gen : ConvTranspose decoder  "
              f"(seed {self.seed_channels}ch×4×8 → 3×256×512)")
        print(f"  Trainable : {n_params:,} params  "
              f"(seed {self.seed.numel():,}  +  decoder {n_params-self.seed.numel():,})")
        print(f"  Dataset   : {len(self.train_loader)+len(self.val_loader)} images")
        print(f"  Epochs    : {num_epochs}  |  Warmup: {warmup_epochs}  |  "
              f"LR: {eta_min:.0e} → {learning_rate:.0e} → {eta_min:.0e}")
        if self.sam_m is not None:
            print(f"  Optimizer : m-SAM  (m={self.sam_m}, rho={self.sam_rho}, base=AdamW)")
        _mode = ('impersonation → ' + self.impersonation_target) if self.impersonation_target else 'disruption'
        if self.disable_disruption:
            _mode += '  [detection loss disabled]'
        print(f"  Mode      : {_mode}")
        print(f"  Run dir   : {self.run_dir}")
        print(f"{'='*60}\n")

        log_path = self.run_dir / "training_log.txt"
        log_file = open(log_path, "w")

        for epoch in range(num_epochs):
            epoch_start    = time.time()
            self.training  = True
            # Disable TV loss during warmup so the patch can move freely early on
            saved_tv = self.tv_weight
            if epoch < warmup_epochs:
                self.tv_weight = 0.0
            train_loss, train_det, train_ocr, train_tv = self.train_epoch(optimizer, epoch)
            self.tv_weight = saved_tv
            self.training  = False
            val_loss       = self.validate()
            scheduler.step()
            lr = optimizer.param_groups[0]["lr"]
            epoch_time     = time.time() - epoch_start

            history["loss"].append(train_loss)
            history["val_score"].append(val_loss)
            history["learning_rate"].append(lr)

            init_val    = history["val_score"][0]
            change      = (val_loss / (init_val + 1e-9) - 1) * 100
            best_marker = ""

            if val_loss < best_loss:
                best_loss  = val_loss
                best_epoch = epoch
                self.save_patch(epoch, "patches", stem=f"patch_{self.detector.name}_best")
                best_marker = "  ★ best"

            line = (f"Epoch {epoch+1:3d}/{num_epochs} "
                    f"[{self.detector.name}/{self.ocr.name}] | "
                    f"loss: {train_loss:.4f}  det: {train_det:.4f}  "
                    f"ocr: {train_ocr:.4f}  tv: {train_tv:.4f} | "
                    f"val: {val_loss:.4f} Δ{change:+.1f}% | "
                    f"lr: {lr:.2e} | "
                    f"time: {epoch_time:.1f}s{best_marker}")
            print(line)
            log_file.write(line + "\n")
            log_file.flush()

            if (epoch + 1) % save_interval == 0:
                self.save_patch(epoch, "patches")

        log_file.close()

        import pandas as pd
        hist_path = self.run_dir / "training_history.csv"
        pd.DataFrame(history).assign(
            epoch=range(1, len(history["loss"]) + 1)
        ).to_csv(str(hist_path), index=False)

        print(f"\nDone. Best val loss {best_loss:.4f} at epoch {best_epoch+1}.")
        print(f"Training history → {hist_path}")
        return history


# ====================================================================
# CLI
# ====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Adversarial patch trainer — ConvTranspose decoder + pluggable backends"
    )
    parser.add_argument("--csv", default="preproc_labels.csv")

    TRAINABLE_DET = ["sam", "yolov8", "fasterrcnn", "yolov11", "rtdetr", "yolo-v9-384"]
    TRAINABLE_OCR = ["crnn", "trocr", "dtrb", "lprnet", "cct", "fastanpr-ocr", "doctr-vitstr"]

    parser.add_argument("--backend",      default="yolov8",  choices=TRAINABLE_DET)
    parser.add_argument("--model-path",   default="license_plate_detector.pt")
    parser.add_argument("--ocr-backend",  default="crnn",    choices=TRAINABLE_OCR)
    parser.add_argument("--ocr-model-path", default="none")
    parser.add_argument("--ocr-repo-root",  default=None)
    parser.add_argument("--dtrb-feature-extraction", default="vitstr_small_patch16_224")
    parser.add_argument("--dtrb-sequence-modeling",  default="None")
    parser.add_argument("--dtrb-transformation",     default="None")
    parser.add_argument("--seed-channels", type=int, default=128,
                        help="Number of channels in the decoder seed (default 128 → 4096 seed params).")
    parser.add_argument("--device",   default="cuda")
    parser.add_argument("--epochs",   type=int,   default=100)
    parser.add_argument("--lr",       type=float, default=5e-4)
    parser.add_argument("--lr-min",   type=float, default=1e-5,
                        help="Minimum LR for cosine annealing (and warmup start). Default: 1e-5.")
    parser.add_argument("--grad-accumulate", type=int, default=64)
    parser.add_argument("--preload-images",  action="store_true")
    parser.add_argument("--gpu-preload",     action="store_true",
                        help="Preload entire dataset as GPU tensors (implies --preload-images, forces num-workers=0)")
    parser.add_argument("--num-workers",     type=int, default=0)
    parser.add_argument("--pin-memory",      action="store_true")
    parser.add_argument("--limit",           type=int, default=0)
    parser.add_argument("--use-all-for-train", action="store_true")
    parser.add_argument("--impersonation-target", default="SHX8459")
    parser.add_argument("--expected-plate", default="VRJ7774")
    parser.add_argument("--train-detector", action="store_true",
                        help="Allow detector backend weights to update during training.")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--dry-run",    action="store_true")
    parser.add_argument("--skip-sanity", action="store_true",
                        help="Skip pre-training sanity check.")
    parser.add_argument("--tv-weight", type=float, default=10.0,
                        help="Weight for total variation loss (default: 2.5).")
    parser.add_argument("--ocr-loss-scale", type=float, default=1.0,
                        help="Scalar multiplier on OCR loss (default: 1.0).")
    parser.add_argument("--det-loss-scale", type=float, default=1.0,
                        help="Scalar multiplier on detection loss (default: 1.0).")
    parser.add_argument("--no-disruption", action="store_true",
                        help="Disable the detection (disruption) loss component entirely. "
                             "Detection is still computed for pipeline purposes but contributes "
                             "zero gradient to the total loss.")
    parser.add_argument("--eval-batch-size", type=int, default=1,
                        help="Number of images to batch for detector/OCR evaluation (default 1).")
    parser.add_argument("--sam-m", type=int, default=None,
                        help="Enable m-SAM: number of images for the ascent step. "
                             "Recommended: ~25%% of --grad-accumulate (e.g. 8 for accum=32). "
                             "Disabled by default.")
    parser.add_argument("--sam-rho", type=float, default=0.025,
                        help="SAM perturbation radius rho (default 0.025). "
                             "Only used when --sam-m is set.")
    parser.add_argument("--top-extend", action="store_true",
                        help="Double patch height upward: suppress real-plate detection "
                             "and attract detection into the attacker-controlled top region.")
    parser.add_argument("--augment", action="store_true",
                        help="Apply differentiable photometric augmentations (brightness, "
                             "contrast, saturation, color temperature, directional shadow) "
                             "after patch application at each training step.")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile the detector and OCR models (PyTorch 2.0+, "
                             "gradients still flow through compiled models).")
    args = parser.parse_args()

    backend = build_backend(args.backend, args.model_path, device=args.device)
    backend.load()

    ocr_kwargs = {}
    if args.ocr_backend == "dtrb":
        if args.ocr_repo_root:
            ocr_kwargs["dtrb_root"] = args.ocr_repo_root
        ocr_kwargs["feature_extraction"] = args.dtrb_feature_extraction
        ocr_kwargs["sequence_modeling"]  = args.dtrb_sequence_modeling
        ocr_kwargs["transformation"]     = args.dtrb_transformation
    ocr = build_ocr_backend(args.ocr_backend, args.ocr_model_path,
                             device=args.device, **ocr_kwargs)
    ocr.load()

    if args.compile:
        if hasattr(backend, "_model") and backend._model is not None:
            print(f"[compile] Compiling detector ({backend.name})...")
            backend._model = torch.compile(backend._model)
        if hasattr(ocr, "_model") and ocr._model is not None:
            print(f"[compile] Compiling OCR ({ocr.name})...")
            ocr._model = torch.compile(ocr._model)

    trainer = AdversarialPatchTrainer(
        detector             = backend,
        ocr                  = ocr,
        csv_path             = args.csv,
        seed_channels        = args.seed_channels,
        preload_images       = args.preload_images,
        gpu_preload          = args.gpu_preload,
        num_workers          = args.num_workers,
        pin_memory           = args.pin_memory,
        limit                = args.limit,
        use_all_for_train    = args.use_all_for_train,
        grad_accumulate      = args.grad_accumulate,
        impersonation_target = args.impersonation_target,
        expected_plate_text  = args.expected_plate,
        training             = True,
        train_detector       = args.train_detector,
        run_name             = args.run_name,
        tv_weight            = args.tv_weight,
        ocr_loss_scale       = args.ocr_loss_scale,
        det_loss_scale       = args.det_loss_scale,
        disable_disruption   = args.no_disruption,
        eval_batch_size      = args.eval_batch_size,
        sam_m                = args.sam_m,
        sam_rho              = args.sam_rho,
        skip_sanity          = args.skip_sanity,
        augment              = args.augment,
        top_extend           = args.top_extend,
    )

    trainer.train(
        num_epochs    = args.epochs,
        learning_rate = args.lr,
        lr_min        = args.lr_min,
        save_interval = 10,
        dry_run       = args.dry_run,
    )


if __name__ == "__main__":
    main()
