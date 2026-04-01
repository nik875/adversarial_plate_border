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
import traceback
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
from ocr_backends import OCRBackend, OCRResult, build_ocr_backend
from dataset import (create_dataloaders, create_ccpd_dataloaders,
                     make_letterbox_prep, make_resize_prep, make_passthrough_prep,
                     _chw_uint8)

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


def _diff_letterbox_batch(
    imgs_bchw: torch.Tensor, target_size: int
) -> Tuple[torch.Tensor, float, float, float]:
    """Batched bilinear resize + grey pad — same logic as _diff_letterbox."""
    B, C, H, W = imgs_bchw.shape
    r      = min(target_size / H, target_size / W)
    new_h  = int(round(H * r))
    new_w  = int(round(W * r))
    imgs   = F.interpolate(imgs_bchw, size=(new_h, new_w),
                           mode="bilinear", align_corners=False)
    dw     = (target_size - new_w) / 2
    dh     = (target_size - new_h) / 2
    top    = int(round(dh - 0.1)); bottom = int(round(dh + 0.1))
    left   = int(round(dw - 0.1)); right  = int(round(dw + 0.1))
    imgs   = F.pad(imgs, (left, right, top, bottom), value=114.0 / 255.0)
    return imgs, r, dw, dh


def _diff_resize_batch(
    imgs_bchw: torch.Tensor, target_w: int, target_h: int
) -> Tuple[torch.Tensor, float, float]:
    """Batched bilinear hard-resize — same logic as _diff_resize."""
    B, C, H, W = imgs_bchw.shape
    imgs = F.interpolate(imgs_bchw, size=(target_h, target_w),
                         mode="bilinear", align_corners=False)
    return imgs, target_w / W, target_h / H


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
    th, tw = target_size
    if x2 <= x1 or y2 <= y1:
        return torch.zeros(img.shape[0], img.shape[1], th, tw or 128, device=img.device)
    crop = img[..., y1:y2, x1:x2]
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
        num_workers:          int            = -1,
        pin_memory:           bool           = True,
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
        det_loss_weight:      float          = 0.0,
        disable_disruption:   bool           = False,
        eval_batch_size:      int            = 1,
        sam_m:                Optional[int]  = None,
        sam_m_auto:           bool           = False,
        sam_rho:              float          = 0.025,
        skip_sanity:          bool           = False,
        augment:              bool           = False,
        top_extend:           bool           = False,
        ccpd_csv:             Optional[str]  = None,
        continue_run_dir:     Optional[str]  = None,
    ):
        self.training             = training
        self.tv_weight            = tv_weight
        self.det_loss_weight      = det_loss_weight
        self.disable_disruption   = disable_disruption
        self.eval_batch_size      = eval_batch_size
        self.sam_m                = sam_m
        self._sam_m_auto          = sam_m_auto and (sam_m is None)
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
        if continue_run_dir:
            self.run_dir = Path(continue_run_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            suffix    = run_name or timestamp
            self.run_dir = Path("runs") / f"{detector.name}_{ocr.name}_{suffix}"
        self.profiling: bool = False
        self._prof:     dict = {}
        (self.run_dir / "patches").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "debug").mkdir(parents=True, exist_ok=True)
        print(f"  Run directory : {self.run_dir}")

        # ── Patch decoder (seed + decoder jointly optimised) ───────────
        self.patch_width   = PATCH_WIDTH
        self.patch_height  = PATCH_HEIGHT if not top_extend else PATCH_HEIGHT * 2
        self.seed_channels = seed_channels
        # Small initialisation → decoder output ≈ 0.5 (neutral grey patch)
        seed_h = 8 if top_extend else 4
        self.seed    = nn.Parameter(
            torch.randn(1, seed_channels, seed_h, 8, device=self.device) * 0.1
        )
        self.decoder = PatchDecoder(seed_channels).to(self.device)

        # ── Image transform ────────────────────────────────────────────
        self.transform = _chw_uint8

        # ── Sanity check (before any preloading) ───────────────────────
        if not skip_sanity:
            _sanity_accum = grad_accumulate if grad_accumulate is not None else 4
            _sanity_limit = eval_batch_size * _sanity_accum
            if ccpd_csv:
                _sanity_loader, _ = create_ccpd_dataloaders(
                    ccpd_csv, batch_size=1, n_jobs=0, pin_memory=False,
                    limit=_sanity_limit,
                )
            else:
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
        _n_jobs = os.cpu_count() if num_workers < 0 else num_workers
        if ccpd_csv:
            self.train_loader, self.val_loader = create_ccpd_dataloaders(
                ccpd_csv, batch_size=1,
                n_jobs=_n_jobs, pin_memory=pin_memory,
                limit=limit,
            )
        else:
            self.train_loader, self.val_loader = create_dataloaders(
                csv_path,
                transform=self.transform,
                preload=preload_images,
                gpu_device=self.device if gpu_preload else None,
                batch_size=1,
                n_jobs=0 if (gpu_preload or preload_images) else _n_jobs,
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
        elif name == "yolo-v9-608":
            def fn(img_chw, corners_np):
                img, r, dw, dh = _diff_letterbox(img_chw, 608)
                return img, _corners_letterbox(corners_np, r, dw, dh)
        else:
            def fn(img_chw, corners_np):
                img, r, dw, dh = _diff_letterbox(img_chw, 384)
                return img, _corners_letterbox(corners_np, r, dw, dh)
        return fn

    def _diff_prep_batch(
        self, imgs_bchw: torch.Tensor, corners_batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batched differentiable preprocessing.

        imgs_bchw:     [B, C, H, W]
        corners_batch: [B, 4, 2]
        Returns:       (imgs_prep [B, C, H', W'], new_corners [B, 4, 2])

        All images in the batch must have the same spatial size (caller's
        responsibility — guaranteed by the by_size grouping in _prepare_batch).
        """
        name = self.detector.name
        if name in ("yolov8", "yolov11"):
            imgsz = 640
            if hasattr(self.detector, "_yolo") and self.detector._yolo is not None:
                raw = self.detector._yolo.overrides.get("imgsz", 640)
                imgsz = int(raw[0] if hasattr(raw, "__len__") else raw)
            imgs, r, dw, dh = _diff_letterbox_batch(imgs_bchw, imgsz)
            c = corners_batch.clone()
            c[..., 0] = c[..., 0] * r + dw; c[..., 1] = c[..., 1] * r + dh
        elif name == "rtdetr":
            imgs, sx, sy = _diff_resize_batch(imgs_bchw, 640, 640)
            c = corners_batch.clone()
            c[..., 0] = c[..., 0] * sx; c[..., 1] = c[..., 1] * sy
        elif name == "fasterrcnn":
            imgs = imgs_bchw
            c = corners_batch.clone()
        elif name == "yolo-v9-608":
            imgs, r, dw, dh = _diff_letterbox_batch(imgs_bchw, 608)
            c = corners_batch.clone()
            c[..., 0] = c[..., 0] * r + dw; c[..., 1] = c[..., 1] * r + dh
        else:
            imgs, r, dw, dh = _diff_letterbox_batch(imgs_bchw, 384)
            c = corners_batch.clone()
            c[..., 0] = c[..., 0] * r + dw; c[..., 1] = c[..., 1] * r + dh
        return imgs, c

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
        elif name == "yolo-v9-608":
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

    def _prepare_batch(self, raw_items: list, patch_norm: torch.Tensor,
                       augment: Optional[bool] = None) -> list:
        """Batch patch application + preprocessing for a list of raw items.

        Groups images by spatial size so apply_patch_to_image can run as a
        single batched warp call per size group (kornia requires uniform HxW).
        diff_prep runs as a single batched F.interpolate call per size group.
        """
        _prof = self.profiling
        if _prof: _t_batch0 = _t0 = _pt()

        # ── CPU-side downscale + size grouping ──────────────────────────
        # Downscale on CPU before transfer so we only move the smaller tensor.
        loaded_cpu = []
        for raw in raw_items:
            t = raw["orig_image"]       # [C, H, W] — stays on CPU
            c = raw["orig_corners"]     # [4, 2]
            _, H, W = t.shape
            if max(H, W) > 2000:
                # F.interpolate requires float; if uint8 normalize now so the
                # GPU transfer path doesn't see a float image in [0,255].
                t_f = t.float().div_(255.0) if t.dtype == torch.uint8 else t
                t = F.interpolate(t_f.unsqueeze(0), scale_factor=0.5,
                                  mode="bilinear", align_corners=False).squeeze(0)
                c = c * 0.5
            loaded_cpu.append((t, c, raw.get("label")))

        aug = self.augment if augment is None else augment

        # Group indices by image spatial size for batched warp.
        by_size: Dict[Tuple[int, int], List[int]] = {}
        for i, (t, _, _) in enumerate(loaded_cpu):
            key = (int(t.shape[-2]), int(t.shape[-1]))
            by_size.setdefault(key, []).append(i)

        # ── Async GPU transfer + float cast ─────────────────────────────
        # Images are uint8 from the DataLoader (4× smaller over PCIe than
        # float32). Transfer async on pinned memory, then cast to float32 on
        # the GPU immediately — all enqueued on the same CUDA stream so the
        # cast runs only after each DMA transfer completes.
        loaded = []
        for t, c, label in loaded_cpu:
            t_gpu = t.to(self.device, non_blocking=True)
            if t_gpu.dtype == torch.uint8:
                t_gpu = t_gpu.float().div_(255.0)
            loaded.append((t_gpu, c.to(self.device, non_blocking=True), label))

        if _prof:
            self._prof.setdefault("prepare/to_gpu", []).append(_pt() - _t0)
            _t1 = _pt()

        # ── Batched patch application ────────────────────────────────────
        patched_orig = [None] * len(loaded)
        for indices in by_size.values():
            imgs    = torch.stack([loaded[i][0] for i in indices])       # [G, C, H, W]
            corners = torch.stack([loaded[i][1] for i in indices])       # [G, 4, 2]
            patched, _ = self.apply_patch_to_image(
                imgs, corners, patch_norm=patch_norm, augment=aug)
            for j, i in enumerate(indices):
                patched_orig[i] = patched[j:j + 1]                       # [1, C, H, W]

        if _prof:
            self._prof.setdefault("prepare/patch_apply", []).append(_pt() - _t1)
            _t2 = _pt()

        # ── Batched diff_prep per size group ─────────────────────────────
        # All images in a size group share the same H×W → same letterbox
        # parameters → one F.interpolate call replaces N sequential calls.
        prep_results = [None] * len(loaded)
        for indices in by_size.values():
            batch     = torch.cat([patched_orig[i] for i in indices])    # [G, C, H, W]
            corners_b = torch.stack([loaded[i][1] for i in indices])     # [G, 4, 2]
            prep_b, new_corners_b = self._diff_prep_batch(batch, corners_b)
            for j, i in enumerate(indices):
                prep_results[i] = (prep_b[j], new_corners_b[j])

        if _prof:
            self._prof.setdefault("prepare/diff_prep", []).append(_pt() - _t2)

        result = []
        for i, (orig_tensor, orig_corners, label) in enumerate(loaded):
            po                        = patched_orig[i]                  # [1, C, H, W]
            patched_prep_chw, new_corners = prep_results[i]
            target_box = self.corners_to_bbox(new_corners)
            rim_box    = self._rim_bbox(new_corners)
            ocr_crop   = _bbox_ocr_crop(po, orig_corners, self.ocr.ocr_crop_size)
            if self.top_extend:
                top_region_box   = self._top_extend_region_bbox(new_corners)
                top_corners_orig = self._top_extend_region_corners(orig_corners)
                top_ocr_crop     = _bbox_ocr_crop(po, top_corners_orig, self.ocr.ocr_crop_size)
            else:
                top_region_box = None
                top_ocr_crop   = None
            result.append({
                "patched_prep":   patched_prep_chw,  # [C, H_p, W_p]
                "target_box":     target_box,         # [4]
                "rim_box":        rim_box,             # [4]
                "new_corners":    new_corners,         # [4, 2]
                "ocr_crop":       ocr_crop,            # [1, 3, H_c, W_c]
                "top_region_box": top_region_box,      # [4] or None
                "top_ocr_crop":   top_ocr_crop,        # [1, 3, H_c, W_c] or None
                "label":          label,
            })

        if _prof:
            self._prof.setdefault("prepare/total", []).append(_pt() - _t_batch0)
        return result

    def _prepare_one(self, batch_item: dict, patch_norm: torch.Tensor,
                     augment: Optional[bool] = None) -> dict:
        """Single-image wrapper around _prepare_batch."""
        return self._prepare_batch([batch_item], patch_norm, augment=augment)[0]

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

        two_target_results = self.detector.differentiable_predict_box_batch_two_targets(
            batched_prep, target_boxes, top_region_boxes)

        # ── Pass 1: detection losses + crop extraction (no OCR calls) ────
        det_losses, crop_info = [], []
        for i, ((conf_real, pred_box), (conf_top, top_pred_box)) in enumerate(two_target_results):
            if pred_box is not None:
                iou_real = self._boxes_iou(
                    pred_box.unsqueeze(0),
                    items[i]["target_box"].to(self.device).unsqueeze(0)).squeeze()
                det_real = iou_real
            else:
                det_real = conf_real

            if top_pred_box is not None:
                iou_top = self._boxes_iou(
                    top_pred_box.unsqueeze(0),
                    items[i]["top_region_box"].to(self.device).unsqueeze(0)).squeeze()
                det_top = -iou_top
            else:
                det_top = -conf_top

            det_losses.append((det_real, det_top))

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

            crop_info.append((real_crop, top_crop))

        # ── Pass 2: batch OCR — one model call for all crops ──────────────
        image_losses = []
        det_real_l_list, det_top_l_list, ocr_real_l_list, ocr_top_l_list = [], [], [], []
        use_encode = hasattr(self.ocr, 'encode_batch')

        if use_encode:
            flat_crops, crop_idx_per_item = [], []
            for real_crop, top_crop in crop_info:
                ri = (len(flat_crops), flat_crops.append(real_crop))[0] if real_crop is not None else None
                ti = (len(flat_crops), flat_crops.append(top_crop))[0]  if top_crop  is not None else None
                crop_idx_per_item.append((ri, ti))
            raw_all = self.ocr.encode_batch(flat_crops) if flat_crops else None

        _zero = torch.tensor(0.0, device=self.device)
        for i, (real_crop, top_crop) in enumerate(crop_info):
            det_real_i, det_top_i = det_losses[i]

            if use_encode:
                ri, ti = crop_idx_per_item[i]
                ocr_real_i = (self.ocr.loss_from_raw(raw_all[ri], self.impersonation_target,
                                                      True, None)                              if ri is not None else None)
                ocr_top_i  = (self.ocr.loss_from_raw(raw_all[ti], self.impersonation_target,
                                                      True, None)                              if ti is not None else None)
            else:
                ocr_real_i = (self.ocr.differentiable_loss_batch(
                                  [real_crop], self.impersonation_target,
                                  impersonation=True)[0]                              if real_crop is not None else None)
                ocr_top_i  = (self.ocr.differentiable_loss_batch(
                                  [top_crop], self.impersonation_target,
                                  impersonation=True)[0]                              if top_crop is not None else None)

            # Weights: proportional to each detection term's magnitude.
            # Not detached: gradients flow through the weights so that reducing
            # det_top_mag (failing to attract detection) increases w_real and thus
            # the weighted-average OCR loss, penalising poor top-region detection.
            det_real_mag = det_real_i.clamp(min=0)
            det_top_mag  = (-det_top_i).clamp(min=0)
            total_mag    = det_real_mag + det_top_mag + 1e-6

            if ocr_real_i is not None and ocr_top_i is not None:
                w_real = det_real_mag / total_mag
                w_top  = det_top_mag  / total_mag
                ocr_i  = w_real * ocr_real_i + w_top * ocr_top_i
            elif ocr_real_i is not None:
                ocr_i = ocr_real_i
            elif ocr_top_i is not None:
                ocr_i = ocr_top_i
            else:
                ocr_i = _zero

            image_losses.append(ocr_i)
            det_real_l_list.append(det_real_i.detach())
            det_top_l_list.append(det_top_i.detach())
            ocr_real_l_list.append(ocr_real_i.detach() if ocr_real_i is not None else _zero)
            ocr_top_l_list.append(ocr_top_i.detach()  if ocr_top_i  is not None else _zero)

        det_top_l_for_loss = torch.stack([d for _, d in det_losses]).mean()
        total      = (torch.stack(image_losses).mean()
                      + self.tv_weight * tv_l
                      + self.det_loss_weight * det_top_l_for_loss)
        det_real_l = torch.stack(det_real_l_list).mean()
        det_top_l  = torch.stack(det_top_l_list).mean()
        ocr_real_l = torch.stack(ocr_real_l_list).mean()
        ocr_top_l  = torch.stack(ocr_top_l_list).mean()

        return (total,
                det_real_l, det_top_l,
                ocr_real_l, ocr_top_l,
                (self.tv_weight * tv_l).detach())

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

    def save_patch(self, global_update: int, subdir: str = "patches",
                   stem: Optional[str] = None) -> None:
        save_dir = self.run_dir / subdir
        save_dir.mkdir(parents=True, exist_ok=True)
        if stem is None:
            stem = f"patch_{self.detector.name}_update_{global_update:06d}"
        with torch.no_grad():
            patch_img = self.generate_patch()   # [3, H, W] in [0,1]
            T.ToPILImage()(patch_img.cpu()).save(str(save_dir / f"{stem}.png"))
            torch.save({
                "seed":          self.seed.detach().cpu(),
                "decoder":       self.decoder.state_dict(),
                "seed_channels": self.seed_channels,
                "global_update": global_update,
                "backend":       self.detector.name,
                "ocr":           self.ocr.name,
                "patch_size":    (self.patch_height, self.patch_width),
                "patch":         patch_img.cpu(),   # rendered tensor (incl. print_blur)
            }, str(save_dir / f"{stem}.pt"))

    # ====================================================================
    # Pre-training sanity check
    # ====================================================================

    def _probe_eval_batch_size(self, sample_raw: dict) -> int:
        """
        Estimate the optimal eval_batch_size by running one real item through
        _prepare_one + compute_loss_batch + backward, measuring peak GPU memory,
        and scaling to available free memory with a 0.70 safety margin.

        Only meaningful on CUDA; returns the current eval_batch_size unchanged
        on other devices.
        """
        if not self.device.startswith("cuda"):
            return self.eval_batch_size

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        free_mem, total_mem = torch.cuda.mem_get_info()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        try:
            probe_patch = self.generate_patch(training_aug=False).detach().requires_grad_(True)
            item = self._prepare_one(sample_raw, probe_patch)
            item["_patch_norm"] = probe_patch
            loss, *_ = self.compute_loss_batch([item])
            (loss / 4).backward()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            per_sample_mem = max(1, peak - baseline)
        except Exception as e:
            print(f"  [auto-batch] probe failed ({e}) — keeping eval_batch_size=1")
            return 1
        finally:
            torch.cuda.empty_cache()

        free_after = torch.cuda.mem_get_info()[0]
        safety = 0.90
        bs = max(1, int(free_after * safety / per_sample_mem))
        print(
            f"  [auto-batch] {free_after/1024**3:.1f}/{total_mem/1024**3:.1f} GB free"
            f"  |  {per_sample_mem/1024**2:.0f} MB/sample"
            f"  →  eval_batch_size={bs}"
        )
        return bs

    def validate_pipeline(self, loader=None) -> None:
        """
        Run one full gradient-accumulation cycle (B * update_every items) through
        the training forward+backward path to verify the pipeline end-to-end.
        Raises on any crash.

        When eval_batch_size==1 (the default) and running on CUDA, automatically
        probes peak memory cost per sample and sets eval_batch_size to fill the
        available GPU memory (same approach as finetune_all_models.find_batch_size).
        """
        print("\n── Pre-training sanity check ──────────────────────────────")
        _loader = loader if loader is not None else self.train_loader
        B = self.eval_batch_size
        update_every = (self.grad_accumulate
                        if self.grad_accumulate is not None
                        else min(4, len(_loader)))

        # Load the first item so we can probe memory cost before committing to
        # a batch size (and therefore a total item count to load).
        _iter = iter(_loader)
        try:
            first_batch = next(_iter)
        except StopIteration:
            raise RuntimeError("No training data found for sanity check.")
        items_raw = [{k: v[0] for k, v in first_batch.items()}]

        # Auto-detect batch size when using the default of 1.
        if B == 1:
            self.detector.train_mode()
            self.ocr.train()
            try:
                B = self._probe_eval_batch_size(items_raw[0])
            finally:
                self.detector.eval()
                self.detector.freeze()
                self.ocr.eval()
                self.ocr.freeze()
            self.eval_batch_size = B

        need = B * update_every

        # Load remaining items to fill one full accumulation window.
        for batch in _iter:
            items_raw.append({k: v[0] for k, v in batch.items()})
            if len(items_raw) >= need:
                break

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
            orig_tensor = batch["orig_image"][0].to(self.device, non_blocking=True)
            if orig_tensor.dtype == torch.uint8:
                orig_tensor = orig_tensor.float().div_(255.0)
            orig_corners_np = batch["orig_corners"][0].cpu().numpy()
            orig_corners    = batch["orig_corners"][0].to(self.device, non_blocking=True)

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
            img_plate_text = (batch.get("label", [None])[0]) or self.expected_plate_text
            is_correct = self._plate_text_matches(text, img_plate_text)

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
                target_text = img_plate_text
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

        def _accumulate(indices, weight):
            """
            Generate patch, detach → leaf, accumulate gradients in batches of B,
            then backprop once through the generator.
            Returns (total_loss, det_real_sum, det_top_sum, ocr_real_sum, ocr_top_sum, tv_sum)
            as a sum over all items (consistent with the /B normalisation applied by the caller).

            compute_loss_batch returns the mean over its batch, so to reproduce
            the same gradient as processing items individually we scale the
            backward by len(chunk): mean * len(chunk) * weight == sum * weight.
            """
            patch_with_graph = self.generate_patch(training_aug=self.training)
            patch_leaf = patch_with_graph.detach().requires_grad_(True)

            total_loss = det_real_sum = det_top_sum = ocr_real_sum = ocr_top_sum = tv_sum = 0.0
            for chunk_start in range(0, len(indices), B):
                chunk = indices[chunk_start:chunk_start + B]
                items = []
                for i in chunk:
                    item = self._prepare_one(window_raw[i], patch_leaf)
                    item["_patch_norm"] = patch_leaf
                    items.append(item)
                loss, det_real_l, det_top_l, ocr_real_l, ocr_top_l, tv_l = self.compute_loss_batch(items)
                # loss is mean over len(chunk); scale back to sum for grad equivalence
                (loss * weight * len(chunk)).backward()
                total_loss    += loss.item()       * len(chunk)
                det_real_sum  += det_real_l.item() * len(chunk)
                det_top_sum   += det_top_l.item()  * len(chunk)
                ocr_real_sum  += ocr_real_l.item() * len(chunk)
                ocr_top_sum   += ocr_top_l.item()  * len(chunk)
                tv_sum        += tv_l.item()        * len(chunk)
                del items

            # Propagate accumulated patch gradient through the generator graph.
            patch_with_graph.backward(patch_leaf.grad)
            return total_loss, det_real_sum, det_top_sum, ocr_real_sum, ocr_top_sum, tv_sum

        # ── ASCENT: gradients from m random items ────────────────────────
        optimizer.zero_grad()
        ascent_idx = sorted(random.sample(range(M), m))
        _accumulate(ascent_idx, 1.0 / m)

        torch.nn.utils.clip_grad_norm_(self._trainable_params(), max_norm=1.0)
        optimizer.first_step(zero_grad=True)

        # ── DESCENT: gradients from all M items, perturbed patch ─────────
        total_loss, det_real_sum, det_top_sum, ocr_real_sum, ocr_top_sum, tv_sum = \
            _accumulate(range(M), 1.0 / M)

        torch.nn.utils.clip_grad_norm_(self._trainable_params(), max_norm=1.0)
        optimizer.second_step(zero_grad=True)

        return total_loss, det_real_sum, det_top_sum, ocr_real_sum, ocr_top_sum, tv_sum

    def train_epoch(self, optimizer, epoch: int, scheduler=None,
                    update_log: Optional[list] = None,
                    update_offset: int = 0,
                    save_every: int = 0,
                    tv_warmup_updates: int = 0) -> Tuple[float, float, float, float]:
        if self.ocr.is_trainable:
            self.ocr.train()

        B = self.eval_batch_size
        update_every = (len(self.train_loader)
                        if self.grad_accumulate is None
                        else self.grad_accumulate)
        total_loss = accum_loss = 0.0
        total_det_real = total_det_top = total_ocr_real = total_ocr_top = total_tv = 0.0
        # Per-update accumulators (reset after each optimizer step, like accum_loss)
        _upd_det_real = _upd_det_top = _upd_ocr_real = _upd_ocr_top = _upd_tv = 0.0
        # GPU-side loss accumulator — avoids .item() syncs inside the hot loop.
        # Flushed with one .tolist() call per optimizer step instead.
        _loss_accum_t = torch.zeros(6, device=self.device)
        step = num_updates = 0
        buffer: list = []
        use_sam = isinstance(optimizer, SAM)
        window_raw: list = []   # SAM: raw batch items for the current update window
        # _last_save_milestone is owned by train() and persists across epochs.
        _saved_tv_weight = self.tv_weight  # Save original TV weight for warmup scaling

        # Generate the first patch (with generator graph) for the accumulation window.
        # We use a detach-accumulate-backprop pattern: detach the patch into a
        # leaf tensor so each per-item backward() frees its graph immediately
        # (no retain_graph needed).  Accumulated gradients on the leaf are then
        # back-propagated through the generator in a single call.
        patch_with_graph = self.generate_patch(training_aug=self.training)
        patch_leaf = patch_with_graph.detach().requires_grad_(True)

        # Progress bar tracks optimizer updates, not individual images.
        updates_per_epoch = max(1, len(self.train_loader) // (B * update_every))
        with tqdm(total=updates_per_epoch,
                  desc=f"Epoch {epoch+1}",
                  leave=False) as pbar:
            for idx, batch in enumerate(self.train_loader):
                try:
                    raw_item = {k: v[0] for k, v in batch.items()}

                    if use_sam:
                        window_raw.append(raw_item)
                        window_full = len(window_raw) == B * update_every
                        if not window_full:
                            continue

                        # Apply TV loss warmup: linearly scale from 0 to original weight
                        global_update_before = update_offset + num_updates
                        if tv_warmup_updates > 0 and global_update_before < tv_warmup_updates:
                            tv_warmup_factor = min(1.0, global_update_before / tv_warmup_updates)
                            self.tv_weight = _saved_tv_weight * tv_warmup_factor
                        else:
                            self.tv_weight = _saved_tv_weight

                        loss_t, det_real_t, det_top_t, ocr_real_t, ocr_top_t, tv_t = \
                            self._msam_step(optimizer, window_raw, B, update_every)
                        window_raw = []
                        step       += update_every
                        num_updates += 1
                        # _msam_step sums losses over all M=B*update_every items
                        # individually; divide by B so the scale matches the
                        # non-SAM path (which averages over B inside compute_loss_batch).
                        total_loss     += loss_t      / B
                        total_det_real += det_real_t  / B
                        total_det_top  += det_top_t   / B
                        total_ocr_real += ocr_real_t  / B
                        total_ocr_top  += ocr_top_t   / B
                        total_tv       += tv_t        / B
                        if scheduler is not None:
                            scheduler.step()
                        _lr_now = optimizer.base_optimizer.param_groups[0]["lr"]
                        if update_log is not None:
                            update_log.append({
                                "global_update": update_offset + num_updates,
                                "epoch":    epoch + 1,
                                "loss":     loss_t     / B / update_every,
                                "det_real": det_real_t / B / update_every,
                                "det_top":  det_top_t  / B / update_every,
                                "ocr_real": ocr_real_t / B / update_every,
                                "ocr_top":  ocr_top_t  / B / update_every,
                                "tv":       tv_t       / B / update_every,
                                "lr":       _lr_now,
                            })
                        # Check for 10% milestone checkpoint
                        if save_every > 0:
                            global_update = update_offset + num_updates
                            milestone = global_update // save_every
                            if milestone > self._last_save_milestone:
                                self._last_save_milestone = milestone
                                self.save_patch(global_update, "patches")
                        pbar.update(1)
                        pbar.set_postfix({
                            "loss":   f"{total_loss/step:.4f}",
                            "det_r":  f"{total_det_real/step:.4f}",
                            "det_t":  f"{total_det_top/step:.4f}",
                            "ocr_r":  f"{total_ocr_real/step:.4f}",
                            "ocr_t":  f"{total_ocr_top/step:.4f}",
                            "tv":     f"{total_tv/step:.4f}",
                            "lr":     f"{_lr_now:.2e}",
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

                    # Apply TV loss warmup: linearly scale from 0 to original weight
                    # Check warmup at the start of the accumulation window
                    global_update_for_batch = update_offset + num_updates + 1
                    if tv_warmup_updates > 0 and global_update_for_batch <= tv_warmup_updates:
                        tv_warmup_factor = min(1.0, global_update_for_batch / tv_warmup_updates)
                        self.tv_weight = _saved_tv_weight * tv_warmup_factor
                    else:
                        self.tv_weight = _saved_tv_weight

                    loss, det_real_l, det_top_l, ocr_real_l, ocr_top_l, tv_l = \
                        self.compute_loss_batch(buffer)
                    buffer = []
                    scaled_loss = loss / update_every
                    step += 1
                    scaled_loss.backward()  # frees item graph; accumulates into patch_leaf.grad

                    _loss_accum_t.add_(torch.stack(
                        [loss, det_real_l, det_top_l, ocr_real_l, ocr_top_l, tv_l]
                    ).detach())
                    del loss, scaled_loss

                    if step % update_every == 0:
                        # Propagate accumulated patch gradient through generator.
                        patch_with_graph.backward(patch_leaf.grad)
                        torch.nn.utils.clip_grad_norm_(self._trainable_params(), max_norm=1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                        if scheduler is not None:
                            scheduler.step()
                        # Single .tolist() here = 1 GPU sync per optimizer step
                        # instead of 11 syncs per accumulation batch.
                        accum_loss, _upd_det_real, _upd_det_top, \
                            _upd_ocr_real, _upd_ocr_top, _upd_tv = _loss_accum_t.tolist()
                        _loss_accum_t.zero_()
                        total_det_real += _upd_det_real
                        total_det_top  += _upd_det_top
                        total_ocr_real += _upd_ocr_real
                        total_ocr_top  += _upd_ocr_top
                        total_tv       += _upd_tv
                        total_loss  += accum_loss
                        num_updates += 1
                        _lr_now = optimizer.param_groups[0]["lr"]
                        if update_log is not None:
                            update_log.append({
                                "global_update": update_offset + num_updates,
                                "epoch":    epoch + 1,
                                "loss":     accum_loss     / update_every,
                                "det_real": _upd_det_real  / update_every,
                                "det_top":  _upd_det_top   / update_every,
                                "ocr_real": _upd_ocr_real  / update_every,
                                "ocr_top":  _upd_ocr_top   / update_every,
                                "tv":       _upd_tv        / update_every,
                                "lr":       _lr_now,
                            })
                        # Check for 10% milestone checkpoint
                        if save_every > 0:
                            global_update = update_offset + num_updates
                            milestone = global_update // save_every
                            if milestone > self._last_save_milestone:
                                self._last_save_milestone = milestone
                                self.save_patch(global_update, "patches")
                        accum_loss = _upd_det_real = _upd_det_top = \
                            _upd_ocr_real = _upd_ocr_top = _upd_tv = 0.0
                        pbar.update(1)
                        pbar.set_postfix({
                            "loss":  f"{total_loss/step:.4f}",
                            "det_r": f"{total_det_real/step:.4f}",
                            "det_t": f"{total_det_top/step:.4f}",
                            "ocr_r": f"{total_ocr_real/step:.4f}",
                            "ocr_t": f"{total_ocr_top/step:.4f}",
                            "tv":    f"{total_tv/step:.4f}",
                            "lr":    f"{_lr_now:.2e}",
                        })
                        # New patch for next accumulation window
                        patch_with_graph = self.generate_patch(training_aug=self.training)
                        patch_leaf = patch_with_graph.detach().requires_grad_(True)

                except Exception:
                    print(f"\n[WARNING] Skipping batch {idx} due to error:")
                    traceback.print_exc()
                    # Reset batch state so the next iteration starts clean
                    buffer = []
                    window_raw = []
                    optimizer.zero_grad()
                    patch_with_graph = self.generate_patch(training_aug=self.training)
                    patch_leaf = patch_with_graph.detach().requires_grad_(True)

            # ── Flush remainder at end of epoch ──────────────────────────
            if use_sam and len(window_raw) >= B:
                # Partial window: trim to a multiple of B, then do m-SAM update
                n_complete = (len(window_raw) // B) * B
                loss_t, det_real_t, det_top_t, ocr_real_t, ocr_top_t, tv_t = \
                    self._msam_step(optimizer, window_raw[:n_complete], B, n_complete // B)
                step           += n_complete // B
                num_updates    += 1
                total_loss     += loss_t      / B
                total_det_real += det_real_t  / B
                total_det_top  += det_top_t   / B
                total_ocr_real += ocr_real_t  / B
                total_ocr_top  += ocr_top_t   / B
                total_tv       += tv_t        / B
                if update_log is not None:
                    _n = n_complete // B
                    _opt = optimizer.base_optimizer
                    update_log.append({
                        "global_update": update_offset + num_updates,
                        "epoch":    epoch + 1,
                        "loss":     loss_t     / B / _n,
                        "det_real": det_real_t / B / _n,
                        "det_top":  det_top_t  / B / _n,
                        "ocr_real": ocr_real_t / B / _n,
                        "ocr_top":  ocr_top_t  / B / _n,
                        "tv":       tv_t       / B / _n,
                        "lr":       _opt.param_groups[0]["lr"],
                    })
            elif not use_sam:
                # Flush remainder buffer (< B images left at end of epoch)
                if buffer:
                    # Apply TV loss warmup for end-of-epoch flush
                    global_update_for_flush = update_offset + num_updates + 1
                    if tv_warmup_updates > 0 and global_update_for_flush <= tv_warmup_updates:
                        tv_warmup_factor = min(1.0, global_update_for_flush / tv_warmup_updates)
                        self.tv_weight = _saved_tv_weight * tv_warmup_factor
                    else:
                        self.tv_weight = _saved_tv_weight

                    loss, det_real_l, det_top_l, ocr_real_l, ocr_top_l, tv_l = \
                        self.compute_loss_batch(buffer)
                    buffer = []
                    scaled_loss = loss / update_every
                    scaled_loss.backward()  # accumulates into patch_leaf.grad
                    _loss_accum_t.add_(torch.stack(
                        [loss, det_real_l, det_top_l, ocr_real_l, ocr_top_l, tv_l]
                    ).detach())
                    step           += 1
                    del loss, scaled_loss
                    accum_loss, _upd_det_real, _upd_det_top, \
                        _upd_ocr_real, _upd_ocr_top, _upd_tv = _loss_accum_t.tolist()
                    _loss_accum_t.zero_()
                    total_det_real += _upd_det_real
                    total_det_top  += _upd_det_top
                    total_ocr_real += _upd_ocr_real
                    total_ocr_top  += _upd_ocr_top
                    total_tv       += _upd_tv

                if step % update_every != 0 and self.grad_accumulate is not None:
                    # Propagate remainder gradients through generator.
                    patch_with_graph.backward(patch_leaf.grad)
                    torch.nn.utils.clip_grad_norm_(self._trainable_params(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    if scheduler is not None:
                        scheduler.step()
                    total_loss  += accum_loss
                    num_updates += 1
                    if update_log is not None:
                        _rem = step % update_every or update_every
                        update_log.append({
                            "global_update": update_offset + num_updates,
                            "epoch":    epoch + 1,
                            "loss":     accum_loss    / _rem,
                            "det_real": _upd_det_real / _rem,
                            "det_top":  _upd_det_top  / _rem,
                            "ocr_real": _upd_ocr_real / _rem,
                            "ocr_top":  _upd_ocr_top  / _rem,
                            "tv":       _upd_tv       / _rem,
                            "lr":       optimizer.param_groups[0]["lr"],
                        })

        # Restore original TV weight
        self.tv_weight = _saved_tv_weight

        n = max(step, 1)
        return (total_loss / n,
                total_det_real / n, total_det_top / n,
                total_ocr_real / n, total_ocr_top / n,
                total_tv / n, num_updates)

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

                total, *_ = self.compute_loss_batch(buffer)
                losses.append(total.item())
                buffer = []

            if buffer:
                total, *_ = self.compute_loss_batch(buffer)
                losses.append(total.item())

        return float(np.mean(losses)) if losses else 0.0

    def train(
        self,
        num_epochs:    int   = 100,
        learning_rate: float = 5e-4,
        lr_min:        float = 1e-5,
        dry_run:       bool  = False,
        tv_warmup:     float = 0.1,
        continue_path: Optional[str] = None,
        continue_lr:   bool = False,
    ) -> dict:
        """
        tv_warmup: fraction of total gradient updates during which TV loss is
        suppressed so the patch can move freely early on.  Set to 0.0 to
        disable TV warmup entirely.

        continue_path: path to a .pt checkpoint produced by save_patch().
                       Loads seed + decoder weights and resumes from
                       global_update stored in the checkpoint.
        continue_lr:   if True, fast-forward the new LR schedule to the
                       checkpoint's global_update so training continues
                       from the same point in the schedule.  Without this
                       flag (default) the schedule always resets to the
                       LR params passed in.
        """
        if dry_run:
            print("\nDry run: saving debug images...")
            self.save_debug_images()
            print("Dry run complete.")
            return {}

        # ── Load checkpoint (--continue) ──────────────────────────────
        ckpt_global_update = 0
        if continue_path:
            ckpt = torch.load(continue_path, map_location="cpu")
            with torch.no_grad():
                self.seed.copy_(ckpt["seed"].to(self.device))
            self.decoder.load_state_dict(ckpt["decoder"])
            self.decoder.to(self.device)
            ckpt_global_update = int(ckpt.get("global_update", 0))
            print(f"  Resumed from  : {continue_path}  (update {ckpt_global_update})")

        eta_min       = lr_min

        # Cap eval_batch_size so total gradient updates >= 10 000.
        _update_every_cap = self.grad_accumulate or len(self.train_loader)
        _max_bs = max(1, num_epochs * len(self.train_loader) // (10_000 * _update_every_cap))
        if self.eval_batch_size > _max_bs:
            print(f"[auto-batch] capping eval_batch_size {self.eval_batch_size} → {_max_bs} "
                  f"to ensure ≥10 000 total updates")
            self.eval_batch_size = _max_bs

        # Resolve auto sam_m now that eval_batch_size is known (probed in sanity check).
        if self._sam_m_auto:
            self.sam_m = max(1, (self.grad_accumulate or 64) * self.eval_batch_size // 4)
            self._sam_m_auto = False
            print(f"[m-SAM] auto sam_m={self.sam_m}  "
                  f"(grad_accumulate={self.grad_accumulate or 64} × eval_batch_size={self.eval_batch_size} // 4)")

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
        # Scheduler counts are in gradient updates, not epochs.
        # _updates_per_epoch is computed below alongside tv_warmup; duplicate
        # the formula here so we can build the scheduler before the print block.
        _ue_update_every = (len(self.train_loader)
                            if self.grad_accumulate is None
                            else self.grad_accumulate)
        _ue_per_epoch = max(1, len(self.train_loader) //
                            (self.eval_batch_size * _ue_update_every))
        total_updates_sched = num_epochs * _ue_per_epoch
        warmup_updates = max(1, int(0.1 * total_updates_sched))
        cosine_updates = total_updates_sched - warmup_updates
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            sched_optimizer,
            start_factor=eta_min / learning_rate,
            end_factor=1.0,
            total_iters=max(1, warmup_updates),
        )
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            sched_optimizer, T_max=max(1, cosine_updates), eta_min=eta_min,
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            sched_optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[max(1, warmup_updates)],
        )

        # Fast-forward scheduler when --continue-lr is requested.
        # We call scheduler.step() once per past update so that the
        # scheduler's internal state (last_epoch) reaches the right
        # position.  This is pure Python math so it completes quickly
        # even for tens-of-thousands of updates.
        if continue_lr and ckpt_global_update > 0:
            steps_to_skip = min(ckpt_global_update, total_updates_sched)
            print(f"  LR schedule  : fast-forwarding {steps_to_skip} steps to match checkpoint")
            for _ in range(steps_to_skip):
                scheduler.step()

        history    = {"loss": [], "val_score": [], "learning_rate": [],
                      "det_real": [], "det_top": [], "ocr_real": [], "ocr_top": [], "tv": []}
        best_loss  = float("inf")
        best_epoch = -1

        # Estimate total gradient updates for tv_warmup_updates calculation.
        # update_every mirrors the logic inside train_epoch.
        _update_every = (len(self.train_loader)
                         if self.grad_accumulate is None
                         else self.grad_accumulate)
        _updates_per_epoch = max(1, len(self.train_loader) //
                                 (self.eval_batch_size * _update_every))
        total_updates     = num_epochs * _updates_per_epoch
        tv_warmup_updates = int(tv_warmup * total_updates)
        save_every        = max(1, total_updates // 10)   # checkpoint every 10% of updates

        n_params = sum(p.numel() for p in self._trainable_params())
        print(f"\n{'='*60}")
        print(f"  Adversarial Patch Training")
        print(f"  Detector  : {self.detector.name}   OCR: {self.ocr.name}")
        print(f"  Patch gen : ConvTranspose decoder  "
              f"(seed {self.seed_channels}ch×4×8 → 3×256×512)")
        print(f"  Trainable : {n_params:,} params  "
              f"(seed {self.seed.numel():,}  +  decoder {n_params-self.seed.numel():,})")
        print(f"  Dataset   : {len(self.train_loader)+len(self.val_loader)} images")
        print(f"  Epochs    : {num_epochs}  |  Updates: {total_updates}  |  "
              f"Save every: {save_every} updates (~10%)")
        print(f"  LR warmup : {warmup_updates} updates  |  "
              f"LR: {eta_min:.0e} → {learning_rate:.0e} → {eta_min:.0e}")
        if tv_warmup_updates > 0:
            print(f"  TV warmup : {tv_warmup:.0%} of updates = {tv_warmup_updates} updates")
        else:
            print(f"  TV warmup : disabled")
        if self.sam_m is not None:
            print(f"  Optimizer : m-SAM  (m={self.sam_m}, rho={self.sam_rho}, base=AdamW)")
        _mode = ('impersonation → ' + self.impersonation_target) if self.impersonation_target else 'disruption'
        if self.disable_disruption:
            _mode += '  [detection loss disabled]'
        print(f"  Mode      : {_mode}")
        print(f"  Run dir   : {self.run_dir}")
        print(f"{'='*60}\n")

        _resuming     = continue_path is not None
        log_path      = self.run_dir / "training_log.txt"
        log_file      = open(log_path, "a" if _resuming else "w")
        batch_log_path = self.run_dir / "batch_log.csv"
        _batch_log_fields = ["global_update", "epoch", "loss",
                             "det_real", "det_top", "ocr_real", "ocr_top", "tv", "lr"]
        _batch_log_write_header = not (_resuming and batch_log_path.exists())
        batch_log_file = open(batch_log_path, "a" if _resuming else "w", newline="")
        batch_log_writer = csv.writer(batch_log_file)
        if _batch_log_write_header:
            batch_log_writer.writerow(_batch_log_fields)

        global_updates            = ckpt_global_update
        self._last_save_milestone = ckpt_global_update // save_every if save_every > 0 else 0
        for epoch in range(num_epochs):
            epoch_start    = time.time()
            self.training  = True
            epoch_update_records: list = []
            (train_loss,
             train_det_real, train_det_top,
             train_ocr_real, train_ocr_top,
             train_tv, epoch_updates) = self.train_epoch(
                optimizer, epoch, scheduler,
                update_log=epoch_update_records,
                update_offset=global_updates,
                save_every=save_every,
                tv_warmup_updates=tv_warmup_updates,
            )
            for rec in epoch_update_records:
                batch_log_writer.writerow([rec[f] for f in _batch_log_fields])
            batch_log_file.flush()
            global_updates += epoch_updates
            self.training  = False
            val_loss       = self.validate()
            lr = optimizer.param_groups[0]["lr"]
            epoch_time     = time.time() - epoch_start

            history["loss"].append(train_loss)
            history["val_score"].append(val_loss)
            history["learning_rate"].append(lr)
            history["det_real"].append(train_det_real)
            history["det_top"].append(train_det_top)
            history["ocr_real"].append(train_ocr_real)
            history["ocr_top"].append(train_ocr_top)
            history["tv"].append(train_tv)

            init_val    = history["val_score"][0]
            change      = (val_loss / (init_val + 1e-9) - 1) * 100
            best_marker = ""

            if val_loss < best_loss:
                best_loss  = val_loss
                best_epoch = epoch
                self.save_patch(global_updates, "patches", stem=f"patch_{self.detector.name}_best")
                best_marker = "  ★ best"

            line = (f"Epoch {epoch+1:3d}/{num_epochs} "
                    f"[{self.detector.name}/{self.ocr.name}] | "
                    f"loss: {train_loss:.4f}  "
                    f"det_r: {train_det_real:.4f}  det_t: {train_det_top:.4f}  "
                    f"ocr_r: {train_ocr_real:.4f}  ocr_t: {train_ocr_top:.4f}  "
                    f"tv: {train_tv:.4f} | "
                    f"val: {val_loss:.4f} Δ{change:+.1f}% | "
                    f"lr: {lr:.2e} | "
                    f"time: {epoch_time:.1f}s{best_marker}")
            print(line)
            log_file.write(line + "\n")
            log_file.flush()

        log_file.close()
        batch_log_file.close()

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
    parser.add_argument("--ccpd-train-csv", default=None,
                        help="Path to train_split.csv written by finetune_all_models.py. "
                             "When set, overrides --csv and uses CCPD data with per-image "
                             "ground-truth labels.")

    TRAINABLE_DET = ["sam", "yolov8", "fasterrcnn", "yolov11", "rtdetr", "owlvit", "yolo-v9-608"]
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
    parser.add_argument("--grad-accumulate", type=int, default=1)
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
                        help="Weight for total variation loss (default: 10.0).")
    parser.add_argument("--det-loss-weight", type=float, default=0.0,
                        help="Weight for the impersonation-zone detection loss added directly "
                             "to the total loss (default: 0.0, disabled).")
    parser.add_argument("--tv-warmup", type=float, default=0.1,
                        help="Fraction of total gradient updates to suppress TV loss "
                             "so the patch can move freely early on (default: 0.1). "
                             "Set to 0 to disable TV warmup entirely.")
    parser.add_argument("--no-disruption", action="store_true",
                        help="Disable the detection (disruption) loss component entirely. "
                             "Detection is still computed for pipeline purposes but contributes "
                             "zero gradient to the total loss.")
    parser.add_argument("--eval-batch-size", type=int, default=1,
                        help="Number of images to batch for detector/OCR evaluation (default 1).")
    parser.add_argument("--sam-m", type=int, default=None,
                        help="m-SAM ascent step size.  Defaults to grad-accumulate//4 "
                             "(auto).  Set to 0 to disable m-SAM entirely.")
    parser.add_argument("--sam-rho", type=float, default=0.025,
                        help="SAM perturbation radius rho (default 0.025). "
                             "Only used when m-SAM is enabled.")
    parser.add_argument("--top-extend", action="store_true", default=True,
                        help="Double patch height upward: suppress real-plate detection "
                             "and attract detection into the attacker-controlled top region.")
    parser.add_argument("--no-top-extend", dest="top_extend", action="store_false")
    parser.add_argument("--augment", action="store_true",
                        help="Apply differentiable photometric augmentations (brightness, "
                             "contrast, saturation, color temperature, directional shadow) "
                             "after patch application at each training step.")
    parser.add_argument("--continue", dest="continue_path", default=None, metavar="CHECKPOINT",
                        help="Path to a .pt checkpoint produced by a previous run.  "
                             "Loads seed + decoder weights and resumes from that run's directory.  "
                             "By default the LR schedule resets to the new --lr / --lr-min values; "
                             "use --continue-lr to fast-forward it instead.")
    parser.add_argument("--continue-lr", action="store_true",
                        help="When resuming from a checkpoint (--continue), advance the new LR "
                             "schedule by the checkpoint's global_update count so training "
                             "continues from the same schedule position.  Without this flag "
                             "(default) the schedule always resets to the new --lr / --lr-min.")
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

    # sam_m auto-detection is deferred to train() where eval_batch_size is already known.
    _sam_m_auto = (args.sam_m is None)
    if args.sam_m == 0:
        args.sam_m = None   # None is the internal sentinel for "disabled"

    # If resuming, reuse the original run directory (two levels up from the .pt file).
    _continue_run_dir = None
    if args.continue_path:
        _continue_run_dir = str(Path(args.continue_path).parent.parent)

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
        det_loss_weight      = args.det_loss_weight,
        disable_disruption   = args.no_disruption,
        eval_batch_size      = args.eval_batch_size,
        sam_m                = args.sam_m,
        sam_m_auto           = _sam_m_auto,
        sam_rho              = args.sam_rho,
        skip_sanity          = args.skip_sanity,
        augment              = args.augment,
        top_extend           = args.top_extend,
        ccpd_csv             = args.ccpd_train_csv,
        continue_run_dir     = _continue_run_dir,
    )

    trainer.train(
        num_epochs    = args.epochs,
        learning_rate = args.lr,
        lr_min        = args.lr_min,
        dry_run       = args.dry_run,
        tv_warmup     = args.tv_warmup,
        continue_path = args.continue_path,
        continue_lr   = args.continue_lr,
    )


if __name__ == "__main__":
    main()
