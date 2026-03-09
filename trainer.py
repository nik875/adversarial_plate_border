"""
trainer.py  (rewritten)

AdversarialPatchTrainer — trains an adversarial border patch against a
pluggable detector + OCR backend pair.

Key design decisions
--------------------
* Full-resolution HEIC images (from preproc_labels.csv) are loaded; detector-
  specific preprocessing (letterbox, hard-resize, passthrough) is applied once
  at init time and cached in memory.
* The patch is applied to the detector-preprocessed image, and the OCR crop is
  taken from the same preprocessed image — no more inversion from detector space
  back to original space.
* Detection loss target: bounding box of plate corners directly (not the
  expanded border region).
* Detection loss proposal: best detection by max(IoU × confidence).
* Scheduler: CosineAnnealingLR (no more ReduceLROnPlateau).
* Outputs go into a unique timestamped run directory.
* validate_pipeline() sanity-checks the pipeline before training starts.
* save_debug_images() saves 20 annotated images to run_dir/debug/.
"""

from __future__ import annotations

import csv
import os
import warnings
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict

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
from dataset import create_dataloaders

warnings.filterwarnings("ignore")

PATCH_WIDTH  = 512
PATCH_HEIGHT = 256


# ---------------------------------------------------------------------------
# Helper: detector-specific preprocessing
# ---------------------------------------------------------------------------

def preprocess_for_detector(
    img_hwc_uint8: np.ndarray,
    corners_np: np.ndarray,
    backend: DetectorBackend,
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Apply detector-specific preprocessing to a single image.

    This is called once per image at init time (non-differentiable).
    Returns a CHW float32 [0,1] tensor and corners mapped into the new space.

    Parameters
    ----------
    img_hwc_uint8 : np.ndarray
        Original image in HWC uint8 format (BGR, from cv2/pillow_heif).
    corners_np : np.ndarray
        Plate corners, shape (4, 2), in original image pixel space.
    backend : DetectorBackend
        The detector whose preprocessing convention we must match.

    Returns
    -------
    img_tensor : torch.Tensor  [C, H, W]  float32  [0, 1]
    new_corners : np.ndarray   (4, 2)   float32  — corners in preprocessed space
    """
    name = backend.name
    H, W = img_hwc_uint8.shape[:2]

    if name in ("yolov8", "yolov11"):
        from ultralytics.utils.ops import letterbox
        # Read imgsz from the backend at runtime
        imgsz = 640
        if hasattr(backend, "_yolo") and backend._yolo is not None:
            raw = backend._yolo.imgsz
            imgsz = int(raw[0] if hasattr(raw, "__len__") else raw)
        img_lb, ratio, (dw, dh) = letterbox(
            img_hwc_uint8, (imgsz, imgsz),
            color=(114, 114, 114), auto=False, stride=32,
        )
        img_tensor = torch.from_numpy(img_lb).permute(2, 0, 1).float() / 255.0
        new_corners = corners_np * ratio + np.array([dw, dh], dtype=np.float32)

    elif name == "rtdetr":
        # Hard resize to 640×640, no padding, ÷255, no normalization
        img_resized = cv2.resize(img_hwc_uint8, (640, 640),
                                 interpolation=cv2.INTER_LINEAR)
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        sx, sy = 640.0 / W, 640.0 / H
        new_corners = corners_np * np.array([sx, sy], dtype=np.float32)

    elif name == "fasterrcnn":
        # Full-res passthrough ÷255.  GeneralizedRCNNTransform handles resize
        # and normalization internally; boxes are returned in original space.
        img_tensor = torch.from_numpy(img_hwc_uint8).permute(2, 0, 1).float() / 255.0
        new_corners = corners_np.copy().astype(np.float32)

    elif name == "yolo-v9-384":
        from ultralytics.utils.ops import letterbox
        img_lb, ratio, (dw, dh) = letterbox(
            img_hwc_uint8, (384, 384),
            color=(114, 114, 114), auto=False,
        )
        img_tensor = torch.from_numpy(img_lb).permute(2, 0, 1).float() / 255.0
        new_corners = corners_np * ratio + np.array([dw, dh], dtype=np.float32)

    else:
        # Fallback: letterbox to 384 for any unknown backend
        from ultralytics.utils.ops import letterbox
        img_lb, ratio, (dw, dh) = letterbox(
            img_hwc_uint8, (384, 384),
            color=(114, 114, 114), auto=False,
        )
        img_tensor = torch.from_numpy(img_lb).permute(2, 0, 1).float() / 255.0
        new_corners = corners_np * ratio + np.array([dw, dh], dtype=np.float32)

    return img_tensor, new_corners.astype(np.float32)


# ---------------------------------------------------------------------------
# Main trainer class
# ---------------------------------------------------------------------------

class AdversarialPatchTrainer:
    def __init__(
        self,
        detector: DetectorBackend,
        ocr: OCRBackend,
        csv_path: str = "preproc_labels.csv",
        preload_images: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
        limit: int = 0,
        use_all_for_train: bool = False,
        grad_accumulate: Optional[int] = None,
        impersonation_target: Optional[str] = None,
        expected_plate_text: str = "VRJ7774",
        print_blur: float = 0.0,
        training: bool = False,
        use_tv_loss: bool = True,
        use_homography: bool = True,
        run_name: Optional[str] = None,
    ):
        self.training             = training
        self.print_blur           = print_blur
        self.use_tv_loss          = use_tv_loss
        self.use_homography       = use_homography
        self.grad_accumulate      = grad_accumulate
        self.impersonation_target = impersonation_target
        self.expected_plate_text  = expected_plate_text

        # ── Detector ───────────────────────────────────────────────────
        self.detector = detector
        self.device   = detector.device
        self.detector.eval()
        self.detector.freeze()

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

        # ── Patch parameter ────────────────────────────────────────────
        self.patch_width  = PATCH_WIDTH
        self.patch_height = PATCH_HEIGHT
        self.patch = nn.Parameter(
            torch.randn(3, self.patch_height, self.patch_width,
                        device=self.device) * 0.1
        )

        # ── Transform used for image loading ───────────────────────────
        self.transform = T.Compose([T.ToTensor()])

        # ── DataLoaders (use_original=True: full-res HEIC, no letterbox) ─
        self.train_loader, self.val_loader = create_dataloaders(
            csv_path,
            transform=self.transform,
            preload=preload_images,
            batch_size=1,
            n_jobs=num_workers,
            pin_memory=pin_memory,
            limit=limit,
            use_all_for_train=use_all_for_train,
            use_original=True,
        )

        # ── Preprocessing cache ────────────────────────────────────────
        # Precompute detector-specific preprocessing for every image once.
        # Maps filename → (prep_tensor [C,H,W] float, new_corners [4,2] float32)
        self._prep_cache: Dict[str, Tuple[torch.Tensor, np.ndarray]] = {}
        self._build_prep_cache()

        self.epoch_stats: list = []

    # ====================================================================
    # Preprocessing cache
    # ====================================================================

    def _build_prep_cache(self) -> None:
        loaders = [("train", self.train_loader), ("val", self.val_loader)]
        total   = sum(len(ld) for _, ld in loaders)
        with tqdm(total=total, desc="Preprocessing images for detector") as pbar:
            for _, loader in loaders:
                for batch in loader:
                    fn      = batch["filename"][0]
                    if fn in self._prep_cache:
                        pbar.update(1)
                        continue
                    # orig_image is CHW float [0,1] — convert back to HWC uint8
                    orig_chw = batch["orig_image"][0]   # [3, H, W]
                    img_hwc  = (orig_chw.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    corners  = batch["orig_corners"][0].numpy()   # [4, 2]
                    prep_t, new_c = preprocess_for_detector(img_hwc, corners, self.detector)
                    self._prep_cache[fn] = (prep_t, new_c)
                    pbar.update(1)
        print(f"  Cached preprocessed tensors: {len(self._prep_cache)} images")

    # ====================================================================
    # Helpers
    # ====================================================================

    def bbox_to_corners(self, bbox: torch.Tensor,
                         device=None) -> torch.Tensor:
        x1, y1, x2, y2 = bbox
        return torch.tensor([[
            [x1, y1], [x2, y1], [x2, y2], [x1, y2]
        ]], device=device or self.device)

    def corners_to_bbox(self, corners: torch.Tensor) -> torch.Tensor:
        return torch.stack([
            corners[:, 0].min(), corners[:, 1].min(),
            corners[:, 0].max(), corners[:, 1].max(),
        ])

    def get_patch_bounding_box(self, corners: torch.Tensor,
                                border_scale: float = 1.4) -> torch.Tensor:
        cx     = corners[:, 0].mean()
        cy     = corners[:, 1].mean()
        center = torch.tensor([cx, cy], device=self.device)
        border = center.unsqueeze(0) + (corners - center.unsqueeze(0)) * border_scale
        return torch.stack([border[:, 0].min(), border[:, 1].min(),
                            border[:, 0].max(), border[:, 1].max()])

    @staticmethod
    def _boxes_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
        b1    = boxes1.unsqueeze(1)
        b2    = boxes2.unsqueeze(0)
        iw    = torch.clamp(torch.min(b1[..., 2], b2[..., 2]) -
                            torch.max(b1[..., 0], b2[..., 0]), min=0)
        ih    = torch.clamp(torch.min(b1[..., 3], b2[..., 3]) -
                            torch.max(b1[..., 1], b2[..., 1]), min=0)
        inter = iw * ih
        union = area1 + area2 - inter
        return inter / (union + 1e-8)

    # ====================================================================
    # Patch application
    # ====================================================================

    def _apply_patch_simple(self, image: torch.Tensor, corners: torch.Tensor,
                             patch_norm: torch.Tensor,
                             border_scale: float = 1.4) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = image.shape
        plate = corners[0]
        cx    = plate[:, 0].mean()
        cy    = plate[:, 1].mean()
        ctr   = torch.tensor([cx, cy], device=self.device)
        border = ctr.unsqueeze(0) + (plate - ctr.unsqueeze(0)) * border_scale

        bx1 = torch.clamp(border[:, 0].min(), 0, W).int()
        bx2 = torch.clamp(border[:, 0].max(), 0, W).int()
        by1 = torch.clamp(border[:, 1].min(), 0, H).int()
        by2 = torch.clamp(border[:, 1].max(), 0, H).int()
        px1 = torch.clamp(plate[:, 0].min(), 0, W).int()
        px2 = torch.clamp(plate[:, 0].max(), 0, W).int()
        py1 = torch.clamp(plate[:, 1].min(), 0, H).int()
        py2 = torch.clamp(plate[:, 1].max(), 0, H).int()

        result = image.clone()
        mask   = torch.zeros(B, 3, H, W, device=self.device)
        bh, bw = by2 - by1, bx2 - bx1
        if bh > 0 and bw > 0:
            resized = F.interpolate(patch_norm.unsqueeze(0), size=(bh, bw),
                                    mode="bilinear", align_corners=True)
            for b in range(B):
                result[b, :, by1:by2, bx1:bx2] = resized[0]
                mask[b,   :, by1:by2, bx1:bx2] = 1.0
                if py2 > py1 and px2 > px1:
                    result[b, :, py1:py2, px1:px2] = image[b, :, py1:py2, px1:px2]
                    mask[b,   :, py1:py2, px1:px2] = 0.0
        return torch.clamp(result, 0, 1), mask

    def apply_patch_to_image(self, image: torch.Tensor, corners: torch.Tensor,
                              border_scale: float = 1.4) -> Tuple[torch.Tensor, torch.Tensor]:
        B    = image.shape[0]
        H, W = image.shape[2], image.shape[3]

        patch_norm = torch.tanh(self.patch) * 0.5 + 0.5

        if self.print_blur > 0:
            patch_norm = kornia.filters.gaussian_blur2d(
                patch_norm.unsqueeze(0), (3, 3),
                (self.print_blur, self.print_blur),
            ).squeeze(0)

        if self.training:
            factor     = torch.rand(1, device=self.device) * 0.2
            patch_norm = patch_norm * (1.0 - factor)

        if not self.use_homography:
            return self._apply_patch_simple(image, corners, patch_norm, border_scale)

        plate  = corners[0]
        cx     = plate[:, 0].mean()
        cy     = plate[:, 1].mean()
        center = torch.tensor([cx, cy], device=self.device)
        border = (center.unsqueeze(0) +
                  (plate - center.unsqueeze(0)) * border_scale).unsqueeze(0)

        ph, pw = self.patch_height, self.patch_width
        src    = torch.tensor([[0, 0], [pw, 0], [pw, ph], [0, ph]],
                               dtype=torch.float32, device=self.device).unsqueeze(0)

        M_border = K.get_perspective_transform(src, border)
        M_plate  = K.get_perspective_transform(src, corners)

        patch_batch = patch_norm.unsqueeze(0).repeat(B, 1, 1, 1)
        ones        = torch.ones(B, 1, ph, pw, device=self.device)

        warped  = K.warp_perspective(patch_batch, M_border, (H, W),
                                     mode="bilinear", padding_mode="zeros",
                                     align_corners=True)
        w_bord  = K.warp_perspective(ones,        M_border, (H, W),
                                     mode="bilinear", padding_mode="zeros",
                                     align_corners=True)
        w_plate = K.warp_perspective(ones,        M_plate,  (H, W),
                                     mode="bilinear", padding_mode="zeros",
                                     align_corners=True)

        mask   = torch.clamp(w_bord - w_plate, 0, 1).expand(-1, 3, -1, -1)
        result = image * (1 - mask) + warped * mask
        return torch.clamp(result, 0, 1), mask

    # ====================================================================
    # Loss
    # ====================================================================

    def patch_reg_loss(self) -> torch.Tensor:
        p    = self.patch
        C, H, W = p.shape
        tv_h = torch.pow(p[:, :, 1:] - p[:, :, :-1], 2).sum()
        tv_v = torch.pow(p[:, 1:, :] - p[:, :-1, :], 2).sum()
        n    = C * (H * (W - 1) + (H - 1) * W)
        return (tv_h + tv_v) / n * 2.5

    def partial_loss(
        self,
        patched_prep: torch.Tensor,   # [1, C, H, W] — patched, preprocessed for detector
        new_corners: torch.Tensor,    # [4, 2] — corners in preprocessed-image space
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Core loss: detection suppression + OCR disruption / impersonation.

        target_box  = bounding box of plate corners directly.
        best_det    = detection with maximum IoU × confidence against target_box.
        det_loss    = IoU(best_det, target_box) × confidence  (minimise → suppress).
        ocr_loss    = differentiable CE/CTC loss on crop from patched preprocessed image.
        """
        target_box = self.corners_to_bbox(new_corners)   # [4]

        # ── Detection ──────────────────────────────────────────────────
        detections = self.detector.predict(patched_prep.squeeze(0))

        best_det: Optional[Detection] = None
        det_loss = torch.tensor(0.0, device=self.device)

        if detections:
            # Pick the detection with highest IoU×conf against the plate
            best_det = max(
                detections,
                key=lambda d: (
                    self._boxes_iou(
                        d.box.to(self.device).unsqueeze(0),
                        target_box.unsqueeze(0),
                    ).item() * d.confidence
                ),
            )
            iou_val  = self._boxes_iou(best_det.box.to(self.device).unsqueeze(0),
                                        target_box.unsqueeze(0))
            conf_val = best_det.conf.to(self.device)
            det_loss = (iou_val * conf_val).squeeze()

        # ── OCR ────────────────────────────────────────────────────────
        target_text = self.impersonation_target or self.expected_plate_text
        ocr_loss    = torch.tensor(0.0, device=self.device)

        # Crop from patched preprocessed image using plate corners
        crop_corners = new_corners.unsqueeze(0).to(self.device)   # [1, 4, 2]
        crop = kornia.geometry.crop_and_resize(
            patched_prep,                # [1, C, H, W]
            crop_corners,
            self.ocr.ocr_crop_size,      # (H, W)
            mode="bilinear", align_corners=True,
        )   # [1, 3, H_ocr, W_ocr]

        if self.ocr.is_trainable and hasattr(self.ocr, "differentiable_loss"):
            ocr_loss = self.ocr.differentiable_loss(
                crop, target_text,
                impersonation=bool(self.impersonation_target),
            )
        else:
            # Non-differentiable path: character-accuracy heuristic
            with torch.no_grad():
                ocr_result = self.ocr.predict(crop.squeeze(0))
            accuracy = ocr_result.char_accuracy(target_text)
            ocr_loss = torch.tensor(
                (1.0 - accuracy) if self.impersonation_target else accuracy,
                device=self.device,
            )

        return det_loss, ocr_loss

    def compute_loss(self, batch: dict) -> torch.Tensor:
        filename = (batch["filename"][0]
                    if isinstance(batch["filename"], (list, tuple))
                    else batch["filename"])

        prep_tensor, new_corners_np = self._prep_cache[filename]
        prep_tensor  = prep_tensor.to(self.device)
        new_corners  = torch.from_numpy(new_corners_np).to(self.device)   # [4, 2]

        # Apply patch to preprocessed image
        patched_prep, _ = self.apply_patch_to_image(
            prep_tensor.unsqueeze(0),   # [1, C, H, W]
            new_corners.unsqueeze(0),   # [1, 4, 2]
        )   # → [1, C, H, W]

        det_loss, ocr_loss = self.partial_loss(patched_prep, new_corners)
        reg_loss = self.patch_reg_loss() if self.use_tv_loss else 0.0
        return (det_loss + ocr_loss) / 2 + reg_loss

    # ====================================================================
    # Pre-training sanity check
    # ====================================================================

    def validate_pipeline(self, n_samples: Optional[int] = None) -> dict:
        """
        Run clean images (no patch) through detector + OCR and categorise results.

        Categories
        ----------
        correct     : OCR text matches expected_plate_text exactly
        impersonation: OCR reads something, but not the expected text
        misread     : detector found the plate but OCR returned nothing
        no_detection: detector found nothing

        Raises RuntimeError if correct reads < 50%.
        """
        print("\n── Pre-training sanity check ──────────────────────────────")
        results = []
        items   = list(self._prep_cache.items())
        if n_samples is not None:
            items = items[:n_samples]

        for fn, (prep_t, new_c) in tqdm(items, desc="Sanity check", leave=False):
            prep_tensor   = prep_t.to(self.device)
            new_corners   = torch.from_numpy(new_c).to(self.device)
            target_box    = self.corners_to_bbox(new_corners)

            with torch.no_grad():
                detections = self.detector.predict(prep_tensor)

            if not detections:
                results.append({"filename": fn, "category": "no_detection",
                                 "text": None, "confidence": 0.0})
                continue

            best_det = max(
                detections,
                key=lambda d: (
                    self._boxes_iou(
                        d.box.to(self.device).unsqueeze(0),
                        target_box.unsqueeze(0),
                    ).item() * d.confidence
                ),
            )

            # Crop using plate corners
            crop_corners = new_corners.unsqueeze(0)
            with torch.no_grad():
                crop = kornia.geometry.crop_and_resize(
                    prep_tensor.unsqueeze(0),
                    crop_corners,
                    self.ocr.ocr_crop_size,
                    mode="bilinear", align_corners=True,
                )
                ocr_result = self.ocr.predict(crop.squeeze(0))

            text = ocr_result.text
            conf = ocr_result.confidence

            if text is None or text == "":
                cat = "misread"
            elif text.upper() == self.expected_plate_text.upper():
                cat = "correct"
            else:
                cat = "impersonation"

            results.append({"filename": fn, "category": cat,
                             "text": text, "confidence": conf})

        # Tally
        counts = {"correct": 0, "impersonation": 0, "misread": 0, "no_detection": 0}
        for r in results:
            counts[r["category"]] += 1
        total = max(len(results), 1)

        lines = [
            f"Sanity check over {len(results)} images  "
            f"(expected plate: '{self.expected_plate_text}')",
            f"  Correct reads : {counts['correct']:4d}  ({counts['correct']/total*100:.1f}%)",
            f"  Impersonation : {counts['impersonation']:4d}  ({counts['impersonation']/total*100:.1f}%)",
            f"  Misread       : {counts['misread']:4d}  ({counts['misread']/total*100:.1f}%)",
            f"  No detection  : {counts['no_detection']:4d}  ({counts['no_detection']/total*100:.1f}%)",
        ]
        report = "\n".join(lines)
        print(report)

        sanity_path = self.run_dir / "sanity_check.txt"
        sanity_path.write_text(report + "\n")
        print(f"  Saved → {sanity_path}")

        correct_frac = counts["correct"] / total
        if correct_frac < 0.50:
            raise RuntimeError(
                f"\nSanity check FAILED: only {counts['correct']}/{total} images "
                f"({correct_frac*100:.1f}%) read correctly.\n"
                "Possible causes:\n"
                "  • Wrong CSV (check --csv points to preproc_labels.csv)\n"
                "  • Wrong expected_plate_text (check --expected-plate)\n"
                "  • Detector / OCR model not loading correctly\n"
                "  • Preprocessing mismatch between backend and image\n"
                f"Full results → {sanity_path}"
            )

        print(f"  Sanity check passed ({correct_frac*100:.1f}% correct).\n")
        return counts

    # ====================================================================
    # Debug image output
    # ====================================================================

    def save_debug_images(self, n: int = 20) -> None:
        """
        Save n annotated debug images to run_dir/debug/.

        For each sample i:
          {i:02d}_a_preprocessed_detection.png  — preprocessed image + corners + OCR label
          {i:02d}_b_ocr_crop.png                — raw OCR crop (at ocr_crop_size)
          {i:02d}_c_random_patch.png            — preprocessed image + random patch applied
        Also saves debug_summary.csv.
        """
        debug_dir  = self.run_dir / "debug"
        items      = list(self._prep_cache.items())
        np.random.seed(42)
        indices    = np.random.choice(len(items), min(n, len(items)), replace=False)
        sample_items = [items[i] for i in sorted(indices)]

        summary_rows = []

        print(f"  Saving {len(sample_items)} debug images → {debug_dir}")
        for img_idx, (fn, (prep_t, new_c)) in enumerate(sample_items):
            prep_tensor  = prep_t.to(self.device)
            new_corners  = torch.from_numpy(new_c).to(self.device)

            with torch.no_grad():
                # ── Detection ──────────────────────────────────────────
                detections   = self.detector.predict(prep_tensor)
                target_box   = self.corners_to_bbox(new_corners)
                best_det     = (max(
                    detections,
                    key=lambda d: (
                        self._boxes_iou(
                            d.box.to(self.device).unsqueeze(0),
                            target_box.unsqueeze(0),
                        ).item() * d.confidence
                    ),
                ) if detections else None)

                # ── OCR ────────────────────────────────────────────────
                crop_corners = new_corners.unsqueeze(0)
                crop = kornia.geometry.crop_and_resize(
                    prep_tensor.unsqueeze(0),
                    crop_corners,
                    self.ocr.ocr_crop_size,
                    mode="bilinear", align_corners=True,
                )
                ocr_result = self.ocr.predict(crop.squeeze(0))

            text = ocr_result.text or ""
            conf = ocr_result.confidence
            is_correct = text.upper() == self.expected_plate_text.upper()

            # Convert preprocessed tensor to BGR uint8 for cv2
            vis = (prep_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8).copy()

            # ── (a) Preprocessed image with corners + OCR label ────────
            pts = new_c.astype(np.int32)
            cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            label = f"{text} ({conf:.2f})"
            cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)
            cv2.imwrite(str(debug_dir / f"{img_idx:02d}_a_preprocessed_detection.png"), vis)

            # ── (b) OCR crop ───────────────────────────────────────────
            crop_np = (crop.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            cv2.imwrite(str(debug_dir / f"{img_idx:02d}_b_ocr_crop.png"), crop_np)

            # ── (c) Random patch applied ───────────────────────────────
            rand_raw = torch.randn_like(self.patch)   # in patch parameter space
            old_data = self.patch.data.clone()
            self.patch.data.copy_(rand_raw)
            with torch.no_grad():
                patched, _ = self.apply_patch_to_image(
                    prep_tensor.unsqueeze(0),
                    new_corners.unsqueeze(0),
                )
            self.patch.data.copy_(old_data)
            patch_vis = (patched.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            cv2.imwrite(str(debug_dir / f"{img_idx:02d}_c_random_patch.png"), patch_vis)

            summary_rows.append({
                "index":    img_idx,
                "filename": fn,
                "detected_text":  text,
                "confidence":     f"{conf:.4f}",
                "correct":        is_correct,
            })

        # Save summary CSV
        csv_path = debug_dir / "debug_summary.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["index", "filename", "detected_text", "confidence", "correct"]
            )
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"  Debug summary → {csv_path}")

    # ====================================================================
    # Patch persistence
    # ====================================================================

    def save_patch(self, epoch: int, subdir: str = "patches") -> None:
        save_dir = self.run_dir / subdir
        save_dir.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            img = torch.tanh(self.patch) * 0.5 + 0.5
            T.ToPILImage()(img.cpu()).save(
                str(save_dir / f"patch_{self.detector.name}_epoch_{epoch:04d}.png"))
            torch.save({
                "patch":      self.patch.detach().cpu(),
                "epoch":      epoch,
                "backend":    self.detector.name,
                "ocr":        self.ocr.name,
                "patch_size": (self.patch_height, self.patch_width),
            }, str(save_dir / f"patch_{self.detector.name}_epoch_{epoch:04d}.pt"))

    # ====================================================================
    # Training loop
    # ====================================================================

    def train_epoch(self, optimizer, epoch: int) -> float:
        if self.ocr.is_trainable:
            self.ocr.train()

        update_every = (len(self.train_loader)
                        if self.grad_accumulate is None
                        else self.grad_accumulate)
        total_loss = accum_loss = 0.0
        step = num_updates = 0

        with tqdm(enumerate(self.train_loader),
                  desc=f"Epoch {epoch+1} [{self.detector.name}/{self.ocr.name}]",
                  total=len(self.train_loader), leave=False) as pbar:
            for idx, batch in pbar:
                loss        = self.compute_loss(batch)
                scaled_loss = loss / update_every
                scaled_loss.backward()

                accum_loss += loss.item()
                step       += 1

                if step % update_every == 0:
                    torch.nn.utils.clip_grad_norm_([self.patch], max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    total_loss  += accum_loss
                    num_updates += 1
                    accum_loss   = 0.0
                    del loss, scaled_loss
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                    elif self.device == "mps":
                        torch.mps.empty_cache()
                    avg = total_loss / (num_updates * update_every)
                    pbar.set_postfix({"loss": f"{avg:.4f}"})
                else:
                    del loss, scaled_loss

            # Flush remaining gradients
            if step % update_every != 0 and self.grad_accumulate is not None:
                torch.nn.utils.clip_grad_norm_([self.patch], max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                total_loss  += accum_loss
                num_updates += 1

        total_steps = (num_updates * update_every
                       if self.grad_accumulate else len(self.train_loader))
        return total_loss / max(total_steps, 1)

    def validate(self) -> float:
        if self.ocr.is_trainable:
            self.ocr.eval()
        losses = []
        with torch.no_grad():
            for batch in self.val_loader:
                losses.append(self.compute_loss(batch).item())
        return float(np.mean(losses)) if losses else 0.0

    def train(
        self,
        num_epochs:    int   = 150,
        learning_rate: float = 0.01,
        save_interval: int   = 10,
        dry_run:       bool  = False,
    ) -> dict:
        # ── Pre-training checks ────────────────────────────────────────
        self.validate_pipeline()
        self.save_debug_images()

        if dry_run:
            print("\nDry run complete. Exiting before training.")
            return {}

        # ── Optimiser + scheduler ──────────────────────────────────────
        optimizer = optim.AdamW([self.patch], lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=1e-4
        )

        history   = {"loss": [], "val_score": [], "learning_rate": []}
        best_loss = float("inf")
        best_epoch = -1

        print(f"\n{'='*60}")
        print(f"  Adversarial Patch Training")
        print(f"  Detector : {self.detector.name}   OCR : {self.ocr.name}")
        print(f"{'='*60}")
        print(f"  Dataset  : {len(self.train_loader)+len(self.val_loader)} images")
        print(f"  Patch    : {self.patch_height}×{self.patch_width}")
        print(f"  Device   : {self.device}")
        print(f"  Epochs   : {num_epochs}  |  LR: {learning_rate}")
        print(f"  Mode     : "
              f"{'impersonation → ' + self.impersonation_target if self.impersonation_target else 'disruption'}")
        print(f"  Run dir  : {self.run_dir}")
        print(f"{'='*60}\n")

        log_path = self.run_dir / "training_log.txt"
        log_file = open(log_path, "w")

        for epoch in range(num_epochs):
            self.training = True
            train_loss    = self.train_epoch(optimizer, epoch)
            self.training = False
            val_loss      = self.validate()
            scheduler.step()
            lr = optimizer.param_groups[0]["lr"]

            history["loss"].append(train_loss)
            history["val_score"].append(val_loss)
            history["learning_rate"].append(lr)

            init_val = history["val_score"][0]
            change   = (val_loss / (init_val + 1e-9) - 1) * 100

            best_marker = ""
            if val_loss < best_loss:
                best_loss  = val_loss
                best_epoch = epoch
                self.save_patch(epoch, "patches")   # overwrite "best" checkpoint
                best_marker = "  ★ best"

            line = (f"Epoch {epoch+1:3d}/{num_epochs} | "
                    f"Loss: {train_loss:.4f} | Val: {val_loss:.4f} | "
                    f"Δ: {change:+.1f}% | LR: {lr:.2e}{best_marker}")
            print(line)
            log_file.write(line + "\n")
            log_file.flush()

            # Periodic checkpoint every save_interval epochs
            if (epoch + 1) % save_interval == 0:
                self.save_patch(epoch, "patches")

        log_file.close()
        print(f"\nBest val loss: {best_loss:.4f} at epoch {best_epoch+1}")

        # Save training history CSV
        import pandas as pd
        hist_path = self.run_dir / "training_history.csv"
        pd.DataFrame(history).assign(
            epoch=range(1, len(history["loss"]) + 1)
        ).to_csv(str(hist_path), index=False)
        print(f"\nDone. Best val loss: {best_loss:.4f}")
        print(f"Training history → {hist_path}")
        return history


# ====================================================================
# CLI entry-point
# ====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Adversarial patch trainer — pluggable detector + OCR backend"
    )
    parser.add_argument("--csv", default="preproc_labels.csv",
                        help="CSV with full-res image paths and plate corners.")

    TRAINABLE_DET = ["yolov8", "fasterrcnn", "yolov11", "rtdetr", "yolo-v9-384"]
    TRAINABLE_OCR = ["crnn", "trocr", "dtrb", "lprnet", "cct", "fastanpr-ocr"]

    parser.add_argument("--backend", default="yolov8", choices=TRAINABLE_DET)
    parser.add_argument("--model-path", default="license_plate_detector.pt",
                        help="Path to detector weights (.pt).  "
                             "For yolo-v9-384 and rtdetr pass 'none' (auto-download).")
    parser.add_argument("--ocr-backend", default="crnn", choices=TRAINABLE_OCR)
    parser.add_argument("--ocr-model-path", default="none",
                        help="Path to OCR weights.  Pass 'none' for auto-download / HF.")
    parser.add_argument("--ocr-repo-root", default=None,
                        help="DTRB repo root (required for --ocr-backend dtrb).")
    parser.add_argument("--dtrb-feature-extraction", default="vitstr_small_patch16_224")
    parser.add_argument("--dtrb-sequence-modeling",   default="None")
    parser.add_argument("--dtrb-transformation",      default="None")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--grad-accumulate", type=int, default=64)
    parser.add_argument("--preload-images", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--use-all-for-train", action="store_true")
    parser.add_argument("--impersonation-target", default=None)
    parser.add_argument("--expected-plate", default="VRJ7774",
                        help="True plate text — used for sanity check and OCR loss target.")
    parser.add_argument("--disable-tv-loss", action="store_true")
    parser.add_argument("--disable-homography", action="store_true")
    parser.add_argument("--run-name", default=None,
                        help="Override the timestamp suffix in the run directory name.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run sanity check + debug images only, then exit.")
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

    if not ocr.is_trainable:
        print(
            f"Note: '{args.ocr_backend}' OCR has no gradient graph — "
            "OCR loss uses character-accuracy heuristic only.\n"
            "Use --ocr-backend crnn or cct for differentiable training."
        )

    trainer = AdversarialPatchTrainer(
        detector             = backend,
        ocr                  = ocr,
        csv_path             = args.csv,
        preload_images       = args.preload_images,
        num_workers          = args.num_workers,
        pin_memory           = args.pin_memory,
        limit                = args.limit,
        use_all_for_train    = args.use_all_for_train,
        grad_accumulate      = args.grad_accumulate,
        impersonation_target = args.impersonation_target,
        expected_plate_text  = args.expected_plate,
        training             = True,
        use_tv_loss          = not args.disable_tv_loss,
        use_homography       = not args.disable_homography,
        run_name             = args.run_name,
    )

    trainer.train(
        num_epochs    = args.epochs,
        learning_rate = args.lr,
        save_interval = 10,
        dry_run       = args.dry_run,
    )


if __name__ == "__main__":
    main()
