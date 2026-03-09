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
* CosineAnnealingLR, 150 epochs, no early stopping.
* validate_pipeline() sanity-checks before training.
* save_debug_images() writes 20 annotated images to run_dir/debug/.
"""

from __future__ import annotations

import csv
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
from dataset import create_dataloaders, make_letterbox_prep, make_resize_prep, make_passthrough_prep

warnings.filterwarnings("ignore")

PATCH_WIDTH  = 512
PATCH_HEIGHT = 256




# ---------------------------------------------------------------------------
# Patch decoder
# ---------------------------------------------------------------------------

class PatchDecoder(nn.Module):
    """
    Trainable ConvTranspose2d decoder that maps a compact seed tensor
    [1, seed_channels, 4, 8] → [1, 3, 256, 512].

    Six stride-2 layers double the spatial dimensions at each step:
        4×8 → 8×16 → 16×32 → 32×64 → 64×128 → 128×256 → 256×512

    Channel schedule halves from seed_channels→256→128→64→32→16→3,
    so the expensive (many-channel) work is done at small spatial grids.
    Output is passed through tanh and scaled to [0, 1].

    Both the seed and the decoder weights are optimised during training —
    the seed controls global structure while the decoder weights determine
    how that structure is elaborated into pixel-level detail.
    """

    def __init__(self, seed_channels: int = 128):
        super().__init__()
        c = seed_channels
        self.net = nn.Sequential(
            nn.ConvTranspose2d(c,   256, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128,  64, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d( 64,  32, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d( 32,  16, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d( 16,   3, 4, stride=2, padding=1),
        )

    def forward(self, seed: torch.Tensor) -> torch.Tensor:
        """seed: [1, C, 4, 8]  →  patch: [1, 3, 256, 512] in [0, 1]"""
        return torch.tanh(self.net(seed)) * 0.5 + 0.5


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
        num_workers:          int            = 0,
        pin_memory:           bool           = False,
        limit:                int            = 0,
        use_all_for_train:    bool           = False,
        grad_accumulate:      Optional[int]  = None,
        impersonation_target: Optional[str]  = None,
        expected_plate_text:  str            = "VRJ7774",
        print_blur:           float          = 0.0,
        training:             bool           = False,
        use_homography:       bool           = True,
        run_name:             Optional[str]  = None,
    ):
        self.training             = training
        self.print_blur           = print_blur
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

        # ── DataLoaders ────────────────────────────────────────────────
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
            prep_fn=self._make_prep_fn(),
        )

        self.epoch_stats: list = []

    # ====================================================================
    # Patch generation
    # ====================================================================

    def _trainable_params(self):
        """All parameters that the optimiser should update."""
        return [self.seed] + list(self.decoder.parameters())

    def generate_patch(self, training_aug: bool = False) -> torch.Tensor:
        """
        Run the decoder forward to produce the patch.

        Parameters
        ----------
        training_aug : bool
            When True, applies the random brightness jitter used during
            training to simulate print variation.

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

        if training_aug:
            factor = torch.rand(1, device=self.device) * 0.2
            patch  = patch * (1.0 - factor)

        return patch   # [3, 256, 512]

    # ====================================================================
    # Detector preprocessing selection
    # ====================================================================

    def _make_prep_fn(self):
        """Return the dataset preprocessing function for the current detector."""
        name = self.detector.name
        if name in ("yolov8", "yolov11"):
            imgsz = 640
            if hasattr(self.detector, "_yolo") and self.detector._yolo is not None:
                raw = self.detector._yolo.imgsz
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

    # ====================================================================
    # Patch application
    # ====================================================================

    def _apply_patch_simple(self, image: torch.Tensor, corners: torch.Tensor,
                             patch_norm: torch.Tensor,
                             border_scale: float = 1.4) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = image.shape
        plate = corners[0]
        cx, cy = plate[:, 0].mean(), plate[:, 1].mean()
        ctr    = torch.tensor([cx, cy], device=self.device)
        border = ctr.unsqueeze(0) + (plate - ctr.unsqueeze(0)) * border_scale

        bx1 = torch.clamp(border[:, 0].min(), 0, W).int()
        bx2 = torch.clamp(border[:, 0].max(), 0, W).int()
        by1 = torch.clamp(border[:, 1].min(), 0, H).int()
        by2 = torch.clamp(border[:, 1].max(), 0, H).int()
        px1 = torch.clamp(plate[:, 0].min(),  0, W).int()
        px2 = torch.clamp(plate[:, 0].max(),  0, W).int()
        py1 = torch.clamp(plate[:, 1].min(),  0, H).int()
        py2 = torch.clamp(plate[:, 1].max(),  0, H).int()

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

    def apply_patch_to_image(
        self,
        image:          torch.Tensor,
        corners:        torch.Tensor,
        patch_norm:     Optional[torch.Tensor] = None,
        border_scale:   float = 1.4,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Warp the patch onto the image using the plate corners.

        Parameters
        ----------
        patch_norm : torch.Tensor or None
            Pre-generated [3, H, W] patch in [0, 1].  If None, calls
            generate_patch(training_aug=self.training) internally.
        """
        if patch_norm is None:
            patch_norm = self.generate_patch(training_aug=self.training)

        B    = image.shape[0]
        H, W = image.shape[2], image.shape[3]

        if not self.use_homography:
            return self._apply_patch_simple(image, corners, patch_norm, border_scale)

        plate  = corners[0]
        cx, cy = plate[:, 0].mean(), plate[:, 1].mean()
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
        w_bord  = K.warp_perspective(ones, M_border, (H, W),
                                     mode="bilinear", padding_mode="zeros",
                                     align_corners=True)
        w_plate = K.warp_perspective(ones, M_plate, (H, W),
                                     mode="bilinear", padding_mode="zeros",
                                     align_corners=True)

        mask   = torch.clamp(w_bord - w_plate, 0, 1).expand(-1, 3, -1, -1)
        result = image * (1 - mask) + warped * mask
        return torch.clamp(result, 0, 1), mask

    # ====================================================================
    # Loss
    # ====================================================================

    def partial_loss(
        self,
        patched_prep: torch.Tensor,   # [1, C, H, W]
        new_corners:  torch.Tensor,   # [4, 2]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        target_box = self.corners_to_bbox(new_corners)

        # ── Detection ──────────────────────────────────────────────────
        detections = self.detector.predict(patched_prep.squeeze(0))
        det_loss   = torch.tensor(0.0, device=self.device)

        if detections:
            best_det = max(
                detections,
                key=lambda d: (
                    self._boxes_iou(d.box.to(self.device).unsqueeze(0),
                                    target_box.unsqueeze(0)).item()
                    * d.confidence
                ),
            )
            iou_val  = self._boxes_iou(best_det.box.to(self.device).unsqueeze(0),
                                        target_box.unsqueeze(0))
            det_loss = (iou_val * best_det.conf.to(self.device)).squeeze()

        # ── OCR ────────────────────────────────────────────────────────
        target_text = self.impersonation_target or self.expected_plate_text
        ocr_loss    = torch.tensor(0.0, device=self.device)

        crop = kornia.geometry.crop_and_resize(
            patched_prep,
            new_corners.unsqueeze(0).to(self.device),
            self.ocr.ocr_crop_size,
            mode="bilinear", align_corners=True,
        )   # [1, 3, H_ocr, W_ocr]

        if self.ocr.is_trainable and hasattr(self.ocr, "differentiable_loss"):
            ocr_loss = self.ocr.differentiable_loss(
                crop, target_text, impersonation=bool(self.impersonation_target),
            )
        else:
            with torch.no_grad():
                ocr_result = self.ocr.predict(crop.squeeze(0))
            accuracy = ocr_result.char_accuracy(target_text)
            ocr_loss = torch.tensor(
                (1.0 - accuracy) if self.impersonation_target else accuracy,
                device=self.device,
            )

        return det_loss, ocr_loss

    def compute_loss(self, batch: dict) -> torch.Tensor:
        fn = (batch["filename"][0]
              if isinstance(batch["filename"], (list, tuple))
              else batch["filename"])

        prep_tensor = batch["prep_image"][0].to(self.device)
        new_corners = batch["new_corners"][0].to(self.device)

        # Generate patch once — shared between apply and loss
        patch_norm = self.generate_patch(training_aug=self.training)

        patched_prep, _ = self.apply_patch_to_image(
            prep_tensor.unsqueeze(0),
            new_corners.unsqueeze(0),
            patch_norm=patch_norm,
        )

        det_loss, ocr_loss = self.partial_loss(patched_prep, new_corners)
        return (det_loss + ocr_loss) / 2

    # ====================================================================
    # Patch persistence
    # ====================================================================

    def save_patch(self, epoch: int, subdir: str = "patches") -> None:
        save_dir = self.run_dir / subdir
        save_dir.mkdir(parents=True, exist_ok=True)
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
            }, str(save_dir / f"{stem}.pt"))

    # ====================================================================
    # Pre-training sanity check
    # ====================================================================

    def validate_pipeline(self, n_samples: Optional[int] = None) -> dict:
        """
        Run clean images (no patch) through detector + OCR.
        Raises RuntimeError if < 50% read correctly.
        """
        print("\n── Pre-training sanity check ──────────────────────────────")
        total = (min(n_samples, len(self.train_loader) + len(self.val_loader))
                 if n_samples is not None
                 else len(self.train_loader) + len(self.val_loader))
        results = []
        count = 0
        done = False
        with tqdm(total=total, desc="Sanity check", leave=False) as pbar:
            for loader in [self.train_loader, self.val_loader]:
                if done:
                    break
                for batch in loader:
                    if n_samples is not None and count >= n_samples:
                        done = True
                        break
                    fn = (batch["filename"][0]
                          if isinstance(batch["filename"], (list, tuple))
                          else batch["filename"])
                    prep_tensor = batch["prep_image"][0].to(self.device)
                    new_corners = batch["new_corners"][0].to(self.device)
                    count += 1
                    pbar.update(1)
                    target_box  = self.corners_to_bbox(new_corners)

                    with torch.no_grad():
                        detections = self.detector.predict(prep_tensor)

                    if not detections:
                        results.append({"filename": fn, "category": "no_detection",
                                        "text": None, "confidence": 0.0})
                        continue

                    best_det = max(
                        detections,
                        key=lambda d: (
                            self._boxes_iou(d.box.to(self.device).unsqueeze(0),
                                            target_box.unsqueeze(0)).item()
                            * d.confidence
                        ),
                    )

                    with torch.no_grad():
                        crop = kornia.geometry.crop_and_resize(
                            prep_tensor.unsqueeze(0),
                            new_corners.unsqueeze(0),
                            self.ocr.ocr_crop_size,
                            mode="bilinear", align_corners=True,
                        )
                        ocr_result = self.ocr.predict(crop.squeeze(0))

                    text = ocr_result.text or ""
                    if text.upper() == self.expected_plate_text.upper():
                        cat = "correct"
                    elif self.impersonation_target and text.upper() == self.impersonation_target.upper():
                        cat = "impersonation"
                    else:
                        cat = "misread"
                    results.append({"filename": fn, "category": cat,
                                    "text": text, "confidence": ocr_result.confidence})

        counts = {"correct": 0, "impersonation": 0, "misread": 0, "no_detection": 0}
        for r in results:
            counts[r["category"]] += 1
        total = max(len(results), 1)

        lines = [
            f"Sanity check over {len(results)} images "
            f"(expected: '{self.expected_plate_text}')",
            f"  Correct reads : {counts['correct']:4d} ({counts['correct']/total*100:.1f}%)",
            f"  Impersonation : {counts['impersonation']:4d} ({counts['impersonation']/total*100:.1f}%)",
            f"  Misread       : {counts['misread']:4d} ({counts['misread']/total*100:.1f}%)",
            f"  No detection  : {counts['no_detection']:4d} ({counts['no_detection']/total*100:.1f}%)",
        ]
        report = "\n".join(lines)
        print(report)
        (self.run_dir / "sanity_check.txt").write_text(report + "\n")

        frac = counts["correct"] / total
        if frac < 0.50:
            raise RuntimeError(
                f"\nSanity check FAILED: {counts['correct']}/{total} correct "
                f"({frac*100:.1f}%). Check CSV, expected_plate_text, and model paths."
            )
        print(f"  Passed ({frac*100:.1f}% correct).\n")
        return counts

    # ====================================================================
    # Debug images
    # ====================================================================

    def save_debug_images(self, n: int = 20) -> None:
        debug_dir = self.run_dir / "debug"

        # Collect (fn, prep_tensor, new_corners_np) from both loaders
        total = len(self.train_loader) + len(self.val_loader)
        all_items = []
        with tqdm(total=total, desc="Loading images for debug", leave=False) as pbar:
            for loader in [self.train_loader, self.val_loader]:
                for batch in loader:
                    fn = (batch["filename"][0]
                          if isinstance(batch["filename"], (list, tuple))
                          else batch["filename"])
                    all_items.append((fn, batch["prep_image"][0], batch["new_corners"][0].numpy()))
                    pbar.update(1)

        np.random.seed(42)
        indices      = np.random.choice(len(all_items), min(n, len(all_items)), replace=False)
        sample_items = sorted([all_items[i] for i in indices], key=lambda x: x[0])

        print(f"  Saving {len(sample_items)} debug images → {debug_dir}")
        summary_rows = []

        for img_idx, (fn, prep_t, new_c) in enumerate(sample_items):
            prep_tensor = prep_t.to(self.device)
            new_corners = torch.from_numpy(new_c).to(self.device)
            target_box  = self.corners_to_bbox(new_corners)

            with torch.no_grad():
                detections  = self.detector.predict(prep_tensor)
                crop = kornia.geometry.crop_and_resize(
                    prep_tensor.unsqueeze(0), new_corners.unsqueeze(0),
                    self.ocr.ocr_crop_size, mode="bilinear", align_corners=True,
                )
                ocr_result  = self.ocr.predict(crop.squeeze(0))

            text       = ocr_result.text or ""
            conf       = ocr_result.confidence
            is_correct = text.upper() == self.expected_plate_text.upper()

            vis = (prep_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8).copy()
            pts = new_c.astype(np.int32)

            # (a) preprocessed image + corners + OCR label
            cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.putText(vis, f"{text} ({conf:.2f})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imwrite(str(debug_dir / f"{img_idx:02d}_a_preprocessed_detection.png"), vis)

            # (b) raw OCR crop
            crop_np = (crop.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            cv2.imwrite(str(debug_dir / f"{img_idx:02d}_b_ocr_crop.png"), crop_np)

            # (c) random patch applied — use a fresh random seed through the decoder
            with torch.no_grad():
                rand_seed  = torch.randn_like(self.seed)
                rand_patch = self.decoder(rand_seed).squeeze(0)   # [3, H, W]
                patched, _ = self.apply_patch_to_image(
                    prep_tensor.unsqueeze(0), new_corners.unsqueeze(0),
                    patch_norm=rand_patch,
                )
            patch_vis = (patched.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            cv2.imwrite(str(debug_dir / f"{img_idx:02d}_c_random_patch.png"), patch_vis)

            summary_rows.append({
                "index": img_idx, "filename": fn,
                "detected_text": text, "confidence": f"{conf:.4f}",
                "correct": is_correct,
            })

        csv_path = debug_dir / "debug_summary.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["index", "filename", "detected_text", "confidence", "correct"])
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"  Debug summary → {csv_path}")

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
                    torch.nn.utils.clip_grad_norm_(self._trainable_params(), max_norm=1.0)
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
                    pbar.set_postfix({"loss": f"{total_loss/(num_updates*update_every):.4f}"})
                else:
                    del loss, scaled_loss

            if step % update_every != 0 and self.grad_accumulate is not None:
                torch.nn.utils.clip_grad_norm_(self._trainable_params(), max_norm=1.0)
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
                prep_tensor = batch["prep_image"][0].to(self.device)
                new_corners = batch["new_corners"][0].to(self.device)
                patch_norm  = self.generate_patch(training_aug=False)
                patched_prep, _ = self.apply_patch_to_image(
                    prep_tensor.unsqueeze(0), new_corners.unsqueeze(0),
                    patch_norm=patch_norm,
                )
                det_loss, ocr_loss = self.partial_loss(patched_prep, new_corners)
                losses.append(((det_loss + ocr_loss) / 2).item())
        return float(np.mean(losses)) if losses else 0.0

    def train(
        self,
        num_epochs:    int   = 150,
        learning_rate: float = 0.01,
        save_interval: int   = 10,
        dry_run:       bool  = False,
    ) -> dict:
        self.save_debug_images()
        self.validate_pipeline()

        if dry_run:
            print("\nDry run complete.")
            return {}

        optimizer = optim.AdamW(
            self._trainable_params(), lr=learning_rate, weight_decay=1e-4
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=1e-4,
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
        print(f"  Epochs    : {num_epochs}  |  LR: {learning_rate}")
        print(f"  Mode      : "
              f"{'impersonation → ' + self.impersonation_target if self.impersonation_target else 'disruption'}")
        print(f"  Run dir   : {self.run_dir}")
        print(f"{'='*60}\n")

        log_path = self.run_dir / "training_log.txt"
        log_file = open(log_path, "w")

        for epoch in range(num_epochs):
            self.training  = True
            train_loss     = self.train_epoch(optimizer, epoch)
            self.training  = False
            val_loss       = self.validate()
            scheduler.step()
            lr = optimizer.param_groups[0]["lr"]

            history["loss"].append(train_loss)
            history["val_score"].append(val_loss)
            history["learning_rate"].append(lr)

            init_val    = history["val_score"][0]
            change      = (val_loss / (init_val + 1e-9) - 1) * 100
            best_marker = ""

            if val_loss < best_loss:
                best_loss  = val_loss
                best_epoch = epoch
                self.save_patch(epoch, "patches")
                best_marker = "  ★ best"

            line = (f"Epoch {epoch+1:3d}/{num_epochs} | "
                    f"Loss: {train_loss:.4f} | Val: {val_loss:.4f} | "
                    f"Δ: {change:+.1f}% | LR: {lr:.2e}{best_marker}")
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

    TRAINABLE_DET = ["yolov8", "fasterrcnn", "yolov11", "rtdetr", "yolo-v9-384"]
    TRAINABLE_OCR = ["crnn", "trocr", "dtrb", "lprnet", "cct", "fastanpr-ocr"]

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
    parser.add_argument("--epochs",   type=int,   default=150)
    parser.add_argument("--lr",       type=float, default=0.01)
    parser.add_argument("--grad-accumulate", type=int, default=64)
    parser.add_argument("--preload-images",  action="store_true")
    parser.add_argument("--num-workers",     type=int, default=0)
    parser.add_argument("--pin-memory",      action="store_true")
    parser.add_argument("--limit",           type=int, default=0)
    parser.add_argument("--use-all-for-train", action="store_true")
    parser.add_argument("--impersonation-target", default=None)
    parser.add_argument("--expected-plate", default="VRJ7774")
    parser.add_argument("--disable-homography", action="store_true")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--dry-run",  action="store_true")
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

    trainer = AdversarialPatchTrainer(
        detector             = backend,
        ocr                  = ocr,
        csv_path             = args.csv,
        seed_channels        = args.seed_channels,
        preload_images       = args.preload_images,
        num_workers          = args.num_workers,
        pin_memory           = args.pin_memory,
        limit                = args.limit,
        use_all_for_train    = args.use_all_for_train,
        grad_accumulate      = args.grad_accumulate,
        impersonation_target = args.impersonation_target,
        expected_plate_text  = args.expected_plate,
        training             = True,
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
