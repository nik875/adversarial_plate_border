"""
trainer.py  (refactored)

AdversarialPatchTrainer now accepts any DetectorBackend so you can swap
the detector without touching training logic.

Key changes from the original
------------------------------
* ``self.model`` replaced by ``self.detector`` (a DetectorBackend).
* Model loading removed from __init__ / load_yolo_model – callers supply
  a pre-built, *already loaded* backend.
* ``partial_loss`` reads detections via ``backend.predict()`` which returns
  ``List[Detection]``; the raw 7-element tensor is still available via
  ``det.raw`` for gradient-compatible operations.
* ``build_trainer_from_args`` helper creates the backend + trainer in one
  call for CLI use.
"""

from __future__ import annotations

import os
import warnings
import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torchvision.transforms as T
import kornia
import kornia.geometry as K
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
from tqdm import tqdm

from detector_backends import DetectorBackend, Detection, build_backend
from ocr_backends import OCRBackend, OCRResult, build_ocr_backend
from dataset import create_dataloaders

warnings.filterwarnings("ignore")

PATCH_WIDTH  = 512
PATCH_HEIGHT = 256


class AdversarialPatchTrainer:
    def __init__(
        self,
        csv_path: str,
        detector: DetectorBackend,
        ocr: OCRBackend,                     # ← injected, swappable
        preload_images: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
        limit: int = 0,
        use_all_for_train: bool = False,
        grad_accumulate: int = None,
        match_detection: bool = False,
        impersonation_target: Optional[str] = None,
        print_blur: float = 0.0,
        training: bool = False,
        use_tv_loss: bool = True,
        use_homography: bool = True,
    ):
        self.training             = training
        self.print_blur           = print_blur
        self.use_tv_loss          = use_tv_loss
        self.use_homography       = use_homography
        self.grad_accumulate      = grad_accumulate
        self.match_detection      = match_detection
        self.impersonation_target = impersonation_target

        # ── Detector (swappable) ────────────────────────────────────────
        self.detector = detector
        self.device   = detector.device
        self.detector.eval()
        self.detector.freeze()

        # ── Image transforms ────────────────────────────────────────────
        self.transform = T.Compose([T.ToTensor()])

        self.patch_width  = PATCH_WIDTH
        self.patch_height = PATCH_HEIGHT

        # ── Data ────────────────────────────────────────────────────────
        self.train_loader, self.val_loader = create_dataloaders(
            csv_path, transform=self.transform,
            preload=preload_images,
            batch_size=1,
            n_jobs=num_workers,
            pin_memory=pin_memory,
            limit=limit,
            use_all_for_train=use_all_for_train,
        )

        # ── Adversarial patch parameter ─────────────────────────────────
        self.patch = nn.Parameter(
            torch.randn(3, self.patch_height, self.patch_width,
                        device=self.device) * 0.1
        )

        # ── OCR (swappable) ─────────────────────────────────────────────
        self.ocr = ocr
        self.ocr_input_shape = (64, 128, 3)   # H, W fed to OCR crop
        # Keep trainable OCR unfrozen so gradients can flow
        if not self.ocr.is_trainable:
            self.ocr.eval()
            self.ocr.freeze()
        else:
            self.ocr.eval()  # Start in eval, switch to train during training

        # Baselines (computed on clean images)
        self.detection_baseline, self.ocr_baseline = self._calculate_baseline_loss()

        self.epoch_stats: list = []

    # ====================================================================
    # Helpers
    # ====================================================================

    def _text_to_target_tensor(self, text: str, max_slots: int,
                                alphabet: str) -> torch.Tensor:
        padded  = (text + "_" * max_slots)[:max_slots]
        indices = [alphabet.index(c) for c in padded]
        target  = torch.zeros(1, max_slots, len(alphabet))
        for i, idx in enumerate(indices):
            target[0, i, idx] = 1.0
        return target.to(self.device)

    def invert_bbox(self, corners: torch.Tensor, transform) -> torch.Tensor:
        r, dw, dh = transform
        corners = corners.clone()
        corners[::2]  -= dw
        corners[1::2] -= dh
        corners /= r
        return corners

    def bbox_to_corners(self, bbox, device=None) -> torch.Tensor:
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
        cx = corners[:, 0].mean()
        cy = corners[:, 1].mean()
        center = torch.tensor([cx, cy], device=self.device)
        border = center.unsqueeze(0) + (corners - center.unsqueeze(0)) * border_scale
        return torch.stack([border[:, 0].min(), border[:, 1].min(),
                            border[:, 0].max(), border[:, 1].max()])

    @staticmethod
    def _boxes_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
        b1 = boxes1.unsqueeze(1)
        b2 = boxes2.unsqueeze(0)
        inter_w = torch.clamp(torch.min(b1[..., 2], b2[..., 2]) -
                              torch.max(b1[..., 0], b2[..., 0]), min=0)
        inter_h = torch.clamp(torch.min(b1[..., 3], b2[..., 3]) -
                              torch.max(b1[..., 1], b2[..., 1]), min=0)
        inter = inter_w * inter_h
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
        cx = plate[:, 0].mean()
        cy = plate[:, 1].mean()
        ctr = torch.tensor([cx, cy], device=self.device)
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
        B = image.shape[0]
        H, W = image.shape[2], image.shape[3]

        patch_norm = torch.tanh(self.patch) * 0.5 + 0.5

        if self.print_blur > 0:
            patch_norm = kornia.filters.gaussian_blur2d(
                patch_norm.unsqueeze(0), (3, 3),
                (self.print_blur, self.print_blur),
            ).squeeze(0)

        if self.training:
            factor = torch.rand(1, device=self.device) * 0.2
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
        src = torch.tensor([[0, 0], [pw, 0], [pw, ph], [0, ph]],
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
    # Loss computation
    # ====================================================================

    def patch_reg_loss(self) -> torch.Tensor:
        p   = self.patch
        C, H, W = p.shape
        tv_h = torch.pow(p[:, :, 1:] - p[:, :, :-1], 2).sum()
        tv_v = torch.pow(p[:, 1:, :] - p[:, :-1, :], 2).sum()
        n    = C * (H * (W - 1) + (H - 1) * W)
        return (tv_h + tv_v) / n * 2.5

    def partial_loss(self, batch: dict,
                     use_ocr_baseline: bool = True) -> Tuple[torch.Tensor, float]:
        """
        Core loss: detection suppression + OCR disruption/impersonation.

        Uses ``self.detector.predict()`` so the backend is fully swappable.
        Gradient flow is preserved through ``Detection.raw``.
        """
        prep_image = batch["prep_image"].to(self.device)
        corners    = batch["new_corners"].to(self.device)

        if self.match_detection:
            target_box = self.get_patch_bounding_box(corners)
        else:
            target_box = self.corners_to_bbox(corners)

        # ── Detection loss ──────────────────────────────────────────────
        detections = self.detector.predict(prep_image)

        best_detection: Optional[Detection] = None
        det_loss = torch.tensor(0.0, device=self.device)

        for det in detections:
            pred_box = det.box.to(self.device)
            conf     = det.conf.to(self.device)
            iou      = self._boxes_iou(pred_box.unsqueeze(0),
                                       target_box.unsqueeze(0))

            if self.match_detection:
                this_loss = -iou * conf       # minimise → maximise overlap
            else:
                this_loss = iou * conf        # minimise → suppress plate

            # Pick the detection that contributes most to current objective
            if self.match_detection:
                if (-this_loss).item() > (-det_loss).item():
                    det_loss       = this_loss
                    best_detection = det
            else:
                if this_loss.item() > det_loss.item():
                    det_loss       = this_loss
                    best_detection = det

        # ── OCR loss ────────────────────────────────────────────────────
        ocr_loss = 0.0
        if best_detection is not None:
            pred_box        = best_detection.box
            orig_projection = self.invert_bbox(pred_box.cpu(), batch["transform"])
            corners_box     = self.bbox_to_corners(orig_projection, device="cpu")

            # Crop plate region — keep in the autograd graph for trainable OCR
            cropped = kornia.geometry.crop_and_resize(
                batch["orig_image"].unsqueeze(0),
                corners_box,
                self.ocr_input_shape[:2],
                mode="bilinear", align_corners=True,
            ).to(self.device).squeeze(0)   # [C, H, W]

            target_text = self.impersonation_target or "VRJ7774"
            ocr_result  = self.ocr.predict(cropped)

            if self.ocr.is_trainable and hasattr(self.ocr, "compute_target_loss"):
                # Backend-native differentiable loss (e.g., TrOCR seq2seq loss)
                base_loss = self.ocr.compute_target_loss(cropped, target_text)
                if self.impersonation_target:
                    ocr_loss = base_loss
                else:
                    # Disruption: maximise recognition loss
                    ocr_loss = -base_loss
            elif self.ocr.is_trainable and ocr_result.logits is not None:
                # Differentiable CTC path (CRNN-like backends)
                if self.impersonation_target:
                    # Impersonation: minimise CTC loss toward target
                    ocr_loss = self.ocr.ctc_loss(ocr_result.logits, target_text)
                else:
                    # Disruption: maximise CTC loss for the true plate text
                    # (minimise negative CTC loss)
                    ocr_loss = -self.ocr.ctc_loss(ocr_result.logits, target_text)
            else:
                # Non-differentiable path: character-accuracy heuristic
                accuracy = ocr_result.char_accuracy(target_text)
                ocr_loss = (1.0 - accuracy) if self.impersonation_target else accuracy

            if use_ocr_baseline and hasattr(self, "ocr_baseline"):
                if isinstance(ocr_loss, torch.Tensor):
                    ocr_loss = ocr_loss / (self.ocr_baseline + 1e-8)
                else:
                    # Keep scalar heuristic stable across OCR backends.
                    # The previous disruption path used baseline/accuracy,
                    # which explodes when accuracy ~= 0 and causes huge
                    # oscillations in val loss.
                    baseline = max(float(self.ocr_baseline), 1e-4)
                    ocr_loss = float(ocr_loss) / baseline

        return det_loss, ocr_loss

    def _calculate_baseline_loss(self) -> Tuple[float, float]:
        total_det = total_ocr = count = 0.0
        with tqdm(self.train_loader, desc="Baseline", leave=False):
            with torch.no_grad():
                for batch in self.train_loader:
                    batch = {k: v[0] for k, v in batch.items()}
                    d, o = self.partial_loss(batch, use_ocr_baseline=False)
                    total_det += d if isinstance(d, float) else d.item()
                    total_ocr += o
                    count     += 1
        return total_det / max(count, 1), total_ocr / max(count, 1)

    def compute_loss(self, batch: dict) -> torch.Tensor:
        batch = {k: v[0] for k, v in batch.items()}

        # Apply patch to preprocessed image
        patched_prep, _ = self.apply_patch_to_image(
            batch["prep_image"].to(self.device).unsqueeze(0),
            batch["new_corners"].to(self.device).unsqueeze(0),
        )
        batch["prep_image"] = patched_prep.squeeze()

        # Apply patch to original-resolution image (for OCR crop)
        patched_orig, _ = self.apply_patch_to_image(
            batch["orig_image"].to(self.device).unsqueeze(0),
            batch["orig_corners"].to(self.device).unsqueeze(0),
        )
        batch["orig_image"] = patched_orig.squeeze()

        det_loss, ocr_loss = self.partial_loss(batch)
        reg_loss = self.patch_reg_loss() if self.use_tv_loss else 0.0
        return (det_loss + ocr_loss) / 2 + reg_loss

    # ====================================================================
    # Training loop
    # ====================================================================

    def train_epoch(self, optimizer, epoch: int) -> float:
        # Ensure trainable OCR is in train mode for CuDNN RNN backward.
        if self.ocr.is_trainable:
            self.ocr.train()
        
        update_every = (len(self.train_loader)
                        if self.grad_accumulate is None
                        else self.grad_accumulate)
        total_loss = accum_loss = 0.0
        step = num_updates = 0

        with tqdm(enumerate(self.train_loader),
                  desc=f"Epoch {epoch+1} [{self.detector.name}]",
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
        # Set OCR back to eval mode for validation
        if self.ocr.is_trainable:
            self.ocr.eval()
        
        losses = []
        with torch.no_grad():
            for batch in self.val_loader:
                losses.append(self.compute_loss(batch).item())
        return float(np.mean(losses))

    def save_patch(self, epoch: int, save_dir: str = "patches") -> None:
        Path(save_dir).mkdir(exist_ok=True)
        with torch.no_grad():
            img = torch.tanh(self.patch) * 0.5 + 0.5
            T.ToPILImage()(img.cpu()).save(
                f"{save_dir}/patch_{self.detector.name}_epoch_{epoch:04d}.png")
            torch.save({"patch": self.patch.detach().cpu(), "epoch": epoch,
                        "backend": self.detector.name,
                        "patch_size": (self.patch_height, self.patch_width)},
                       f"{save_dir}/patch_{self.detector.name}_epoch_{epoch:04d}.pt")

    def train(self, num_epochs: int = 100, learning_rate: float = 0.01,
              save_interval: int = 10, early_stop_patience: int = 15) -> dict:
        optimizer = optim.AdamW([self.patch], lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5)

        history = {"loss": [], "val_score": [], "learning_rate": []}
        best_loss = float("inf")
        patience  = 0

        print(f"\n{'='*60}")
        print(f"  Adversarial Patch Training  |  backend: {self.detector.name}")
        print(f"{'='*60}")
        print(f"  Dataset   : {len(self.train_loader)+len(self.val_loader)} images")
        print(f"  Patch     : {self.patch_height}×{self.patch_width}")
        print(f"  Device    : {self.device}")
        print(f"  Epochs    : {num_epochs}  |  LR: {learning_rate}")
        print(f"  TV loss   : {self.use_tv_loss}  |  Homography: {self.use_homography}")
        print(f"  Mode      : "
              f"{'impersonation → ' + self.impersonation_target if self.impersonation_target else 'disruption'}")
        print(f"{'='*60}\n")

        for epoch in range(num_epochs):
            train_loss = self.train_epoch(optimizer, epoch)
            val_loss   = self.validate()
            scheduler.step(train_loss)
            lr = optimizer.param_groups[0]["lr"]

            history["loss"].append(train_loss)
            history["val_score"].append(val_loss)
            history["learning_rate"].append(lr)

            init_val = history["val_score"][0]
            change   = (val_loss / init_val - 1) * 100

            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Loss: {train_loss:.4f} | Val: {val_loss:.4f} | "
                  f"Δ: {change:+.1f}% | LR: {lr:.2e}")

            if val_loss < best_loss:
                best_loss = val_loss
                patience  = 0
                self.save_patch(epoch, "best_patches")
            else:
                patience += 1

            if (epoch + 1) % save_interval == 0:
                self.save_patch(epoch, "checkpoint_patches")

            if patience >= early_stop_patience:
                print(f"  Early stop: no improvement for {early_stop_patience} epochs")
                break

            if len(history["loss"]) >= 20:
                recent = history["loss"][-20:]
                if max(recent) - min(recent) < 1e-4:
                    print("  Converged: loss stabilised")
                    break

        print(f"\nDone. Best val loss: {best_loss:.4f}")
        return history


# ====================================================================
# CLI entry-point
# ====================================================================

def main():
    parser = argparse.ArgumentParser(description="Adversarial patch trainer (modular backend)")
    parser.add_argument("--csv", default="updated_control_corners.csv")
    # Only backends with a fully differentiable PyTorch forward pass are valid
    # training targets. ONNX/numpy backends (fastanpr, open-image-models,
    # yolov5, yolo-nas) break the autograd graph -- use evaluator.py for those.
    TRAINABLE_DET = ["yolov8", "fasterrcnn", "yolov11", "rtdetr"]
    TRAINABLE_OCR = ["crnn", "trocr", "dtrb", "fastanpr-ocr"]
    parser.add_argument("--backend", default="yolov8", choices=TRAINABLE_DET,
                        help=f"Detector to train a patch against. "
                             f"One of: {', '.join(TRAINABLE_DET)}. "
                             f"For eval-only backends use evaluator.py.")
    parser.add_argument("--model-path", default="license_plate_detector.pt",
                        help="Path to detector weights (.pt file).")
    parser.add_argument("--ocr-backend", default="fastanpr-ocr", choices=TRAINABLE_OCR,
                        help="OCR backend. Use 'crnn', 'trocr', or 'dtrb' for differentiable training.")
    parser.add_argument("--ocr-model-path", default="none",
                        help="Path to OCR weights/checkpoint. For trocr pass 'none' (uses HF default) or a model id/path.")
    parser.add_argument("--ocr-repo-root", default="/home/ubuntu/deep-text-recognition-benchmark",
                        help="Path to DTRB repo for model definitions (required for --ocr-backend dtrb).")
    parser.add_argument("--dtrb-feature-extraction", default="vitstr_small_patch16_224",
                        help="DTRB feature extraction module (e.g. 'ResNet', 'vitstr_small_patch16_224').")
    parser.add_argument("--dtrb-sequence-modeling", default="None",
                        help="DTRB sequence modeling (e.g. 'BiLSTM', 'None' for ViTSTR).")
    parser.add_argument("--dtrb-transformation", default="None",
                        help="DTRB transformation (e.g. 'TPS', 'None').")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--grad-accumulate", type=int, default=64)
    parser.add_argument("--preload-images", action="store_true",
                        help="Preload all images into RAM (faster, but high memory use).")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers. 0 is safest for low-memory hosts.")
    parser.add_argument("--pin-memory", action="store_true",
                        help="Enable pinned host memory for DataLoader (can increase RAM usage).")
    parser.add_argument("--limit", type=int, default=0,
                        help="Use only the last N samples from CSV (0 = all).")
    parser.add_argument("--use-all-for-train", action="store_true",
                        help="Skip validation split and train on the full CSV.")
    parser.add_argument("--match-detection", action="store_true")
    parser.add_argument("--impersonation-target", default=None)
    parser.add_argument("--disable-tv-loss", action="store_true")
    parser.add_argument("--disable-homography", action="store_true")
    args = parser.parse_args()

    backend = build_backend(args.backend, args.model_path, device=args.device)
    backend.load()

    ocr_kwargs = {}
    if args.ocr_backend == "dtrb":
        if args.ocr_repo_root:
            ocr_kwargs["dtrb_root"] = args.ocr_repo_root
        ocr_kwargs["feature_extraction"] = args.dtrb_feature_extraction
        ocr_kwargs["sequence_modeling"] = args.dtrb_sequence_modeling
        ocr_kwargs["transformation"] = args.dtrb_transformation
    ocr = build_ocr_backend(args.ocr_backend, args.ocr_model_path, device=args.device, **ocr_kwargs)
    ocr.load()

    if not ocr.is_trainable:
        print(
            f"Note: '{args.ocr_backend}' OCR has no gradient graph — "
            "OCR loss will use character-accuracy heuristic only.\n"
            "      Use --ocr-backend crnn for fully differentiable training."
        )

    trainer = AdversarialPatchTrainer(
        csv_path             = args.csv,
        detector             = backend,
        ocr                  = ocr,
        preload_images       = args.preload_images,
        num_workers          = args.num_workers,
        pin_memory           = args.pin_memory,
        limit                = args.limit,
        use_all_for_train    = args.use_all_for_train,
        grad_accumulate      = args.grad_accumulate,
        match_detection      = args.match_detection,
        impersonation_target = args.impersonation_target,
        training             = True,
        use_tv_loss          = not args.disable_tv_loss,
        use_homography       = not args.disable_homography,
    )

    history = trainer.train(num_epochs=args.epochs, learning_rate=args.lr,
                            save_interval=10, early_stop_patience=20)

    # Save results
    import pandas as pd
    pd.DataFrame(history).assign(
        epoch=range(1, len(history["loss"]) + 1)
    ).to_csv("training_history.csv", index=False)
    print("Training history → training_history.csv")


if __name__ == "__main__":
    main()
