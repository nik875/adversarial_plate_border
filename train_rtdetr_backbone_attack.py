#!/usr/bin/env python3
"""
train_rtdetr_backbone_attack.py

Alternative RT-DETR attack training that targets internal backbone/encoder features,
not only final detection outputs.

This script trains a universal border patch with the same dual-zone structure
used in trainer.py and adds a backbone feature term.

The objective combines:
1) OCR impersonation loss on real + top-attractor crops (weighted as in trainer.py),
2) top-zone detection attractor loss from the dual-zone selector,
3) backbone feature similarity between clean and patched RT-DETR representations,
4) total variation regularization for patch smoothness.

It is intentionally separate from trainer.py/train_segmented.py so you can run
feature-level attacks without changing the existing pipeline.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch import optim
from tqdm import tqdm

from dataset import create_dataloaders
from ocr_backends import build_ocr_backend


DEFAULT_MODEL_ID = "justjuu/rtdetr-v2-license-plate-detection"
PATCH_H = 256
PATCH_W = 512


@dataclass
class TrainState:
    epoch: int = 0
    global_step: int = 0


class RTDETRBackbonePatchTrainer:
    def __init__(
        self,
        csv_path: str,
        model_path: str,
        run_dir: Path,
        device: str,
        batch_size: int,
        n_jobs: int,
        preload: bool,
        border_scale: float,
        feature_weight: float,
        det_weight: float,
        ocr_weight: float,
        tv_weight: float,
        lr: float,
        conf_threshold: float,
        backbone_source: str,
        ocr_backend: str,
        ocr_model_path: str,
        impersonation_target: str,
    ) -> None:
        self.csv_path = csv_path
        self.model_path = model_path
        self.run_dir = run_dir
        self.device = device
        self.batch_size = batch_size
        self.border_scale = border_scale
        self.feature_weight = feature_weight
        self.det_weight = det_weight
        self.ocr_weight = ocr_weight
        self.tv_weight = tv_weight
        self.conf_threshold = conf_threshold
        self.backbone_source = backbone_source
        self.ocr_backend_name = ocr_backend
        self.ocr_model_path = ocr_model_path
        self.impersonation_target = impersonation_target

        self.patch = torch.nn.Parameter(
            torch.randn(3, PATCH_H, PATCH_W, device=self.device) * 0.1
        )
        self.optimizer = optim.AdamW([self.patch], lr=lr, weight_decay=1e-4)

        self._load_model()
        self._load_ocr()

        self.train_loader, self.val_loader = create_dataloaders(
            csv_path,
            batch_size=batch_size,
            n_jobs=n_jobs,
            preload=preload,
            pin_memory=(device.startswith("cuda") and not preload),
            transform=None,
        )

    def _load_ocr(self) -> None:
        source = self.ocr_model_path
        if source in {"", "none"}:
            source = "none"

        print(f"[backbone-attack] Loading OCR backend={self.ocr_backend_name} from: {source}")
        self.ocr = build_ocr_backend(
            self.ocr_backend_name,
            model_path=source,
            device=self.device,
        )
        self.ocr.load()
        self.ocr.eval()
        self.ocr.freeze()

        if not getattr(self.ocr, "is_trainable", False):
            raise RuntimeError(
                f"OCR backend {self.ocr_backend_name!r} is not differentiable; "
                "choose a trainable backend (e.g. crnn, cct, lprnet, trocr)."
            )

    def _load_model(self) -> None:
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        source = self.model_path
        if source in {"", "none"} or not Path(source).exists():
            source = DEFAULT_MODEL_ID

        print(f"[backbone-attack] Loading RT-DETR from: {source}")
        self.processor = AutoImageProcessor.from_pretrained(source)
        self.model = AutoModelForObjectDetection.from_pretrained(source)
        self.model.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @staticmethod
    def _to_float01(x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if x.max().item() > 1.0:
            x = x / 255.0
        return x.clamp(0.0, 1.0)

    def _diff_preprocess_batch(self, images: torch.Tensor) -> torch.Tensor:
        size = self.processor.size
        th = size.get("height", size.get("shortest_edge", 640))
        tw = size.get("width", size.get("shortest_edge", 640))

        x = F.interpolate(images, size=(th, tw), mode="bilinear", align_corners=False)
        mean = torch.tensor(self.processor.image_mean, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(self.processor.image_std, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        return (x - mean) / std

    @staticmethod
    def _corners_to_bbox(corners: torch.Tensor) -> torch.Tensor:
        min_xy = corners.min(dim=1).values
        max_xy = corners.max(dim=1).values
        return torch.cat([min_xy, max_xy], dim=1)

    @staticmethod
    def _top_extend_region_corners_batch(corners: torch.Tensor) -> torch.Tensor:
        col_left = corners[:, 0] - corners[:, 3]
        col_right = corners[:, 1] - corners[:, 2]
        return torch.stack([
            corners[:, 0] + 1.4 * col_left,
            corners[:, 1] + 1.4 * col_right,
            corners[:, 1] + 0.4 * col_right,
            corners[:, 0] + 0.4 * col_left,
        ], dim=1)

    def _top_extend_region_bbox_batch(self, corners: torch.Tensor) -> torch.Tensor:
        top = self._top_extend_region_corners_batch(corners)
        min_xy = top.min(dim=1).values
        max_xy = top.max(dim=1).values
        return torch.cat([min_xy, max_xy], dim=1)

    @staticmethod
    def _box_iou_matrix(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        # boxes1: [N,4], boxes2: [M,4]
        area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
        area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

        lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
        rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[..., 0] * wh[..., 1]
        union = area1[:, None] + area2[None, :] - inter + 1e-6
        return inter / union

    @staticmethod
    def _bbox_ocr_crop_diff(
        img: torch.Tensor,
        box: torch.Tensor,
        target_size: Tuple[int, Optional[int]],
    ) -> torch.Tensor:
        h, w = img.shape[-2], img.shape[-1]
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        target_h = target_size[0]
        target_w = target_size[1]

        if target_w is None:
            with torch.no_grad():
                bw = (x2 - x1).clamp(min=1)
                bh = (y2 - y1).clamp(min=1)
                target_w = max(1, int((bw / bh * target_h).item()))

        x1n = x1 / w * 2 - 1
        y1n = y1 / h * 2 - 1
        x2n = x2 / w * 2 - 1
        y2n = y2 / h * 2 - 1

        xs = torch.linspace(0, 1, target_w, device=img.device, dtype=img.dtype)
        ys = torch.linspace(0, 1, target_h, device=img.device, dtype=img.dtype)
        grid_y, grid_x = torch.meshgrid(
            y1n + ys * (y2n - y1n),
            x1n + xs * (x2n - x1n),
            indexing="ij",
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        return F.grid_sample(
            img,
            grid,
            mode="bilinear",
            align_corners=True,
            padding_mode="zeros",
        )

    def _select_best_from_scores_boxes(
        self,
        scores: torch.Tensor,
        boxes_xyxy: torch.Tensor,
        target_box: torch.Tensor,
        exclude_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        tb = target_box.to(self.device)
        proximity_weights = None
        best_idx = None
        with torch.no_grad():
            ious = self._box_iou_matrix(boxes_xyxy.detach(), tb.unsqueeze(0)).squeeze(1)
            if ious.max().item() < 1e-6:
                box_centers = (boxes_xyxy[:, :2] + boxes_xyxy[:, 2:]) / 2
                target_center = torch.stack([(tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2])
                dists = ((box_centers - target_center) ** 2).sum(-1).sqrt()
                target_size = max((tb[2] - tb[0]).item(), (tb[3] - tb[1]).item(), 1.0)
                if exclude_mask is not None and exclude_mask.any() and not exclude_mask.all():
                    dists = dists.clone()
                    dists[exclude_mask.to(self.device)] = float("inf")
                proximity_weights = torch.softmax(-dists / target_size, dim=0)
            else:
                best_idx = int((ious * scores.detach()).argmax().item())

        if proximity_weights is not None:
            return (scores * proximity_weights).sum(), None
        return scores[best_idx], boxes_xyxy[best_idx]

    def _select_two_targets(
        self,
        scores: torch.Tensor,
        boxes_xyxy: torch.Tensor,
        target_box1: torch.Tensor,
        target_box2: torch.Tensor,
    ) -> Tuple[Tuple[torch.Tensor, Optional[torch.Tensor]], Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        with torch.no_grad():
            ious1 = self._box_iou_matrix(boxes_xyxy.detach(), target_box1.unsqueeze(0)).squeeze(1)
            ious2 = self._box_iou_matrix(boxes_xyxy.detach(), target_box2.unsqueeze(0)).squeeze(1)
            has_overlap1 = ious1.max().item() >= 1e-6
            has_overlap2 = ious2.max().item() >= 1e-6

        exclude1 = (ious2 > 1e-6) if (not has_overlap1 and has_overlap2) else None
        exclude2 = (ious1 > 1e-6) if (not has_overlap2 and has_overlap1) else None
        r1 = self._select_best_from_scores_boxes(scores, boxes_xyxy, target_box1, exclude_mask=exclude1)
        r2 = self._select_best_from_scores_boxes(scores, boxes_xyxy, target_box2, exclude_mask=exclude2)
        return r1, r2

    @staticmethod
    def _pred_boxes_xyxy(outputs, image_hw: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = outputs.logits
        pred_boxes = outputs.pred_boxes
        scores = logits.sigmoid().max(dim=-1).values
        h, w = image_hw
        cx, cy, bw, bh = pred_boxes.unbind(dim=-1)
        boxes_xyxy = torch.stack([
            (cx - bw / 2) * w,
            (cy - bh / 2) * h,
            (cx + bw / 2) * w,
            (cy + bh / 2) * h,
        ], dim=-1)
        return scores, boxes_xyxy

    def _dual_zone_losses(
        self,
        out_patch,
        patched: torch.Tensor,
        target_boxes: torch.Tensor,
        top_region_boxes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        scores, boxes_xyxy = self._pred_boxes_xyxy(
            out_patch,
            (patched.shape[2], patched.shape[3]),
        )

        det_real_list = []
        det_top_list = []
        ocr_real_list = []
        ocr_top_list = []
        ocr_mix_list = []

        for i in range(patched.shape[0]):
            (conf_real, pred_real), (conf_top, pred_top) = self._select_two_targets(
                scores[i],
                boxes_xyxy[i],
                target_boxes[i],
                top_region_boxes[i],
            )

            if pred_real is not None:
                iou_real = self._box_iou_matrix(
                    pred_real.unsqueeze(0),
                    target_boxes[i].unsqueeze(0),
                ).squeeze()
                det_real_i = iou_real
            else:
                det_real_i = conf_real

            if pred_top is not None:
                iou_top = self._box_iou_matrix(
                    pred_top.unsqueeze(0),
                    top_region_boxes[i].unsqueeze(0),
                ).squeeze()
                det_top_i = -iou_top
            else:
                det_top_i = -conf_top

            real_box_for_crop = pred_real if pred_real is not None else target_boxes[i]
            top_box_for_crop = pred_top if pred_top is not None else top_region_boxes[i]

            real_crop = self._bbox_ocr_crop_diff(
                patched[i:i + 1],
                real_box_for_crop,
                self.ocr.ocr_crop_size,
            )
            top_crop = self._bbox_ocr_crop_diff(
                patched[i:i + 1],
                top_box_for_crop,
                self.ocr.ocr_crop_size,
            )

            ocr_real_i = self.ocr.differentiable_loss_batch(
                [real_crop],
                self.impersonation_target,
                impersonation=True,
            )[0]
            ocr_top_i = self.ocr.differentiable_loss_batch(
                [top_crop],
                self.impersonation_target,
                impersonation=True,
            )[0]

            det_real_mag = det_real_i.clamp(min=0)
            det_top_mag = (-det_top_i).clamp(min=0)
            total_mag = det_real_mag + det_top_mag + 1e-6
            w_real = det_real_mag / total_mag
            w_top = det_top_mag / total_mag
            ocr_mix_i = w_real * ocr_real_i + w_top * ocr_top_i

            det_real_list.append(det_real_i)
            det_top_list.append(det_top_i)
            ocr_real_list.append(ocr_real_i)
            ocr_top_list.append(ocr_top_i)
            ocr_mix_list.append(ocr_mix_i)

        return (
            torch.stack(ocr_mix_list).mean(),
            torch.stack(det_real_list).mean(),
            torch.stack(det_top_list).mean(),
            torch.stack(ocr_real_list).mean(),
            torch.stack(ocr_top_list).mean(),
        )

    def _apply_border_patch(
        self,
        images: torch.Tensor,
        corners: torch.Tensor,
    ) -> torch.Tensor:
        """Apply a simple rectangular border patch around the plate bbox."""
        b, _, h, w = images.shape
        out = images.clone()
        patch01 = (torch.tanh(self.patch) * 0.5 + 0.5).unsqueeze(0)

        for i in range(b):
            pts = corners[i]
            cx = pts[:, 0].mean()
            cy = pts[:, 1].mean()
            center = torch.tensor([cx, cy], device=self.device)
            border_pts = center.unsqueeze(0) + (pts - center.unsqueeze(0)) * self.border_scale

            bx1 = int(torch.clamp(border_pts[:, 0].min(), 0, w - 1).item())
            bx2 = int(torch.clamp(border_pts[:, 0].max(), 1, w).item())
            by1 = int(torch.clamp(border_pts[:, 1].min(), 0, h - 1).item())
            by2 = int(torch.clamp(border_pts[:, 1].max(), 1, h).item())

            px1 = int(torch.clamp(pts[:, 0].min(), 0, w - 1).item())
            px2 = int(torch.clamp(pts[:, 0].max(), 1, w).item())
            py1 = int(torch.clamp(pts[:, 1].min(), 0, h - 1).item())
            py2 = int(torch.clamp(pts[:, 1].max(), 1, h).item())

            bh, bw = by2 - by1, bx2 - bx1
            if bh <= 2 or bw <= 2:
                continue

            resized = F.interpolate(patch01, size=(bh, bw), mode="bilinear", align_corners=False)[0]
            mask = torch.ones((1, bh, bw), device=self.device)

            # Cut out the inner plate rectangle so only border remains.
            iy1 = max(0, py1 - by1)
            iy2 = min(bh, py2 - by1)
            ix1 = max(0, px1 - bx1)
            ix2 = min(bw, px2 - bx1)
            if iy2 > iy1 and ix2 > ix1:
                mask[:, iy1:iy2, ix1:ix2] = 0.0

            mask3 = mask.expand(3, -1, -1)
            src = out[i, :, by1:by2, bx1:bx2]
            out[i, :, by1:by2, bx1:bx2] = src * (1.0 - mask3) + resized * mask3

        return out.clamp(0.0, 1.0)

    def _extract_feature_tensor(self, outputs) -> torch.Tensor:
        """Pick a robust internal feature tensor from RT-DETR outputs."""
        if self.backbone_source == "encoder":
            if hasattr(outputs, "encoder_last_hidden_state") and outputs.encoder_last_hidden_state is not None:
                return outputs.encoder_last_hidden_state
            if hasattr(outputs, "encoder_hidden_states") and outputs.encoder_hidden_states:
                return outputs.encoder_hidden_states[-1]

        if self.backbone_source == "decoder":
            if hasattr(outputs, "decoder_hidden_states") and outputs.decoder_hidden_states:
                return outputs.decoder_hidden_states[-1]

        # auto fallback
        if hasattr(outputs, "encoder_last_hidden_state") and outputs.encoder_last_hidden_state is not None:
            return outputs.encoder_last_hidden_state
        if hasattr(outputs, "encoder_hidden_states") and outputs.encoder_hidden_states:
            return outputs.encoder_hidden_states[-1]
        if hasattr(outputs, "decoder_hidden_states") and outputs.decoder_hidden_states:
            return outputs.decoder_hidden_states[-1]
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            return outputs.last_hidden_state

        raise RuntimeError("Could not locate hidden states in RT-DETR outputs.")

    def _feature_attack_loss(self, clean_out, patched_out) -> torch.Tensor:
        clean_feat = self._extract_feature_tensor(clean_out).detach()
        patch_feat = self._extract_feature_tensor(patched_out)

        clean_vec = clean_feat.flatten(start_dim=1)
        patch_vec = patch_feat.flatten(start_dim=1)
        cos = F.cosine_similarity(patch_vec, clean_vec, dim=1)
        return cos.mean()

    def _tv_loss(self) -> torch.Tensor:
        p = self.patch
        tv_h = torch.pow(p[:, :, 1:] - p[:, :, :-1], 2).mean()
        tv_v = torch.pow(p[:, 1:, :] - p[:, :-1, :], 2).mean()
        return tv_h + tv_v

    def _save_patch(self, stem: str) -> None:
        patch_dir = self.run_dir / "patches"
        patch_dir.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            patch01 = (torch.tanh(self.patch) * 0.5 + 0.5).cpu()
            T.ToPILImage()(patch01).save(str(patch_dir / f"{stem}.png"))
            torch.save(
                {
                    "patch": self.patch.detach().cpu(),
                    "stem": stem,
                },
                str(patch_dir / f"{stem}.pt"),
            )

    def save_checkpoint(self, state: TrainState, name: str = "latest.pt") -> Path:
        ckpt_dir = self.run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / name
        torch.save(
            {
                "patch": self.patch.detach().cpu(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": state.epoch,
                "global_step": state.global_step,
            },
            str(path),
        )
        return path

    def load_checkpoint(self, ckpt_path: Path) -> TrainState:
        data = torch.load(str(ckpt_path), map_location="cpu")
        self.patch.data.copy_(data["patch"].to(self.device))
        self.optimizer.load_state_dict(data["optimizer"])
        for g in self.optimizer.state.values():
            for k, v in g.items():
                if isinstance(v, torch.Tensor):
                    g[k] = v.to(self.device)
        return TrainState(
            epoch=int(data.get("epoch", 0)),
            global_step=int(data.get("global_step", 0)),
        )

    def train(self, epochs: int, max_steps: int = 0, save_every: int = 200, start: Optional[TrainState] = None) -> TrainState:
        state = start or TrainState()

        log_path = self.run_dir / "backbone_attack_log.csv"
        write_header = not log_path.exists()
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "global_step",
                    "epoch",
                    "loss_total",
                    "loss_feat",
                    "loss_det_real",
                    "loss_det_top",
                    "loss_ocr_real",
                    "loss_ocr_top",
                    "loss_ocr_mix",
                    "loss_tv",
                ])

            stop = False
            for epoch in range(state.epoch, epochs):
                pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
                epoch_loss = 0.0
                n_batches = 0

                for batch in pbar:
                    prep = self._to_float01(batch["prep_image"].to(self.device))
                    corners = batch["new_corners"].to(self.device).float()

                    patched = self._apply_border_patch(prep, corners)
                    pixel_clean = self._diff_preprocess_batch(prep)
                    pixel_patch = self._diff_preprocess_batch(patched)

                    with torch.no_grad():
                        out_clean = self.model(pixel_values=pixel_clean, output_hidden_states=True)
                    out_patch = self.model(pixel_values=pixel_patch, output_hidden_states=True)

                    target_boxes = self._corners_to_bbox(corners)
                    top_region_boxes = self._top_extend_region_bbox_batch(corners)

                    loss_feat = self._feature_attack_loss(out_clean, out_patch)
                    loss_ocr, loss_det_real, loss_det_top, loss_ocr_real, loss_ocr_top = self._dual_zone_losses(
                        out_patch,
                        patched,
                        target_boxes,
                        top_region_boxes,
                    )
                    loss_tv = self._tv_loss()

                    total = (
                        self.ocr_weight * loss_ocr
                        + self.det_weight * loss_det_top
                        + self.feature_weight * loss_feat
                        + self.tv_weight * loss_tv
                    )

                    self.optimizer.zero_grad(set_to_none=True)
                    total.backward()
                    torch.nn.utils.clip_grad_norm_([self.patch], max_norm=1.0)
                    self.optimizer.step()

                    state.global_step += 1
                    epoch_loss += float(total.item())
                    n_batches += 1

                    pbar.set_postfix(
                        loss=f"{float(total.item()):.4f}",
                        feat=f"{float(loss_feat.item()):.4f}",
                        det_top=f"{float(loss_det_top.item()):.4f}",
                        ocr_top=f"{float(loss_ocr_top.item()):.4f}",
                    )

                    writer.writerow([
                        state.global_step,
                        epoch + 1,
                        float(total.item()),
                        float(loss_feat.item()),
                        float(loss_det_real.item()),
                        float(loss_det_top.item()),
                        float(loss_ocr_real.item()),
                        float(loss_ocr_top.item()),
                        float(loss_ocr.item()),
                        float(loss_tv.item()),
                    ])

                    if state.global_step % save_every == 0:
                        stem = f"patch_step_{state.global_step:07d}"
                        self._save_patch(stem)
                        self.save_checkpoint(state, "latest.pt")

                    if max_steps > 0 and state.global_step >= max_steps:
                        stop = True
                        break

                mean_epoch = epoch_loss / max(n_batches, 1)
                print(f"[backbone-attack] epoch {epoch + 1} mean loss: {mean_epoch:.6f}")
                state.epoch = epoch + 1
                self.save_checkpoint(state, "latest.pt")
                self._save_patch(f"patch_epoch_{state.epoch:04d}")

                if stop:
                    break

        return state


def find_latest_run(prefix: str = "backbone_rtdetr_") -> Optional[Path]:
    runs = Path("runs")
    if not runs.exists():
        return None
    matches = sorted(
        (p for p in runs.glob(f"{prefix}*") if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
    )
    return matches[-1] if matches else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a universal patch by attacking RT-DETR backbone features."
    )
    parser.add_argument("--ccpd-train-csv", default="finetuned_models/train_split.csv")
    parser.add_argument("--model-path", default="finetuned_models/rtdetr_finetuned")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--n-jobs", type=int, default=0)
    parser.add_argument("--preload", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=0, help="Stop after this many global steps (0 disables).")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--feature-weight", type=float, default=1.0)
    parser.add_argument("--det-weight", type=float, default=0.5)
    parser.add_argument("--ocr-weight", type=float, default=1.0)
    parser.add_argument("--tv-weight", type=float, default=0.02)
    parser.add_argument("--border-scale", type=float, default=1.35)
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    parser.add_argument("--backbone-source", choices=["auto", "encoder", "decoder"], default="auto")
    parser.add_argument("--ocr-backend", default="crnn")
    parser.add_argument("--ocr-model-path", default="none")
    parser.add_argument("--impersonation-target", default="SHX8459")
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--continue", dest="resume", action="store_true")
    args = parser.parse_args()

    if args.resume:
        run_dir = find_latest_run()
        if run_dir is None:
            raise RuntimeError("--continue was set but no previous backbone RT-DETR run exists in runs/.")
        ckpt_path = run_dir / "checkpoints" / "latest.pt"
        if not ckpt_path.exists():
            raise RuntimeError(f"--continue found run {run_dir} but missing checkpoint {ckpt_path}")
    else:
        suffix = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path("runs") / f"backbone_rtdetr_{suffix}"
        ckpt_path = None

    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("RT-DETR Backbone Attack Training")
    print(f"run_dir         : {run_dir}")
    print(f"csv             : {args.ccpd_train_csv}")
    print(f"model_path      : {args.model_path}")
    print(f"device          : {args.device}")
    print(f"batch_size      : {args.eval_batch_size}")
    print(f"epochs          : {args.epochs}")
    print(f"feature_weight  : {args.feature_weight}")
    print(f"det_weight      : {args.det_weight}")
    print(f"ocr_weight      : {args.ocr_weight}")
    print(f"tv_weight       : {args.tv_weight}")
    print(f"backbone_source : {args.backbone_source}")
    print(f"ocr_backend     : {args.ocr_backend}")
    print(f"ocr_model_path  : {args.ocr_model_path}")
    print(f"target_plate    : {args.impersonation_target}")
    print("=" * 72)

    trainer = RTDETRBackbonePatchTrainer(
        csv_path=args.ccpd_train_csv,
        model_path=args.model_path,
        run_dir=run_dir,
        device=args.device,
        batch_size=args.eval_batch_size,
        n_jobs=args.n_jobs,
        preload=args.preload,
        border_scale=args.border_scale,
        feature_weight=args.feature_weight,
        det_weight=args.det_weight,
        ocr_weight=args.ocr_weight,
        tv_weight=args.tv_weight,
        lr=args.lr,
        conf_threshold=args.conf_threshold,
        backbone_source=args.backbone_source,
        ocr_backend=args.ocr_backend,
        ocr_model_path=args.ocr_model_path,
        impersonation_target=args.impersonation_target,
    )

    state = TrainState()
    if ckpt_path is not None:
        state = trainer.load_checkpoint(ckpt_path)
        print(
            f"[backbone-attack] resumed from {ckpt_path} "
            f"(epoch={state.epoch}, step={state.global_step})"
        )

    end_state = trainer.train(
        epochs=args.epochs,
        max_steps=args.max_steps,
        save_every=args.save_every,
        start=state,
    )

    final_ckpt = trainer.save_checkpoint(end_state, "latest.pt")
    trainer._save_patch("patch_final")
    print("\nTraining complete")
    print(f"Final checkpoint: {final_ckpt}")
    print(f"Final step      : {end_state.global_step}")


if __name__ == "__main__":
    main()
