#!/usr/bin/env python3
"""
eval_physical.py

Evaluate adversarial patches over the physical-world test set
(control_plate_corners.csv) at ORIGINAL image resolution.

Unlike evaluator.py (which feeds 384×384 letterboxed images), this script
loads each HEIC/image at full resolution, applies the patch at full resolution
using the labelled corners, then hands the full-res tensor to each backend.
Backends (YOLOv8, RT-DETR, FasterRCNN) all handle their own internal resizing.

Usage
-----
python eval_physical.py \\
    --pairs patches/patch_fasterrcnn__dtrb_epoch_0059.png:fasterrcnn:weights/model.pt \\
            patches/patch_rtdetr__trocr_epoch_0079.png:rtdetr:weights/rtdetr-v2-license-plates \\
            patches/patch_yolov8__crnn_epoch_0099.png:yolov8:weights/lp_yolov8.pt \\
    --output results/physical/

Pair format:  patch_file:backend_name:backend_path[:key=val,key=val]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
import kornia.geometry as K
from PIL import Image
from tqdm import tqdm

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    print("Warning: pillow-heif not installed; HEIC images may fail to load.")

import re
import torch.nn.functional as F

from detector_backends import DetectorBackend, build_backend
from ocr_backends import OCRBackend, build_ocr_backend
from evaluator import BackendMetrics


# ---------------------------------------------------------------------------
# Patch loading
# ---------------------------------------------------------------------------

def _load_patch(path: str, device: str) -> torch.Tensor:
    """Load a patch PNG/JPG or .pt checkpoint → [3, H, W] float32 in [0, 1]."""
    p = Path(path)
    if p.suffix == ".pt":
        ckpt = torch.load(path, map_location="cpu")
        raw = ckpt.get("patch", ckpt)
        patch = (torch.tanh(raw) * 0.5 + 0.5).clamp(0, 1)
    else:
        img = Image.open(path).convert("RGB")
        patch = T.ToTensor()(img)
    return patch.to(device)


# ---------------------------------------------------------------------------
# Patch application at full resolution
# ---------------------------------------------------------------------------

def _apply_patch(image: torch.Tensor,
                 corners: torch.Tensor,
                 patch: torch.Tensor,
                 border_scale: float = 1.4) -> torch.Tensor:
    """
    Warp patch as a border around the licence plate at original resolution.

    Parameters
    ----------
    image   : [3, H, W] float32 in [0, 1]
    corners : [4, 2] plate corner coordinates in original image space
    patch   : [3, Ph, Pw] float32 in [0, 1]
    """
    device = image.device
    img_h, img_w = image.shape[1], image.shape[2]
    patch_h, patch_w = patch.shape[1], patch.shape[2]

    plate = corners.to(device)                       # [4, 2]
    cx = plate[:, 0].mean()
    cy = plate[:, 1].mean()
    ctr = torch.stack([cx, cy])

    border = (ctr.unsqueeze(0) + (plate - ctr.unsqueeze(0)) * border_scale
              ).unsqueeze(0)                          # [1, 4, 2]
    plate_b = plate.unsqueeze(0)                      # [1, 4, 2]

    src = torch.tensor(
        [[0, 0], [patch_w, 0], [patch_w, patch_h], [0, patch_h]],
        dtype=torch.float32, device=device,
    ).unsqueeze(0)                                    # [1, 4, 2]

    M_border = K.get_perspective_transform(src, border)
    M_plate  = K.get_perspective_transform(src, plate_b)

    batch   = image.unsqueeze(0)
    ones    = torch.ones(1, 1, patch_h, patch_w, device=device)
    dsize   = (img_h, img_w)
    kwargs  = dict(mode="bilinear", padding_mode="zeros", align_corners=True)

    warped        = K.warp_perspective(patch.unsqueeze(0), M_border, dsize, **kwargs)
    w_border_mask = K.warp_perspective(ones,               M_border, dsize, **kwargs)
    w_plate_mask  = K.warp_perspective(ones,               M_plate,  dsize, **kwargs)

    mask = torch.clamp(w_border_mask - w_plate_mask, 0, 1).expand(-1, 3, -1, -1)
    return torch.clamp(batch * (1 - mask) + warped * mask, 0, 1).squeeze(0)


# ---------------------------------------------------------------------------
# IoU helper
# ---------------------------------------------------------------------------

def _iou(a: torch.Tensor, b: torch.Tensor) -> float:
    ix1 = max(a[0].item(), b[0].item())
    iy1 = max(a[1].item(), b[1].item())
    ix2 = min(a[2].item(), b[2].item())
    iy2 = min(a[3].item(), b[3].item())
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1])
    ub = (b[2] - b[0]) * (b[3] - b[1])
    union = ua.item() + ub.item() - inter
    return inter / (union + 1e-8) if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

def preload_images(df: pd.DataFrame, device: str, scale: float = 1.0,
                   num_workers: int = 0) -> List[Tuple]:
    """Load all images and corners to GPU once. Returns list of (image, corners, gt_box)."""
    import os
    from concurrent.futures import ThreadPoolExecutor
    to_tensor = T.ToTensor()
    if num_workers == 0:
        num_workers = os.cpu_count() or 1

    rows = list(df.iterrows())

    def _load_one(args):
        idx, row = args
        try:
            pil_img = Image.open(row["filename"]).convert("RGB")
        except Exception as e:
            print(f"  [warn] could not load {row['filename']}: {e}")
            return idx, None
        if scale != 1.0:
            new_w = int(pil_img.width * scale)
            new_h = int(pil_img.height * scale)
            pil_img = pil_img.resize((new_w, new_h), Image.BILINEAR)
        image = to_tensor(pil_img)
        del pil_img
        corners = torch.tensor([
            [row["p1_x"], row["p1_y"]],
            [row["p2_x"], row["p2_y"]],
            [row["p3_x"], row["p3_y"]],
            [row["p4_x"], row["p4_y"]],
        ], dtype=torch.float32) * scale
        return idx, (image, corners)

    samples = []
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        pbar = tqdm(total=len(rows),
                    desc=f"Preloading images to GPU ({num_workers} threads)")
        # Submit and consume in small batches so at most BATCH
        # CPU tensors exist at any time.
        BATCH = max(num_workers, 64)
        for batch_start in range(0, len(rows), BATCH):
            batch = rows[batch_start:batch_start + BATCH]
            futs = [pool.submit(_load_one, r) for r in batch]
            for f in futs:
                idx, data = f.result()
                if data is not None:
                    image, corners = data
                    corners_gpu = corners.to(device)
                    gt_box = torch.stack([
                        corners_gpu[:, 0].min(), corners_gpu[:, 1].min(),
                        corners_gpu[:, 0].max(), corners_gpu[:, 1].max(),
                    ])
                    samples.append((image.to(device), corners_gpu, gt_box))
                pbar.update(1)
            del futs  # free all Futures + their cached CPU tensors
        pbar.close()

    return samples


def _plate_text_matches(text: str, expected: str) -> bool:
    """Case-insensitive alphanumeric comparison."""
    norm = lambda s: re.sub(r"[^A-Za-z0-9]", "", s).upper()
    return norm(text) == norm(expected)


def _ocr_crop(image: torch.Tensor, corners: torch.Tensor,
              crop_size: Tuple) -> torch.Tensor:
    """Axis-aligned bbox crop + resize for OCR. image: [C,H,W], corners: [4,2]."""
    h, w = image.shape[1], image.shape[2]
    x1 = corners[:, 0].min().long().clamp(0, w - 1)
    y1 = corners[:, 1].min().long().clamp(0, h - 1)
    x2 = corners[:, 0].max().long().clamp(0, w)
    y2 = corners[:, 1].max().long().clamp(0, h)
    crop = image[:, y1:y2, x1:x2].unsqueeze(0)   # [1, C, ch, cw]
    th, tw = crop_size
    if tw is None:
        ch, cw = crop.shape[2], crop.shape[3]
        tw = max(1, int(th * cw / ch))
    return F.interpolate(crop, (th, tw), mode="bilinear", align_corners=False)


def evaluate_one(backend: DetectorBackend,
                 samples: List[Tuple],
                 patch: Optional[torch.Tensor],
                 patch_name: str,
                 device: str,
                 iou_threshold: float = 0.5,
                 ocr_backend: Optional[OCRBackend] = None,
                 expected_plate: str = "",
                 impersonation_target: str = "") -> BackendMetrics:
    """Evaluate a single backend × patch combo over preloaded samples."""
    m = BackendMetrics(name=backend.name, patch_name=patch_name)
    backend.eval()
    if ocr_backend is not None:
        ocr_backend.ensure_loaded()
        ocr_backend.eval()

    with torch.no_grad():
        for image, corners, gt_box in tqdm(samples,
                                           desc=f"  {backend.name} | {patch_name}",
                                           leave=False):
            if patch is not None:
                image = _apply_patch(image, corners, patch)

            t0   = time.perf_counter()
            dets = backend.predict(image)
            m.latency_ms.append((time.perf_counter() - t0) * 1000)

            m.num_images       += 1
            m.total_detections += len(dets)

            best_iou = best_conf = 0.0
            for det in dets:
                iou = _iou(det.box, gt_box)
                if iou > best_iou:
                    best_iou  = iou
                    best_conf = det.confidence

            m.iou_values.append(best_iou)
            m.conf_values.append(best_conf)
            if best_iou >= iou_threshold:
                m.true_positives += 1
            else:
                m.false_negatives += 1

            # OCR evaluation (on labeled corners regardless of detection)
            if ocr_backend is not None:
                if not dets:
                    m.ocr_no_detection += 1
                else:
                    crop = _ocr_crop(image, corners, ocr_backend.ocr_crop_size)
                    result = ocr_backend.predict(crop.squeeze(0))
                    text = result.text or ""
                    if expected_plate and _plate_text_matches(text, expected_plate):
                        m.ocr_correct += 1
                    elif impersonation_target and _plate_text_matches(text, impersonation_target):
                        m.ocr_impersonation += 1
                    else:
                        m.ocr_misread += 1

    return m


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _parse_pair(spec: str) -> Tuple[str, str, str, Optional[str], Optional[str]]:
    """Parse pair spec.

    Formats:
      patch:det:det_weights
      patch:det:det_weights:ocr:ocr_weights
    """
    parts = spec.split(":")
    if len(parts) == 5:
        patch_path, det_name, det_weights, ocr_name, ocr_weights = parts
        return patch_path, det_name, det_weights, ocr_name, ocr_weights
    elif len(parts) >= 3:
        patch_path, det_name, det_weights = parts[0], parts[1], parts[2]
        return patch_path, det_name, det_weights, None, None
    raise ValueError(
        f"Invalid pair spec '{spec}'. "
        "Expected: patch:det:det_weights  or  patch:det:det_weights:ocr:ocr_weights"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

import os as _os

# Default weight paths keyed by backend name
_DEFAULT_WEIGHTS: Dict[str, str] = {
    "fasterrcnn":   "weights/model.pt",
    "rtdetr":       "weights/rtdetr-v2-license-plate",
    "yolov8":       "weights/lp_yolov8.pt",
    "yolov11":      "weights/yolov11s-license-plate.pt",
    "yolo-v9-384":  "none",
}

_DEFAULT_OCR_WEIGHTS: Dict[str, str] = {
    "trocr":        "weights/trocr_small_finetuned.pt",
    "cct":          _os.path.expanduser("~/.cache/fast-plate-ocr/cct-s-v1-global-model/cct_s_v1_global.onnx"),
    "lprnet":       "weights/lprnet_deployable_onnx_v1.1/us_lprnet_patched.onnx",
    "doctr-vitstr": "weights/vitstr_small_finetuned.pt",
}


def _infer_backend(patch_path: str) -> Tuple[str, str]:
    """Infer detector backend name and default weights from a patch path.

    Looks for known backend names in the filename and parent directory.
    Returns (backend_name, weights_path).
    """
    search = str(patch_path)
    stem   = Path(patch_path).stem          # e.g. patch_rtdetr_best

    # Try progressively longer token combinations from the filename
    parts = stem.split("_")
    if len(parts) >= 2 and parts[0] == "patch":
        for end in range(len(parts), 1, -1):
            candidate = "_".join(parts[1:end])
            if candidate in _DEFAULT_WEIGHTS:
                return candidate, _DEFAULT_WEIGHTS[candidate]

    # Fall back: scan full path string for any known backend name
    for name, weights in _DEFAULT_WEIGHTS.items():
        if name in search:
            return name, weights

    raise ValueError(
        f"Cannot infer detector backend from '{patch_path}'. "
        f"Known backends: {list(_DEFAULT_WEIGHTS)}. "
        "Use --pairs patch:det:det_weights:ocr:ocr_weights instead."
    )


def _infer_ocr_backend(patch_path: str) -> Optional[Tuple[str, str]]:
    """Try to infer OCR backend from patch path. Returns (name, weights) or None."""
    search = str(patch_path)
    for name, weights in _DEFAULT_OCR_WEIGHTS.items():
        if name in search:
            return name, weights
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate patches on physical-world images at original resolution",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--pairs", nargs="+", default=[],
        metavar="patch:backend:path[:opts]",
        help="Explicit patch/backend/weights triplets.",
    )
    parser.add_argument(
        "--patches", nargs="+", default=[],
        metavar="patch.png",
        help="Bare patch paths; backend and weights are inferred from the filename.",
    )
    parser.add_argument(
        "--csv", default="control_plate_corners.csv",
        help="Corners CSV (default: control_plate_corners.csv)",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--gpu-preload", action="store_true",
                        help="Preload all images to GPU memory once before evaluation.")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Threads for parallel image loading (default: nproc).")
    parser.add_argument("--scale", type=float, default=0.5,
                        help="Resize images by this factor before evaluation (default: 0.5).")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--expected-plate", default="VRJ7774",
                        help="Correct plate text for OCR categorisation (default: VRJ7774).")
    parser.add_argument("--impersonation-target", default="VJJ7744",
                        help="Impersonation target for OCR categorisation (default: VJJ7744).")
    parser.add_argument("--output", default="results/",
                        help="Output directory (default: results/)")
    args = parser.parse_args()

    if not args.pairs and not args.patches:
        parser.error("Provide at least one of --pairs or --patches.")

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[eval_physical] device={args.device}  csv={args.csv}")

    df = pd.read_csv(args.csv)
    print(f"[eval_physical] {len(df)} images loaded from {args.csv}")

    # Build pairs list: (patch_path, det_name, det_weights, ocr_name, ocr_weights)
    pairs: List[Tuple] = [_parse_pair(s) for s in args.pairs]
    for patch_path in args.patches:
        det_name, det_weights = _infer_backend(patch_path)
        ocr_info = _infer_ocr_backend(patch_path)
        ocr_name, ocr_weights = ocr_info if ocr_info else (None, None)
        print(f"[eval_physical] Inferred det='{det_name}' ocr='{ocr_name}' for {patch_path}")
        pairs.append((patch_path, det_name, det_weights, ocr_name, ocr_weights))

    # Build unique detector backends
    seen_det: Dict[Tuple, DetectorBackend] = {}
    for _, bname, bpath, _, _ in pairs:
        key = (bname, bpath)
        if key not in seen_det:
            seen_det[key] = build_backend(bname, bpath, device=args.device)

    # Build unique OCR backends
    seen_ocr: Dict[Tuple, Optional[OCRBackend]] = {}
    for _, _, _, ocr_name, ocr_weights in pairs:
        if ocr_name is None:
            continue
        key = (ocr_name, ocr_weights)
        if key not in seen_ocr:
            ocr = build_ocr_backend(ocr_name, ocr_weights or "none", device=args.device)
            ocr.load()
            ocr.freeze()
            seen_ocr[key] = ocr

    # Load patches — associate each with its det+ocr backend
    # patch_entries: list of (patch_name, patch_tensor, det_key, ocr_key)
    patch_entries: List[Tuple] = [("clean", None, None, None)]
    for patch_path, det_name, det_weights, ocr_name, ocr_weights in pairs:
        name   = Path(patch_path).stem
        tensor = _load_patch(patch_path, args.device)
        det_key = (det_name, det_weights)
        ocr_key = (ocr_name, ocr_weights) if ocr_name else None
        patch_entries.append((name, tensor, det_key, ocr_key))
        print(f"[eval_physical] Loaded patch '{name}'  det={det_name}  ocr={ocr_name}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = preload_images(df, args.device, args.scale, args.num_workers) if args.gpu_preload else None

    # Evaluate: for clean run all detectors; for each patch use its paired detector
    all_results: List[BackendMetrics] = []

    # Clean baseline — run every detector, no OCR
    eval_data = samples if samples is not None else preload_images(df, args.device, args.scale, args.num_workers)
    for det_key, backend in seen_det.items():
        backend.ensure_loaded()
        backend.freeze()
        m = evaluate_one(backend, eval_data, None, "clean",
                         device=args.device, iou_threshold=args.iou_threshold)
        all_results.append(m)
        print(f"  {m.summary()}")

    # Patch evaluations — use paired detector and OCR
    for patch_name, patch_tensor, det_key, ocr_key in patch_entries:
        if patch_name == "clean":
            continue
        backend = seen_det[det_key]
        ocr_backend = seen_ocr.get(ocr_key) if ocr_key else None
        eval_data = samples if samples is not None else preload_images(df, args.device, args.scale, args.num_workers)
        print(f"\n── Patch: {patch_name} | det={det_key[0]} | ocr={ocr_key[0] if ocr_key else 'none'} ──")
        m = evaluate_one(
            backend, eval_data, patch_tensor, patch_name,
            device=args.device, iou_threshold=args.iou_threshold,
            ocr_backend=ocr_backend,
            expected_plate=args.expected_plate,
            impersonation_target=args.impersonation_target,
        )
        all_results.append(m)
        print(f"  {m.summary()}")

    # Save outputs via evaluator helpers
    from evaluator import DetectorEvaluator
    ev = DetectorEvaluator.__new__(DetectorEvaluator)   # no __init__ needed
    ev.save_csv(all_results,           str(out_dir / "metrics.csv"))
    ev.save_json(all_results,          str(out_dir / "metrics.json"))
    ev.save_summary_table(all_results, str(out_dir / "summary_table.txt"))
    ev.save_bar_chart(all_results,     str(out_dir / "bar_chart.png"))
    ev.save_matrix_heatmaps(all_results, str(out_dir))


if __name__ == "__main__":
    main()
