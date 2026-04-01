"""
evaluate_finetuned.py

Comprehensive evaluation of finetuned detector and OCR models.
Produces paper-ready metrics: AP/mAP for detectors, CRR/LPRR for OCR.

Auto-discovers models in the finetuned_models/ directory and uses val_split.csv.

Usage:
    python evaluate_finetuned.py --finetuned-models weights/finetuned/ --device cuda --output results/eval/
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from difflib import SequenceMatcher
from PIL import Image
from tqdm import tqdm

from detector_backends import DetectorBackend, Detection, build_backend
from ocr_backends import build_ocr_backend, OCRBackend


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_val_split(csv_path: str) -> List[dict]:
    """
    Load validation records from val_split.csv (created by finetune_all_models.py).

    CSV columns: image_path, x1, y1, x2, y2, label

    Returns list of dicts with keys:
        - image: Path to image file
        - box: [x1, y1, x2, y2] in pixel coords
        - text: ground truth plate text (uppercased)
    """
    records = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                img_path = Path(row["image_path"])
                if not img_path.exists():
                    print(f"[warn] image not found: {img_path}")
                    continue
                x1, y1, x2, y2 = float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])
                label = row["label"].strip().upper()

                records.append({
                    "image": img_path,
                    "box": torch.tensor([x1, y1, x2, y2], dtype=torch.float32),
                    "text": label,
                })
            except (ValueError, KeyError) as e:
                print(f"[warn] skipping row: {e}")
                continue

    return records


def discover_finetuned_models(finetuned_dir: Path) -> Tuple[List[str], List[str]]:
    """
    Scan finetuned_dir for checkpoints and determine which models are available.

    Detectors (by checkpoint name):
        - fasterrcnn_finetuned.pt → "fasterrcnn"
        - rtdetr_finetuned/ → "rtdetr"
        - owlvit_finetuned/ → "owlvit"
        - yolo608_finetuned.pt → "yolo-v9-608"

    OCR (by checkpoint name):
        - lprnet_finetuned.pt → "lprnet"
        - trocr_small_finetuned.pt → "trocr"
        - vitstr_small_finetuned.pt → "doctr-vitstr"
        - cct_s_finetuned.pt → "cct"

    Returns (detector_names, ocr_names)
    """
    detectors = []
    ocr_models = []

    # Check for detector checkpoints
    if (finetuned_dir / "fasterrcnn_finetuned.pt").exists():
        detectors.append("fasterrcnn")
    if (finetuned_dir / "rtdetr_finetuned").exists():
        detectors.append("rtdetr")
    if (finetuned_dir / "owlvit_finetuned").exists():
        detectors.append("owlvit")
    if (finetuned_dir / "yolo608_finetuned.pt").exists():
        detectors.append("yolo-v9-608")

    # Check for OCR checkpoints
    if (finetuned_dir / "lprnet_finetuned.pt").exists():
        ocr_models.append("lprnet")
    if (finetuned_dir / "trocr_small_finetuned.pt").exists():
        ocr_models.append("trocr")
    if (finetuned_dir / "vitstr_small_finetuned.pt").exists():
        ocr_models.append("doctr-vitstr")
    if (finetuned_dir / "cct_s_finetuned.pt").exists():
        ocr_models.append("cct")

    return detectors, ocr_models


def resolve_checkpoint_path(model_name: str, finetuned_dir: Path) -> str:
    """
    Map model name to its checkpoint path in finetuned_dir.
    Raises FileNotFoundError if checkpoint doesn't exist.
    """
    checkpoint_map = {
        "fasterrcnn": "fasterrcnn_finetuned.pt",
        "rtdetr": "rtdetr_finetuned",
        "owlvit": "owlvit_finetuned",
        "yolo-v9-608": "yolo608_finetuned.pt",
        "lprnet": "lprnet_finetuned.pt",
        "trocr": "trocr_small_finetuned.pt",
        "doctr-vitstr": "vitstr_small_finetuned.pt",
        "cct": "cct_s_finetuned.pt",
    }

    if model_name not in checkpoint_map:
        raise ValueError(f"Unknown model: {model_name}")

    checkpoint_path = finetuned_dir / checkpoint_map[model_name]
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    return str(checkpoint_path)


# ─────────────────────────────────────────────────────────────────────────────
# Detector Metrics (AP / mAP)
# ─────────────────────────────────────────────────────────────────────────────

def _iou(box_a: torch.Tensor, box_b: torch.Tensor) -> float:
    """Compute IoU between two boxes in [x1, y1, x2, y2] format."""
    ix1 = max(box_a[0].item(), box_b[0].item())
    iy1 = max(box_a[1].item(), box_b[1].item())
    ix2 = min(box_a[2].item(), box_b[2].item())
    iy2 = min(box_a[3].item(), box_b[3].item())
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    ub = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = ua.item() + ub.item() - inter
    return inter / (union + 1e-8) if union > 0 else 0.0


def compute_ap(confidences: List[float], ious: List[float],
               iou_threshold: float = 0.5) -> float:
    """
    Compute AP for a single IoU threshold.

    Args:
        confidences: list of detection confidences (sorted descending)
        ious: list of corresponding IoU values with GT
        iou_threshold: IoU threshold for positive

    Returns:
        AP (area under PR curve, 11-point interpolation)
    """
    if not confidences:
        return 0.0

    # Match detections to GT (single GT per image)
    n_images = len(ious)
    tp = np.array([1 if iou >= iou_threshold else 0 for iou in ious], dtype=np.float32)
    fp = 1 - tp

    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    recall = tp_cumsum / n_images
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)

    # 11-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap += p / 11.0

    return float(ap)


def compute_detector_metrics(backend: DetectorBackend, records: List[dict],
                             device: str = "cpu") -> dict:
    """
    Evaluate detector: compute AP@0.5, AP@0.75, mAP@0.5:0.95, etc.
    """
    transform = T.ToTensor()
    confidences_list = []
    ious_list = []
    latencies = []
    mean_iou_values = []

    backend.eval()
    with torch.no_grad():
        for record in tqdm(records, desc=f"  {backend.name}", leave=False):
            try:
                img = Image.open(record["image"]).convert("RGB")
            except Exception as e:
                print(f"[warn] could not load {record['filename']}: {e}")
                continue

            img_tensor = transform(img).to(device)
            gt_box = record["box"].to(device)

            t0 = time.perf_counter()
            dets = backend.predict(img_tensor)
            latency_ms = (time.perf_counter() - t0) * 1000
            latencies.append(latency_ms)

            # Find best detection by IoU
            best_iou = 0.0
            best_conf = 0.0
            for det in dets:
                iou = _iou(det.box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_conf = det.confidence

            confidences_list.append(best_conf)
            ious_list.append(best_iou)
            mean_iou_values.append(best_iou)

    # Sort by confidence
    sorted_idx = np.argsort(-np.array(confidences_list))
    confidences_sorted = [confidences_list[i] for i in sorted_idx]
    ious_sorted = [ious_list[i] for i in sorted_idx]

    # Compute AP at different IoU thresholds
    ap_50 = compute_ap(confidences_sorted, ious_sorted, iou_threshold=0.50)
    ap_75 = compute_ap(confidences_sorted, ious_sorted, iou_threshold=0.75)

    # mAP: average AP across 0.5 to 0.95 in steps of 0.05
    ap_values = []
    for iou_t in np.arange(0.5, 0.95 + 0.05, 0.05):
        ap = compute_ap(confidences_sorted, ious_sorted, iou_threshold=float(iou_t))
        ap_values.append(ap)
    map_50_95 = float(np.mean(ap_values))

    # Recall at IoU=0.5
    n_detected = sum(1 for iou in ious_list if iou >= 0.5)
    recall_50 = n_detected / len(ious_list) if ious_list else 0.0

    # Precision at confidence=0.5
    confident_dets = [i for i, conf in enumerate(confidences_list) if conf >= 0.5]
    correct_confident = sum(1 for i in confident_dets if ious_list[i] >= 0.5)
    precision_50 = correct_confident / len(confident_dets) if confident_dets else 0.0

    return {
        "ap_50": ap_50,
        "ap_75": ap_75,
        "map_50_95": map_50_95,
        "recall_50": recall_50,
        "precision_50": precision_50,
        "mean_iou": float(np.mean(mean_iou_values)) if mean_iou_values else 0.0,
        "latency_mean_ms": float(np.mean(latencies)) if latencies else 0.0,
        "latency_p95_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "n_images": len(records),
        "ious_list": ious_list,
        "confidences_list": confidences_list,
    }


# ─────────────────────────────────────────────────────────────────────────────
# OCR Metrics (CRR / LPRR / Edit Distance)
# ─────────────────────────────────────────────────────────────────────────────

def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def _compute_char_error_rate(pred: str, gt: str) -> float:
    """
    Compute character-level accuracy.
    Standard: per-position match up to min length, then length penalty.
    """
    min_len = min(len(pred), len(gt))
    correct = sum(1 for i in range(min_len) if pred[i] == gt[i])
    # Penalize length difference
    correct -= abs(len(pred) - len(gt))
    correct = max(0, correct)
    total = max(len(pred), len(gt))
    return correct / total if total > 0 else (1.0 if pred == gt else 0.0)


def compute_ocr_metrics(backend: OCRBackend, records: List[dict],
                        device: str = "cpu") -> dict:
    """
    Evaluate OCR: compute CRR, LPRR, edit distance.
    """
    transform = T.ToTensor()
    latencies = []
    crr_values = []
    lprr_values = []
    edit_distances = []

    backend.eval()
    with torch.no_grad():
        for record in tqdm(records, desc=f"  {backend.name}", leave=False):
            try:
                img = Image.open(record["image"]).convert("RGB")
            except Exception as e:
                print(f"[warn] could not load {record['filename']}: {e}")
                continue

            # Crop to GT box
            x1, y1, x2, y2 = [int(v) for v in record["box"].tolist()]
            img_crop = img.crop((x1, y1, x2, y2))

            # Convert to tensor (model will handle resize)
            crop_tensor = transform(img_crop).to(device)

            gt_text = record["text"]

            t0 = time.perf_counter()
            ocr_result = backend.predict(crop_tensor)
            latency_ms = (time.perf_counter() - t0) * 1000
            latencies.append(latency_ms)

            pred_text = ocr_result.text.strip().upper()

            # LPRR: exact match
            lprr = 1.0 if pred_text == gt_text else 0.0
            lprr_values.append(lprr)

            # CRR: character-level accuracy
            crr = _compute_char_error_rate(pred_text, gt_text)
            crr_values.append(crr)

            # Edit distance (normalized)
            edit_dist = _levenshtein_distance(pred_text, gt_text)
            max_len = max(len(pred_text), len(gt_text))
            norm_edit = edit_dist / max_len if max_len > 0 else 0.0
            edit_distances.append(norm_edit)

    return {
        "crr": float(np.mean(crr_values)) if crr_values else 0.0,
        "lprr": float(np.mean(lprr_values)) if lprr_values else 0.0,
        "mean_edit_distance": float(np.mean(edit_distances)) if edit_distances else 0.0,
        "latency_mean_ms": float(np.mean(latencies)) if latencies else 0.0,
        "latency_p95_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "n_plates": len(records),
        "crr_values": crr_values,
        "lprr_values": lprr_values,
    }


# ─────────────────────────────────────────────────────────────────────────────
# End-to-End Pipeline Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def compute_pipeline_metrics(detector: DetectorBackend, ocr_backend: OCRBackend,
                             records: List[dict], device: str = "cpu") -> dict:
    """
    Evaluate detector + OCR pipeline end-to-end.

    For each image:
    1. Run detector to find plate
    2. If detection succeeds (IoU ≥ 0.5), crop and run OCR
    3. Track: detection failures, OCR failures (given good detection), full successes
    """
    transform = T.ToTensor()
    n_det_failures = 0
    n_ocr_failures = 0  # given good detection
    n_full_successes = 0
    pipeline_latencies = []
    lprr_values = []

    detector.eval()
    ocr_backend.eval()

    with torch.no_grad():
        for record in tqdm(records, desc=f"  {detector.name} + {ocr_backend.name}", leave=False):
            try:
                img = Image.open(record["image"]).convert("RGB")
            except Exception:
                n_det_failures += 1
                continue

            img_tensor = transform(img).to(device)
            gt_box = record["box"].to(device)
            gt_text = record["text"]

            t0 = time.perf_counter()

            # Step 1: Detect
            dets = detector.predict(img_tensor)
            best_iou = 0.0
            best_det_box = None
            for det in dets:
                iou = _iou(det.box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_det_box = det.box

            # Detection failure
            if best_iou < 0.5:
                n_det_failures += 1
                pipeline_latencies.append((time.perf_counter() - t0) * 1000)
                lprr_values.append(0.0)
                continue

            # Step 2: OCR on detected box
            try:
                x1, y1, x2, y2 = [int(v) for v in best_det_box.tolist()]
                img_crop = img.crop((x1, y1, x2, y2))
                crop_tensor = transform(img_crop).to(device)
                ocr_result = ocr_backend.predict(crop_tensor)
                pred_text = ocr_result.text.strip().upper()

                # Full pipeline success
                if pred_text == gt_text:
                    n_full_successes += 1
                    lprr_values.append(1.0)
                else:
                    n_ocr_failures += 1
                    lprr_values.append(0.0)
            except Exception:
                n_ocr_failures += 1
                lprr_values.append(0.0)

            pipeline_latencies.append((time.perf_counter() - t0) * 1000)

    n_images = len(records)
    pipeline_success_rate = n_full_successes / n_images if n_images > 0 else 0.0
    det_failure_rate = n_det_failures / n_images if n_images > 0 else 0.0
    ocr_failure_given_det = n_ocr_failures / (n_images - n_det_failures) if (n_images - n_det_failures) > 0 else 0.0

    return {
        "pipeline_success": pipeline_success_rate,
        "det_failures": n_det_failures,
        "ocr_failures": n_ocr_failures,
        "full_successes": n_full_successes,
        "n_images": n_images,
        "det_failure_rate": det_failure_rate,
        "ocr_failure_given_det": ocr_failure_given_det,
        "latency_mean_ms": float(np.mean(pipeline_latencies)) if pipeline_latencies else 0.0,
        "latency_p95_ms": float(np.percentile(pipeline_latencies, 95)) if pipeline_latencies else 0.0,
        "lprr_values": lprr_values,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Results Container & Saving
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalResults:
    model_name: str
    model_type: str  # "detector" or "ocr"
    metrics: dict

    def to_flat_dict(self) -> dict:
        d = {
            "model": self.model_name,
            "type": self.model_type,
        }
        d.update(self.metrics)
        return d


def save_results_csv(results: List[EvalResults], path: str) -> None:
    rows = [r.to_flat_dict() for r in results]
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"[eval] Results → {path}")


def save_results_json(results: List[EvalResults], path: str) -> None:
    data = []
    for r in results:
        d = r.to_flat_dict()
        # Add per-image/plate detail lists
        if "ious_list" in r.metrics:
            d["ious_list"] = r.metrics["ious_list"]
        if "confidences_list" in r.metrics:
            d["confidences_list"] = r.metrics["confidences_list"]
        if "crr_values" in r.metrics:
            d["crr_values"] = r.metrics["crr_values"]
        if "lprr_values" in r.metrics:
            d["lprr_values"] = r.metrics["lprr_values"]
        data.append(d)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[eval] Full results → {path}")


def save_summary_table(results: List[EvalResults], path: str) -> None:
    """Print and save summary table in three sections: detectors, OCR, and pipeline."""
    detectors = [r for r in results if r.model_type == "detector"]
    ocr_models = [r for r in results if r.model_type == "ocr"]
    pipelines = [r for r in results if r.model_type == "pipeline"]

    lines = []

    if detectors:
        col = 12
        hdr = (
            f"{'Model':<20} {'AP@0.5':>{col}} {'AP@0.75':>{col}} "
            f"{'mAP@0.5:0.95':>{col}} {'mIoU':>{col}} "
            f"{'Latency(ms)':>{col}} {'P95(ms)':>{col}}"
        )
        sep = "─" * len(hdr)
        lines.append("\nDETECTORS")
        lines.append(sep)
        lines.append(hdr)
        lines.append(sep)
        for r in detectors:
            lines.append(
                f"{r.model_name:<20} "
                f"{r.metrics['ap_50']:>{col}.4f} "
                f"{r.metrics['ap_75']:>{col}.4f} "
                f"{r.metrics['map_50_95']:>{col}.4f} "
                f"{r.metrics['mean_iou']:>{col}.4f} "
                f"{r.metrics['latency_mean_ms']:>{col}.2f} "
                f"{r.metrics['latency_p95_ms']:>{col}.2f}"
            )
        lines.append(sep)

    if ocr_models:
        col = 12
        hdr = (
            f"{'Model':<20} {'CRR':>{col}} {'LPRR':>{col}} "
            f"{'Edit Dist':>{col}} {'Latency(ms)':>{col}} {'P95(ms)':>{col}}"
        )
        sep = "─" * len(hdr)
        lines.append("\nOCR")
        lines.append(sep)
        lines.append(hdr)
        lines.append(sep)
        for r in ocr_models:
            lines.append(
                f"{r.model_name:<20} "
                f"{r.metrics['crr']:>{col}.4f} "
                f"{r.metrics['lprr']:>{col}.4f} "
                f"{r.metrics['mean_edit_distance']:>{col}.4f} "
                f"{r.metrics['latency_mean_ms']:>{col}.2f} "
                f"{r.metrics['latency_p95_ms']:>{col}.2f}"
            )
        lines.append(sep)

    if pipelines:
        col = 12
        hdr = (
            f"{'Pipeline':<30} {'Success':>{col}} {'Det Fail':>{col}} "
            f"{'OCR Fail|Det':>{col}} {'Latency(ms)':>{col}}"
        )
        sep = "─" * len(hdr)
        lines.append("\nEND-TO-END PIPELINE")
        lines.append(sep)
        lines.append(hdr)
        lines.append(sep)
        for r in pipelines:
            pipeline_name = f"{r.model_name}"
            lines.append(
                f"{pipeline_name:<30} "
                f"{r.metrics['pipeline_success']:>{col}.4f} "
                f"{r.metrics['det_failure_rate']:>{col}.4f} "
                f"{r.metrics['ocr_failure_given_det']:>{col}.4f} "
                f"{r.metrics['latency_mean_ms']:>{col}.2f}"
            )
        lines.append(sep)

    text = "\n".join(lines)
    print(text)
    with open(path, "w") as f:
        f.write(text + "\n")
    print(f"[eval] Summary → {path}")


def plot_detector_pr_curves(results: List[EvalResults], path: str) -> None:
    """Plot precision-recall curves for all detectors at IoU=0.5."""
    detectors = [r for r in results if r.model_type == "detector"]
    if not detectors:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for r in detectors:
        confs = np.array(r.metrics.get("confidences_list", []))
        ious = np.array(r.metrics.get("ious_list", []))

        if len(confs) == 0:
            continue

        # Sort by confidence
        idx = np.argsort(-confs)
        confs = confs[idx]
        ious = ious[idx]

        # Compute P-R curve at IoU=0.5
        tp = (ious >= 0.5).astype(np.float32)
        fp = 1 - tp
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        n_gt = len(ious)
        recall = tp_cumsum / n_gt
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)

        ap = r.metrics["ap_50"]
        ax.plot(recall, precision, marker="o", markersize=3, label=f"{r.model_name} (AP={ap:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves (IoU=0.5)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[eval] PR curves → {path}")


def plot_detector_metrics_bar(results: List[EvalResults], path: str) -> None:
    """Bar chart: AP@0.5, AP@0.75, mAP across detectors."""
    detectors = [r for r in results if r.model_type == "detector"]
    if not detectors:
        return

    names = [r.model_name for r in detectors]
    ap_50 = [r.metrics["ap_50"] for r in detectors]
    ap_75 = [r.metrics["ap_75"] for r in detectors]
    map_val = [r.metrics["map_50_95"] for r in detectors]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, ap_50, width, label="AP@0.5", alpha=0.8)
    ax.bar(x, ap_75, width, label="AP@0.75", alpha=0.8)
    ax.bar(x + width, map_val, width, label="mAP@0.5:0.95", alpha=0.8)

    ax.set_xlabel("Model")
    ax.set_ylabel("AP")
    ax.set_title("Detector Performance")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[eval] Detector bar chart → {path}")


def plot_ocr_metrics_bar(results: List[EvalResults], path: str) -> None:
    """Bar chart: CRR, LPRR, edit distance across OCR models."""
    ocr_models = [r for r in results if r.model_type == "ocr"]
    if not ocr_models:
        return

    names = [r.model_name for r in ocr_models]
    crr = [r.metrics["crr"] for r in ocr_models]
    lprr = [r.metrics["lprr"] for r in ocr_models]
    edit_dist = [1 - r.metrics["mean_edit_distance"] for r in ocr_models]  # Invert for "accuracy"

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, crr, width, label="CRR", alpha=0.8)
    ax.bar(x, lprr, width, label="LPRR", alpha=0.8)
    ax.bar(x + width, edit_dist, width, label="1 - Edit Dist", alpha=0.8)

    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy")
    ax.set_title("OCR Performance")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[eval] OCR bar chart → {path}")


def plot_pipeline_metrics_bar(results: List[EvalResults], path: str) -> None:
    """Bar chart: end-to-end pipeline success, detection failure, OCR failure rates."""
    pipelines = [r for r in results if r.model_type == "pipeline"]
    if not pipelines:
        return

    names = [r.model_name for r in pipelines]
    success = [r.metrics["pipeline_success"] for r in pipelines]
    det_fail = [r.metrics["det_failure_rate"] for r in pipelines]
    ocr_fail = [r.metrics["ocr_failure_given_det"] for r in pipelines]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, success, width, label="Full Success", alpha=0.8, color="green")
    ax.bar(x, det_fail, width, label="Detection Failure", alpha=0.8, color="red")
    ax.bar(x + width, ocr_fail, width, label="OCR Failure (given det)", alpha=0.8, color="orange")

    ax.set_xlabel("Pipeline")
    ax.set_ylabel("Rate")
    ax.set_title("End-to-End Pipeline Performance")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[eval] Pipeline bar chart → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate finetuned detector and OCR models using val_split.csv"
    )
    parser.add_argument("--finetuned-models", required=True,
                        help="Path to finetuned_models/ directory (output of finetune_all_models.py)")
    parser.add_argument("--device", default="cuda",
                        help="Device: cuda or cpu")
    parser.add_argument("--output", default="results/eval_finetuned/",
                        help="Output directory")
    args = parser.parse_args()

    finetuned_dir = Path(args.finetuned_models)
    if not finetuned_dir.exists():
        print(f"[error] Directory not found: {finetuned_dir}")
        return 1

    # Look for val_split.csv
    val_csv = finetuned_dir / "val_split.csv"
    if not val_csv.exists():
        print(f"[error] val_split.csv not found in {finetuned_dir}")
        print(f"       Expected: {val_csv}")
        return 1

    # Auto-discover available models
    print(f"[eval] Scanning {finetuned_dir} for finetuned models...")
    detectors, ocr_models = discover_finetuned_models(finetuned_dir)

    if not detectors and not ocr_models:
        print("[error] No finetuned models found!")
        print("        Expected checkpoints: fasterrcnn_finetuned.pt, rtdetr_finetuned/, ...")
        return 1

    print(f"  Detectors: {detectors}")
    print(f"  OCR:       {ocr_models}\n")

    # Ensure device exists
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA not available, falling back to CPU")
        args.device = "cpu"

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load validation records
    print(f"[eval] Loading validation data from {val_csv}")
    records = load_val_split(str(val_csv))
    print(f"[eval] Loaded {len(records)} validation records\n")

    results = []

    # Evaluate detectors
    if detectors:
        print(f"[eval] Evaluating {len(detectors)} detectors...")
        for detector_name in detectors:
            print(f"\n  {detector_name}")
            try:
                checkpoint_path = resolve_checkpoint_path(detector_name, finetuned_dir)
                backend = build_backend(detector_name, checkpoint_path, device=args.device)
                backend.ensure_loaded()
                backend.freeze()

                metrics = compute_detector_metrics(backend, records, device=args.device)
                results.append(EvalResults(detector_name, "detector", metrics))

                print(f"    AP@0.5={metrics['ap_50']:.4f}, mAP={metrics['map_50_95']:.4f}")
            except Exception as e:
                print(f"    [error] {e}")
                traceback.print_exc()

    # Evaluate OCR
    if ocr_models:
        print(f"\n[eval] Evaluating {len(ocr_models)} OCR models...")
        for ocr_name in ocr_models:
            print(f"\n  {ocr_name}")
            try:
                checkpoint_path = resolve_checkpoint_path(ocr_name, finetuned_dir)
                backend = build_ocr_backend(ocr_name, checkpoint_path, device=args.device)
                backend.ensure_loaded()
                backend.freeze()

                metrics = compute_ocr_metrics(backend, records, device=args.device)
                results.append(EvalResults(ocr_name, "ocr", metrics))

                print(f"    CRR={metrics['crr']:.4f}, LPRR={metrics['lprr']:.4f}")
            except Exception as e:
                print(f"    [error] {e}")
                traceback.print_exc()

    # Evaluate pipelines (detector + OCR pairings)
    pipeline_pairings = [
        ("fasterrcnn", "lprnet"),
        ("rtdetr", "doctr-vitstr"),
        ("owlvit", "trocr"),
        ("yolo-v9-608", "cct"),
    ]

    print(f"\n[eval] Evaluating {len(pipeline_pairings)} pipelines...")
    detector_cache = {}
    ocr_cache = {}

    for det_name, ocr_name in pipeline_pairings:
        if det_name not in detectors or ocr_name not in ocr_models:
            print(f"\n  {det_name}+{ocr_name}: skipped (missing model)")
            continue

        print(f"\n  {det_name} + {ocr_name}")
        try:
            # Load detector (cache to avoid reloading)
            if det_name not in detector_cache:
                det_path = resolve_checkpoint_path(det_name, finetuned_dir)
                det_backend = build_backend(det_name, det_path, device=args.device)
                det_backend.ensure_loaded()
                det_backend.freeze()
                detector_cache[det_name] = det_backend
            else:
                det_backend = detector_cache[det_name]

            # Load OCR (cache to avoid reloading)
            if ocr_name not in ocr_cache:
                ocr_path = resolve_checkpoint_path(ocr_name, finetuned_dir)
                ocr_backend = build_ocr_backend(ocr_name, ocr_path, device=args.device)
                ocr_backend.ensure_loaded()
                ocr_backend.freeze()
                ocr_cache[ocr_name] = ocr_backend
            else:
                ocr_backend = ocr_cache[ocr_name]

            # Evaluate pipeline
            metrics = compute_pipeline_metrics(det_backend, ocr_backend, records, device=args.device)
            pipeline_name = f"{det_name}+{ocr_name}"
            results.append(EvalResults(pipeline_name, "pipeline", metrics))

            print(f"    Success={metrics['pipeline_success']:.4f}, Det Fail={metrics['det_failure_rate']:.4f}, OCR Fail|Det={metrics['ocr_failure_given_det']:.4f}")
        except Exception as e:
            print(f"    [error] {e}")
            traceback.print_exc()

    # Save results
    print(f"\n[eval] Saving results to {out_dir}")
    if results:
        save_results_csv(results, str(out_dir / "results.csv"))
        save_results_json(results, str(out_dir / "results.json"))
        save_summary_table(results, str(out_dir / "summary_table.txt"))
        plot_detector_pr_curves(results, str(out_dir / "detector_pr_curves.png"))
        plot_detector_metrics_bar(results, str(out_dir / "detector_metrics_bar.png"))
        plot_ocr_metrics_bar(results, str(out_dir / "ocr_metrics_bar.png"))
        plot_pipeline_metrics_bar(results, str(out_dir / "pipeline_metrics_bar.png"))
        print(f"\n[eval] Done!")
        return 0
    else:
        print("[error] No results to save (all evaluations failed)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
