"""
evaluate_finetuned.py

Comprehensive evaluation of finetuned detector and OCR models.
Produces paper-ready metrics: AP/mAP for detectors, CRR/LPRR for OCR.

Usage:
    python evaluate_finetuned.py --csv test_set_labels.csv --weights-dir weights/ --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import time
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

def load_test_records(csv_path: str) -> List[dict]:
    """
    Load test records from test_set_labels.csv.
    Returns list of dicts with keys:
        - filename: path to processed image
        - text: ground truth plate text
        - box: [x1, y1, x2, y2] in pixel coords
    """
    records = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("processed_filename") or not row.get("alpr_text"):
                continue
            try:
                x1 = float(row["alpr_x1"])
                y1 = float(row["alpr_y1"])
                x2 = float(row["alpr_x2"])
                y2 = float(row["alpr_y2"])
                records.append({
                    "filename": row["processed_filename"],
                    "text": row["alpr_text"].strip().upper(),
                    "box": torch.tensor([x1, y1, x2, y2], dtype=torch.float32),
                })
            except (ValueError, KeyError) as e:
                print(f"[warn] skipping row with bad bbox: {e}")
                continue
    return records


def resolve_weights_path(model_name: str, weights_dir: str) -> str:
    """
    Resolve model path: check finetuned first, fall back to base.
    Maps model name to (finetuned_path, base_path) pairs.
    """
    weights_dir = Path(weights_dir)

    detector_paths = {
        "fasterrcnn": (
            weights_dir / "finetuned" / "fasterrcnn_finetuned.pt",
            weights_dir / "model.pt"
        ),
        "rtdetr": (
            weights_dir / "finetuned" / "rtdetr_finetuned",
            weights_dir / "rtdetr-v2-license-plate"
        ),
        "owlvit": (
            weights_dir / "finetuned" / "owlvit_finetuned",
            None
        ),
        "yolo-v9-608": (
            weights_dir / "finetuned" / "yolo608_finetuned.pt",
            None
        ),
    }

    ocr_paths = {
        "lprnet": (
            weights_dir / "finetuned" / "lprnet_finetuned.pt",
            weights_dir / "lprnet_deployable_onnx_v1.1" / "us_lprnet_baseline18_deployable.onnx"
        ),
        "trocr": (
            weights_dir / "trocr_small_finetuned.pt",
            "microsoft/trocr-small-printed"
        ),
        "doctr-vitstr": (
            weights_dir / "vitstr_small_finetuned.pt",
            weights_dir / "vitstr_small_patch16_224.pth"
        ),
        "cct": (
            weights_dir / "finetuned" / "cct_s_finetuned.pt",
            None
        ),
    }

    if model_name in detector_paths:
        finetuned, base = detector_paths[model_name]
    elif model_name in ocr_paths:
        finetuned, base = ocr_paths[model_name]
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Try finetuned first
    if finetuned and Path(finetuned).exists():
        return str(finetuned)
    if base and Path(base).exists():
        return str(base)

    # Fallback: return finetuned path (will fail at load if doesn't exist)
    return str(finetuned) if finetuned else str(base)


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
                img = Image.open(record["filename"]).convert("RGB")
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
                img = Image.open(record["filename"]).convert("RGB")
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
    """Print and save summary table in two sections: detectors and OCR."""
    detectors = [r for r in results if r.model_type == "detector"]
    ocr_models = [r for r in results if r.model_type == "ocr"]

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


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate finetuned detector and OCR models"
    )
    parser.add_argument("--csv", default="test_set_labels.csv",
                        help="Path to test_set_labels.csv")
    parser.add_argument("--weights-dir", default="weights/",
                        help="Root directory for model weights")
    parser.add_argument("--detectors", nargs="*", default=None,
                        choices=["fasterrcnn", "rtdetr", "owlvit", "yolo-v9-608"],
                        help="Detectors to evaluate (default: all)")
    parser.add_argument("--ocr", nargs="*", default=None,
                        choices=["lprnet", "trocr", "doctr-vitstr", "cct"],
                        help="OCR models to evaluate (default: all)")
    parser.add_argument("--device", default="cuda",
                        help="Device: cuda or cpu")
    parser.add_argument("--output", default="results/eval_finetuned/",
                        help="Output directory")
    args = parser.parse_args()

    # Default to all models if not specified
    detectors = args.detectors if args.detectors is not None else ["fasterrcnn", "rtdetr", "owlvit", "yolo-v9-608"]
    ocr_models = args.ocr if args.ocr is not None else ["lprnet", "trocr", "doctr-vitstr", "cct"]

    # Ensure device exists
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA not available, falling back to CPU")
        args.device = "cpu"

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load test records
    print(f"[eval] Loading test data from {args.csv}")
    records = load_test_records(args.csv)
    print(f"[eval] Loaded {len(records)} records")

    results = []

    # Evaluate detectors
    if detectors:
        print(f"\n[eval] Evaluating {len(detectors)} detectors...")
        for detector_name in detectors:
            print(f"\n── {detector_name} ──")
            try:
                weight_path = resolve_weights_path(detector_name, args.weights_dir)
                backend = build_backend(detector_name, weight_path, device=args.device)
                backend.ensure_loaded()
                backend.freeze()

                metrics = compute_detector_metrics(backend, records, device=args.device)
                results.append(EvalResults(detector_name, "detector", metrics))

                print(f"  AP@0.5={metrics['ap_50']:.4f}, mAP={metrics['map_50_95']:.4f}")
            except Exception as e:
                print(f"[error] Failed to evaluate {detector_name}: {e}")

    # Evaluate OCR
    if ocr_models:
        print(f"\n[eval] Evaluating {len(ocr_models)} OCR models...")
        for ocr_name in ocr_models:
            print(f"\n── {ocr_name} ──")
            try:
                weight_path = resolve_weights_path(ocr_name, args.weights_dir)
                backend = build_ocr_backend(ocr_name, weight_path, device=args.device)
                backend.ensure_loaded()
                backend.freeze()

                metrics = compute_ocr_metrics(backend, records, device=args.device)
                results.append(EvalResults(ocr_name, "ocr", metrics))

                print(f"  CRR={metrics['crr']:.4f}, LPRR={metrics['lprr']:.4f}")
            except Exception as e:
                print(f"[error] Failed to evaluate {ocr_name}: {e}")

    # Save results
    print(f"\n[eval] Saving results to {out_dir}")
    save_results_csv(results, str(out_dir / "results.csv"))
    save_results_json(results, str(out_dir / "results.json"))
    save_summary_table(results, str(out_dir / "summary_table.txt"))
    plot_detector_pr_curves(results, str(out_dir / "detector_pr_curves.png"))
    plot_detector_metrics_bar(results, str(out_dir / "detector_metrics_bar.png"))
    plot_ocr_metrics_bar(results, str(out_dir / "ocr_metrics_bar.png"))

    print(f"\n[eval] Done!")


if __name__ == "__main__":
    main()
