"""
evaluator.py

Evaluation harness for comparing adversarial patches across detector backends.

Two modes
---------
Single patch:
    python evaluator.py --csv data.csv --backends yolov8:lp.pt rtdetr:detr.pt
                        --patches patches/patch_yolov8.pt --output results/

Cross-model matrix (one patch trained per model, evaluate all vs all):
    python evaluator.py --csv data.csv --backends yolov8:lp.pt rtdetr:detr.pt
                        --patches patch_yolov8.pt patch_rtdetr.pt --output results/

Outputs (all written to --output directory)
-------------------------------------------
    metrics.csv              — raw per-backend numbers, one row per backend x patch combo
    metrics.json             — same, with full per-image value lists
    summary_table.txt        — formatted ASCII table printed to stdout and saved
    bar_chart.png            — recall / mIoU / confidence / latency bar charts (clean baseline)
    matrix_recall.png        — heatmap: patch trained on (rows) x model evaluated on (cols)
    matrix_iou.png           — same for mIoU
    matrix_attack_drop.png   — recall drop caused by each patch on each model
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
from tqdm import tqdm

from detector_backends import DetectorBackend, Detection, build_backend
from dataset import create_dataloaders


# ---------------------------------------------------------------------------
# Metrics container
# ---------------------------------------------------------------------------

@dataclass
class BackendMetrics:
    name: str                                        # backend identifier
    patch_name: str = "clean"                        # patch file stem, or "clean"
    num_images: int = 0
    true_positives: int = 0
    false_negatives: int = 0
    total_detections: int = 0
    iou_values: List[float] = field(default_factory=list)
    conf_values: List[float] = field(default_factory=list)
    latency_ms: List[float] = field(default_factory=list)
    # OCR categorisation (only populated when an OCR backend is provided)
    ocr_correct: int = 0
    ocr_impersonation: int = 0
    ocr_misread: int = 0
    ocr_no_detection: int = 0

    @property
    def recall(self) -> float:
        total = self.true_positives + self.false_negatives
        return self.true_positives / total if total else 0.0

    @property
    def mean_iou(self) -> float:
        return float(np.mean(self.iou_values)) if self.iou_values else 0.0

    @property
    def mean_conf(self) -> float:
        return float(np.mean(self.conf_values)) if self.conf_values else 0.0

    @property
    def mean_latency_ms(self) -> float:
        return float(np.mean(self.latency_ms)) if self.latency_ms else 0.0

    @property
    def p95_latency_ms(self) -> float:
        return float(np.percentile(self.latency_ms, 95)) if self.latency_ms else 0.0

    @property
    def avg_dets_per_image(self) -> float:
        return self.total_detections / self.num_images if self.num_images else 0.0

    @property
    def ocr_total(self) -> int:
        return self.ocr_correct + self.ocr_impersonation + self.ocr_misread + self.ocr_no_detection

    @property
    def ocr_impersonation_rate(self) -> float:
        return self.ocr_impersonation / self.ocr_total if self.ocr_total else 0.0

    @property
    def ocr_correct_rate(self) -> float:
        return self.ocr_correct / self.ocr_total if self.ocr_total else 0.0

    def to_flat_dict(self) -> dict:
        d = {
            "backend":            self.name,
            "patch":              self.patch_name,
            "num_images":         self.num_images,
            "true_positives":     self.true_positives,
            "false_negatives":    self.false_negatives,
            "total_detections":   self.total_detections,
            "recall":             round(self.recall, 4),
            "mean_iou":           round(self.mean_iou, 4),
            "mean_conf":          round(self.mean_conf, 4),
            "mean_latency_ms":    round(self.mean_latency_ms, 2),
            "p95_latency_ms":     round(self.p95_latency_ms, 2),
            "avg_dets_per_image": round(self.avg_dets_per_image, 2),
        }
        if self.ocr_total > 0:
            d.update({
                "ocr_correct":            self.ocr_correct,
                "ocr_impersonation":      self.ocr_impersonation,
                "ocr_misread":            self.ocr_misread,
                "ocr_no_detection":       self.ocr_no_detection,
                "ocr_correct_rate":       round(self.ocr_correct_rate, 4),
                "ocr_impersonation_rate": round(self.ocr_impersonation_rate, 4),
            })
        return d

    def summary(self) -> str:
        s = (
            f"[{self.name} | patch={self.patch_name}]  "
            f"Recall={self.recall:.3f}  mIoU={self.mean_iou:.3f}  "
            f"mConf={self.mean_conf:.3f}  "
            f"Lat={self.mean_latency_ms:.1f}ms (p95={self.p95_latency_ms:.1f}ms)"
        )
        if self.ocr_total > 0:
            s += (
                f"  |  OCR: correct={self.ocr_correct_rate:.1%}"
                f"  imp={self.ocr_impersonation_rate:.1%}"
                f"  misread={self.ocr_misread/self.ocr_total:.1%}"
                f"  no_det={self.ocr_no_detection/self.ocr_total:.1%}"
            )
        return s


# ---------------------------------------------------------------------------
# Core evaluator
# ---------------------------------------------------------------------------

class DetectorEvaluator:

    def __init__(self,
                 backends: List[DetectorBackend],
                 csv_path: str,
                 device: str = "cpu",
                 iou_threshold: float = 0.5,
                 patch_size: Tuple[int, int] = (256, 512)):
        import torchvision.transforms as T

        self.backends      = backends
        self.device        = device
        self.iou_threshold = iou_threshold
        self.patch_size    = patch_size

        transform = T.Compose([T.ToTensor()])
        _, self.val_loader = create_dataloaders(
            csv_path, transform=transform,
            preload=False, batch_size=1, n_jobs=0,
        )

    # ── patch helpers ─────────────────────────────────────────────────────

    def _load_patch(self, path: str) -> torch.Tensor:
        import torchvision.transforms as T
        from PIL import Image

        p = Path(path)
        if p.suffix == ".pt":
            ckpt = torch.load(path, map_location="cpu")
            raw  = ckpt.get("patch", ckpt)
        else:
            img = Image.open(path).convert("RGB")
            img = img.resize((self.patch_size[1], self.patch_size[0]))
            raw = T.ToTensor()(img)
            raw = torch.arctanh(torch.clamp(raw * 2 - 1, -0.99, 0.99))

        return raw.to(self.device)

    def _apply_patch(self, image: torch.Tensor, corners: torch.Tensor,
                     patch: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F
        import kornia.geometry as K

        batch      = image.unsqueeze(0).to(self.device)
        patch_h, patch_w = self.patch_size
        patch_norm = torch.tanh(patch) * 0.5 + 0.5
        img_h, img_w = batch.shape[2], batch.shape[3]

        plate = corners[0]
        cx    = plate[:, 0].mean()
        cy    = plate[:, 1].mean()
        ctr   = torch.tensor([cx, cy], device=self.device)
        border = (ctr.unsqueeze(0) + (plate - ctr.unsqueeze(0)) * 1.4).unsqueeze(0)

        src = torch.tensor(
            [[0, 0], [patch_w, 0], [patch_w, patch_h], [0, patch_h]],
            dtype=torch.float32, device=self.device,
        ).unsqueeze(0)

        M_b  = K.get_perspective_transform(src, border)
        M_p  = K.get_perspective_transform(src, corners)
        ones = torch.ones(1, 1, patch_h, patch_w, device=self.device)

        warped   = K.warp_perspective(patch_norm.unsqueeze(0), M_b,
                                      (img_h, img_w), mode="bilinear",
                                      padding_mode="zeros", align_corners=True)
        w_border = K.warp_perspective(ones, M_b, (img_h, img_w), mode="bilinear",
                                      padding_mode="zeros", align_corners=True)
        w_plate  = K.warp_perspective(ones, M_p, (img_h, img_w), mode="bilinear",
                                      padding_mode="zeros", align_corners=True)
        mask = torch.clamp(w_border - w_plate, 0, 1).expand(-1, 3, -1, -1)

        return torch.clamp(batch * (1 - mask) + warped * mask, 0, 1).squeeze(0)

    # ── IoU ──────────────────────────────────────────────────────────────

    @staticmethod
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

    # ── single backend x single patch ────────────────────────────────────

    def _evaluate_one(self, backend: DetectorBackend,
                      patch: Optional[torch.Tensor],
                      patch_name: str) -> BackendMetrics:
        m = BackendMetrics(name=backend.name, patch_name=patch_name)
        backend.eval()

        with torch.no_grad():
            for batch in tqdm(self.val_loader,
                              desc=f"  {backend.name} | {patch_name}", leave=False):
                batch = {k: v[0] for k, v in batch.items()}

                prep_image = batch["prep_image"].to(self.device)
                corners    = batch["new_corners"].to(self.device)

                gt_box = torch.stack([
                    corners[:, 0].min(), corners[:, 1].min(),
                    corners[:, 0].max(), corners[:, 1].max(),
                ])

                if patch is not None:
                    prep_image = self._apply_patch(
                        prep_image, corners.unsqueeze(0), patch)

                t0   = time.perf_counter()
                dets = backend.predict(prep_image)
                m.latency_ms.append((time.perf_counter() - t0) * 1000)

                m.num_images       += 1
                m.total_detections += len(dets)

                best_iou = best_conf = 0.0
                for det in dets:
                    iou = self._iou(det.box, gt_box)
                    if iou > best_iou:
                        best_iou  = iou
                        best_conf = det.confidence

                m.iou_values.append(best_iou)
                m.conf_values.append(best_conf)
                if best_iou >= self.iou_threshold:
                    m.true_positives += 1
                else:
                    m.false_negatives += 1

        return m

    # ── public run ────────────────────────────────────────────────────────

    def run(self, patch_paths: Optional[List[str]] = None) -> List[BackendMetrics]:
        """
        Evaluate all backends against clean baseline + all supplied patches.

        Returns a flat list of BackendMetrics, one per (backend, patch) pair.
        """
        patches: List[Tuple[str, Optional[torch.Tensor]]] = [("clean", None)]
        if patch_paths:
            for p in patch_paths:
                name   = Path(p).stem
                tensor = self._load_patch(p)
                patches.append((name, tensor))
                print(f"[evaluator] Loaded patch '{name}' from {p}")

        results: List[BackendMetrics] = []
        for backend in self.backends:
            backend.ensure_loaded()
            backend.freeze()
            print(f"\n── Backend: {backend.name} ──")
            for patch_name, patch_tensor in patches:
                m = self._evaluate_one(backend, patch_tensor, patch_name)
                results.append(m)
                print(f"  {m.summary()}")

        return results

    # ── saving ────────────────────────────────────────────────────────────

    def save_csv(self, results: List[BackendMetrics], path: str) -> None:
        rows       = [m.to_flat_dict() for m in results]
        fieldnames = list(rows[0].keys())
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"[evaluator] Stats saved  → {path}")

    def save_json(self, results: List[BackendMetrics], path: str) -> None:
        data = []
        for m in results:
            d = m.to_flat_dict()
            d["iou_values"]  = m.iou_values
            d["conf_values"] = m.conf_values
            d["latency_ms"]  = m.latency_ms
            data.append(d)
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2)
        print(f"[evaluator] Full stats   → {path}")

    def save_summary_table(self, results: List[BackendMetrics], path: str) -> None:
        col = 12
        hdr = (
            f"{'backend':<22} {'patch':<32} "
            f"{'recall':>{col}} {'mIoU':>{col}} {'mConf':>{col}} "
            f"{'lat(ms)':>{col}} {'p95(ms)':>{col}}"
        )
        sep   = "─" * len(hdr)
        lines = [sep, "EVALUATION RESULTS", sep, hdr, sep]
        prev  = None
        for m in results:
            if prev and m.name != prev:
                lines.append("")
            lines.append(
                f"{m.name:<22} {m.patch_name:<32} "
                f"{m.recall:>{col}.4f} {m.mean_iou:>{col}.4f} "
                f"{m.mean_conf:>{col}.4f} "
                f"{m.mean_latency_ms:>{col}.2f} {m.p95_latency_ms:>{col}.2f}"
            )
            prev = m.name
        lines.append(sep)
        text = "\n".join(lines)
        print("\n" + text)
        with open(path, "w") as fh:
            fh.write(text + "\n")
        print(f"[evaluator] Summary      → {path}")

    # ── visualisations ────────────────────────────────────────────────────

    def save_bar_chart(self, results: List[BackendMetrics], path: str) -> None:
        """Bar charts of clean-baseline metrics across backends."""
        clean = [m for m in results if m.patch_name == "clean"]
        if not clean:
            return
        names   = [m.name for m in clean]
        colours = plt.cm.tab10(np.linspace(0, 1, len(names)))

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("Detector Baseline Performance (no patch)",
                     fontsize=15, weight="bold")

        def _bar(ax, vals, title, ylabel, ylim=None):
            bars = ax.bar(names, vals, color=colours)
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            if ylim:
                ax.set_ylim(*ylim)
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=25, ha="right")
            ax.grid(axis="y", alpha=0.3)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8)

        _bar(axes[0, 0], [m.recall             for m in clean], "Recall",               "Recall",   (0, 1))
        _bar(axes[0, 1], [m.mean_iou           for m in clean], "Mean IoU",             "mIoU",     (0, 1))
        _bar(axes[0, 2], [m.mean_conf          for m in clean], "Mean Confidence",      "Conf",     (0, 1))
        _bar(axes[1, 0], [m.mean_latency_ms    for m in clean], "Mean Latency",         "ms/image")
        _bar(axes[1, 1], [m.p95_latency_ms     for m in clean], "P95 Latency",          "ms/image")
        _bar(axes[1, 2], [m.avg_dets_per_image for m in clean], "Avg Detections/Image", "count")

        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[evaluator] Bar chart    → {path}")

    def save_matrix_heatmaps(self, results: List[BackendMetrics],
                              out_dir: str) -> None:
        """
        Three heatmaps: recall, mIoU, and recall-drop.
        Rows = patch trained on, cols = model evaluated on.
        Skipped silently if no patches were provided.
        """
        patch_names   = sorted({m.patch_name for m in results} - {"clean"})
        backend_names = sorted({m.name for m in results})

        if not patch_names:
            print("[evaluator] No patches — skipping heatmaps")
            return

        out = Path(out_dir)
        idx: Dict[Tuple[str, str], BackendMetrics] = {
            (m.name, m.patch_name): m for m in results
        }
        clean_idx: Dict[str, BackendMetrics] = {
            m.name: m for m in results if m.patch_name == "clean"
        }

        def _heatmap(data, title, fmt, cmap, vmin, vmax, fname):
            nrows, ncols = data.shape
            fig, ax = plt.subplots(
                figsize=(max(6, ncols * 1.5), max(3, nrows * 1.0 + 1))
            )
            im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
            ax.set_xticks(range(ncols))
            ax.set_xticklabels(backend_names, rotation=30, ha="right", fontsize=9)
            ax.set_yticks(range(nrows))
            ax.set_yticklabels(patch_names, fontsize=9)
            ax.set_xlabel("Model evaluated on", fontsize=10)
            ax.set_ylabel("Patch trained on", fontsize=10)
            ax.set_title(title, fontsize=12, weight="bold", pad=12)
            for r in range(nrows):
                for c in range(ncols):
                    v = data[r, c]
                    if not np.isnan(v):
                        brightness = (v - vmin) / (vmax - vmin + 1e-8)
                        txt_col = "white" if brightness > 0.6 else "black"
                        ax.text(c, r, format(v, fmt),
                                ha="center", va="center",
                                fontsize=9, color=txt_col, weight="bold")
            plt.tight_layout()
            fpath = out / fname
            plt.savefig(fpath, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"[evaluator] Heatmap      → {fpath}")

        # recall matrix
        recall = np.full((len(patch_names), len(backend_names)), np.nan)
        for r, pn in enumerate(patch_names):
            for c, bn in enumerate(backend_names):
                m = idx.get((bn, pn))
                if m:
                    recall[r, c] = m.recall
        _heatmap(recall, "Recall under patch attack",
                 ".3f", "RdYlGn", 0.0, 1.0, "matrix_recall.png")

        # mIoU matrix
        iou = np.full((len(patch_names), len(backend_names)), np.nan)
        for r, pn in enumerate(patch_names):
            for c, bn in enumerate(backend_names):
                m = idx.get((bn, pn))
                if m:
                    iou[r, c] = m.mean_iou
        _heatmap(iou, "Mean IoU under patch attack",
                 ".3f", "RdYlGn", 0.0, 1.0, "matrix_iou.png")

        # recall drop (clean - patched): higher = patch more effective
        drop = np.full((len(patch_names), len(backend_names)), np.nan)
        for r, pn in enumerate(patch_names):
            for c, bn in enumerate(backend_names):
                mp = idx.get((bn, pn))
                mc = clean_idx.get(bn)
                if mp and mc:
                    drop[r, c] = mc.recall - mp.recall
        _heatmap(drop, "Recall drop caused by patch  (higher = more effective attack)",
                 "+.3f", "Reds", 0.0, 1.0, "matrix_attack_drop.png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_backend_spec(spec: str) -> Tuple[str, str, dict]:
    parts  = spec.split(":", 2)
    name   = parts[0]
    path   = parts[1] if len(parts) > 1 else "none"
    kwargs: dict = {}
    if len(parts) == 3:
        for kv in parts[2].split(","):
            k, _, v = kv.partition("=")
            try:
                kwargs[k] = float(v) if "." in v else int(v)
            except ValueError:
                kwargs[k] = v
    return name, path, kwargs


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate adversarial patches across detector backends",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--csv", required=True)
    parser.add_argument("--backends", nargs="+", required=True,
                        metavar="name:path[:opts]",
                        help=(
                            "Backend specs e.g.:\n"
                            "  yolov8:weights/lp.pt\n"
                            "  yolov11:yolov11s-license-plate.pt\n"
                            "  rtdetr:weights/detr.pt\n"
                            "  fastanpr:none"
                        ))
    parser.add_argument("--patches", nargs="*", default=None,
                        metavar="patch.pt",
                        help=(
                            "Patch files to evaluate (space-separated).\n"
                            "For cross-model matrix supply one patch per backend.\n"
                            "Omit to evaluate clean baseline only."
                        ))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--output", default="results/",
                        help="Output directory (default: results/)")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    backends: List[DetectorBackend] = []
    for spec in args.backends:
        bname, bpath, bkwargs = _parse_backend_spec(spec)
        backends.append(build_backend(bname, bpath, device=args.device, **bkwargs))

    evaluator = DetectorEvaluator(
        backends=backends,
        csv_path=args.csv,
        device=args.device,
        iou_threshold=args.iou_threshold,
    )

    results = evaluator.run(patch_paths=args.patches)

    evaluator.save_csv(results,             str(out_dir / "metrics.csv"))
    evaluator.save_json(results,            str(out_dir / "metrics.json"))
    evaluator.save_summary_table(results,   str(out_dir / "summary_table.txt"))
    evaluator.save_bar_chart(results,       str(out_dir / "bar_chart.png"))
    evaluator.save_matrix_heatmaps(results, str(out_dir))


if __name__ == "__main__":
    main()
