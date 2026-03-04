"""
evaluator.py

Evaluation harness for comparing license-plate detector backends.

Typical usage
-------------
    python evaluator.py \\
        --csv updated_control_corners.csv \\
        --backends yolov8:weights/lp_v8.pt yolov5:weights/lp_v5.pt \\
        --device cuda \\
        --patch patches/best.pt \\
        --output eval_report.png
"""

from __future__ import annotations

import argparse
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
# Per-image result container
# ---------------------------------------------------------------------------

@dataclass
class ImageResult:
    backend_name: str
    image_id: int
    gt_box: torch.Tensor              # [4]  xyxy in prep-image space
    detections: List[Detection]
    inference_ms: float
    best_iou: float = 0.0
    best_conf: float = 0.0
    detected: bool = False            # True if best_iou > iou_threshold


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

@dataclass
class BackendMetrics:
    name: str
    num_images: int = 0
    # Detection quality
    true_positives: int = 0
    false_negatives: int = 0
    total_detections: int = 0
    iou_values: List[float] = field(default_factory=list)
    conf_values: List[float] = field(default_factory=list)
    # Speed
    latency_ms: List[float] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

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
    def avg_detections_per_image(self) -> float:
        return self.total_detections / self.num_images if self.num_images else 0.0

    def summary(self) -> str:
        return (
            f"[{self.name}]  "
            f"Recall={self.recall:.3f}  "
            f"mIoU={self.mean_iou:.3f}  "
            f"mConf={self.mean_conf:.3f}  "
            f"Latency={self.mean_latency_ms:.1f}ms (p95={self.p95_latency_ms:.1f}ms)"
        )


# ---------------------------------------------------------------------------
# Core evaluator
# ---------------------------------------------------------------------------

class DetectorEvaluator:
    """
    Runs one or more DetectorBackends against a labelled dataset and
    produces side-by-side metric tables and visualisations.
    """

    def __init__(self,
                 backends: List[DetectorBackend],
                 csv_path: str,
                 device: str = "cpu",
                 iou_threshold: float = 0.5,
                 patch_path: Optional[str] = None,
                 patch_size: Tuple[int, int] = (256, 512)):
        import torchvision.transforms as T
        from dataset import create_dataloaders

        self.backends = backends
        self.device = device
        self.iou_threshold = iou_threshold
        self.patch_path = patch_path
        self.patch_size = patch_size           # (H, W)

        transform = T.Compose([T.ToTensor()])
        _, self.val_loader = create_dataloaders(
            csv_path, transform=transform,
            preload=True, batch_size=1, n_jobs=0,
        )

        # Optionally load a pre-trained adversarial patch for robustness eval
        self._patch: Optional[torch.Tensor] = None
        if patch_path:
            self._patch = self._load_patch(patch_path)

    # ------------------------------------------------------------------
    # Patch helpers
    # ------------------------------------------------------------------

    def _load_patch(self, path: str) -> torch.Tensor:
        """Load a saved adversarial patch (.pt or .png)."""
        import torchvision.transforms as T
        from PIL import Image

        p = Path(path)
        if p.suffix == ".pt":
            ckpt = torch.load(path, map_location="cpu")
            raw = ckpt["patch"] if "patch" in ckpt else ckpt
        else:
            img = Image.open(path).convert("RGB")
            img = img.resize((self.patch_size[1], self.patch_size[0]))
            raw = T.ToTensor()(img)
            # Convert to arctanh space to match trainer convention
            raw = torch.arctanh(torch.clamp(raw * 2 - 1, -0.99, 0.99))

        print(f"[evaluator] Loaded adversarial patch from {path}")
        return raw.to(self.device)

    def _apply_patch(self, image: torch.Tensor,
                     corners: torch.Tensor) -> torch.Tensor:
        """
        Apply the adversarial patch around the licence plate corners.
        Thin wrapper that reuses the logic from the trainer; kept here so the
        evaluator has no mandatory dependency on AdversarialPatchTrainer.
        """
        import torch.nn.functional as F
        import kornia.geometry as K

        if self._patch is None:
            return image

        batch = image.unsqueeze(0).to(self.device)
        patch_h, patch_w = self.patch_size
        patch_norm = torch.tanh(self._patch) * 0.5 + 0.5
        img_h, img_w = batch.shape[2], batch.shape[3]

        plate_corners = corners[0]  # [4, 2]
        cx = plate_corners[:, 0].mean()
        cy = plate_corners[:, 1].mean()
        center = torch.tensor([cx, cy], device=self.device)
        border = center.unsqueeze(0) + (plate_corners - center.unsqueeze(0)) * 1.4
        border = border.unsqueeze(0)

        src = torch.tensor([[0, 0], [patch_w, 0], [patch_w, patch_h], [0, patch_h]],
                           dtype=torch.float32, device=self.device).unsqueeze(0)

        M_border = K.get_perspective_transform(src, border)
        M_plate  = K.get_perspective_transform(src, corners)
        mask_ones = torch.ones(1, 1, patch_h, patch_w, device=self.device)

        warped = K.warp_perspective(
            patch_norm.unsqueeze(0), M_border, dsize=(img_h, img_w),
            mode="bilinear", padding_mode="zeros", align_corners=True,
        )
        w_border = K.warp_perspective(
            mask_ones, M_border, dsize=(img_h, img_w),
            mode="bilinear", padding_mode="zeros", align_corners=True,
        )
        w_plate = K.warp_perspective(
            mask_ones, M_plate, dsize=(img_h, img_w),
            mode="bilinear", padding_mode="zeros", align_corners=True,
        )
        mask = torch.clamp(w_border - w_plate, 0, 1).expand(-1, 3, -1, -1)

        result = batch * (1 - mask) + warped * mask
        return torch.clamp(result, 0, 1).squeeze(0)

    # ------------------------------------------------------------------
    # IoU
    # ------------------------------------------------------------------

    @staticmethod
    def _iou(box_a: torch.Tensor, box_b: torch.Tensor) -> float:
        """IoU between two [x1, y1, x2, y2] tensors."""
        ix1 = max(box_a[0].item(), box_b[0].item())
        iy1 = max(box_a[1].item(), box_b[1].item())
        ix2 = min(box_a[2].item(), box_b[2].item())
        iy2 = min(box_a[3].item(), box_b[3].item())
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        a1 = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        a2 = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = a1.item() + a2.item() - inter
        return inter / (union + 1e-8) if union > 0 else 0.0

    # ------------------------------------------------------------------
    # Single-backend evaluation
    # ------------------------------------------------------------------

    def _evaluate_backend(self, backend: DetectorBackend,
                          with_patch: bool = False) -> BackendMetrics:
        metrics = BackendMetrics(
            name=f"{backend.name}{'_patched' if with_patch else ''}"
        )
        backend.eval()

        with torch.no_grad():
            for batch in tqdm(self.val_loader,
                              desc=f"  {metrics.name}", leave=False):
                batch = {k: v[0] for k, v in batch.items()}

                prep_image = batch["prep_image"].to(self.device)
                corners    = batch["new_corners"].to(self.device)

                # Ground-truth box in prep-image space
                c = corners
                gt_box = torch.stack([
                    c[:, 0].min(), c[:, 1].min(),
                    c[:, 0].max(), c[:, 1].max(),
                ])

                if with_patch and self._patch is not None:
                    prep_image = self._apply_patch(prep_image, corners.unsqueeze(0))

                t0 = time.perf_counter()
                dets = backend.predict(prep_image)
                elapsed_ms = (time.perf_counter() - t0) * 1000

                metrics.num_images      += 1
                metrics.total_detections += len(dets)
                metrics.latency_ms.append(elapsed_ms)

                best_iou  = 0.0
                best_conf = 0.0
                for det in dets:
                    iou = self._iou(det.box, gt_box)
                    if iou > best_iou:
                        best_iou  = iou
                        best_conf = det.confidence

                metrics.iou_values.append(best_iou)
                metrics.conf_values.append(best_conf)

                if best_iou >= self.iou_threshold:
                    metrics.true_positives += 1
                else:
                    metrics.false_negatives += 1

        return metrics

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, BackendMetrics]:
        """
        Evaluate every registered backend with and (if a patch is provided)
        without the adversarial patch.

        Returns a mapping  ``{metrics.name: BackendMetrics}``.
        """
        all_metrics: Dict[str, BackendMetrics] = {}

        for backend in self.backends:
            backend.ensure_loaded()
            backend.freeze()

            print(f"\n── Evaluating {backend.name} (clean) ──")
            m_clean = self._evaluate_backend(backend, with_patch=False)
            all_metrics[m_clean.name] = m_clean
            print(f"  {m_clean.summary()}")

            if self._patch is not None:
                print(f"── Evaluating {backend.name} (patched) ──")
                m_patch = self._evaluate_backend(backend, with_patch=True)
                all_metrics[m_patch.name] = m_patch
                print(f"  {m_patch.summary()}")

        return all_metrics

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report(self, metrics: Dict[str, BackendMetrics],
               output_path: str = "eval_report.png") -> None:
        """Save a multi-panel comparison figure."""
        names   = list(metrics.keys())
        recalls = [metrics[n].recall          for n in names]
        mious   = [metrics[n].mean_iou        for n in names]
        mconfs  = [metrics[n].mean_conf       for n in names]
        lats    = [metrics[n].mean_latency_ms for n in names]
        p95s    = [metrics[n].p95_latency_ms  for n in names]
        dets    = [metrics[n].avg_detections_per_image for n in names]

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("Detector Backend Evaluation", fontsize=16, weight="bold")

        colour_map = plt.cm.tab10
        colours = [colour_map(i / max(len(names), 1)) for i in range(len(names))]

        def _bar(ax, values, title, ylabel, ylim=None):
            bars = ax.bar(names, values, color=colours)
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            if ylim:
                ax.set_ylim(*ylim)
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=25, ha="right")
            ax.grid(axis="y", alpha=0.3)
            for bar, v in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8)

        _bar(axes[0, 0], recalls, "Recall",              "Recall",       (0, 1))
        _bar(axes[0, 1], mious,   "Mean IoU (best det)", "mIoU",         (0, 1))
        _bar(axes[0, 2], mconfs,  "Mean Confidence",     "Confidence",   (0, 1))
        _bar(axes[1, 0], lats,    "Mean Latency",        "ms / image")
        _bar(axes[1, 1], p95s,    "P95 Latency",         "ms / image")
        _bar(axes[1, 2], dets,    "Avg Detections / Image", "count")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n[evaluator] Report saved → {output_path}")

    def print_table(self, metrics: Dict[str, BackendMetrics]) -> None:
        """Pretty-print a comparison table to stdout."""
        col_w = 14
        header = (f"{'Backend':<20}  {'Recall':>{col_w}}  {'mIoU':>{col_w}}  "
                  f"{'mConf':>{col_w}}  {'Lat(ms)':>{col_w}}  {'p95(ms)':>{col_w}}")
        print("\n" + "=" * len(header))
        print("EVALUATION RESULTS")
        print("=" * len(header))
        print(header)
        print("-" * len(header))
        for name, m in metrics.items():
            print(
                f"{name:<20}  {m.recall:>{col_w}.4f}  {m.mean_iou:>{col_w}.4f}  "
                f"{m.mean_conf:>{col_w}.4f}  {m.mean_latency_ms:>{col_w}.2f}  "
                f"{m.p95_latency_ms:>{col_w}.2f}"
            )
        print("=" * len(header))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_backend_spec(spec: str) -> Tuple[str, str, dict]:
    """
    Parse a CLI backend specification string.

    Format:  ``name:path``  or  ``name:path:key=val,key=val``

    Examples
    --------
    ``yolov8:weights/lp.pt``
    ``yolov8:weights/lp.pt:conf_threshold=0.3``
    ``mock:none``
    """
    parts = spec.split(":", 2)
    name = parts[0]
    path = parts[1] if len(parts) > 1 else "none"
    kwargs: dict = {}
    if len(parts) == 3:
        for kv in parts[2].split(","):
            k, _, v = kv.partition("=")
            # Try to coerce to float/int
            try:
                kwargs[k] = float(v) if "." in v else int(v)
            except ValueError:
                kwargs[k] = v
    return name, path, kwargs


def main():
    parser = argparse.ArgumentParser(description="Evaluate detector backends")
    parser.add_argument("--csv",      required=True,
                        help="Path to dataset CSV (updated_control_corners.csv)")
    parser.add_argument("--backends", nargs="+", required=True,
                        metavar="name:path[:opts]",
                        help="Backend specs, e.g. yolov8:weights/lp.pt  mock:none")
    parser.add_argument("--device",   default="cpu",
                        help="Torch device (cpu / cuda / mps)")
    parser.add_argument("--iou-threshold", type=float, default=0.5,
                        help="IoU threshold to count a detection as a true positive")
    parser.add_argument("--patch",    default=None,
                        help="Optional adversarial patch (.pt or .png) for robustness eval")
    parser.add_argument("--output",   default="eval_report.png",
                        help="Path for the output comparison figure")
    args = parser.parse_args()

    # Build backend objects
    backends: List[DetectorBackend] = []
    for spec in args.backends:
        bname, bpath, bkwargs = _parse_backend_spec(spec)
        backend = build_backend(bname, bpath, device=args.device, **bkwargs)
        backends.append(backend)

    evaluator = DetectorEvaluator(
        backends=backends,
        csv_path=args.csv,
        device=args.device,
        iou_threshold=args.iou_threshold,
        patch_path=args.patch,
    )

    metrics = evaluator.run()
    evaluator.print_table(metrics)
    evaluator.report(metrics, output_path=args.output)


if __name__ == "__main__":
    main()