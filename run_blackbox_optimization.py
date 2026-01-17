#!/usr/bin/env python3
"""
Black-box adversarial patch optimization using fast-alpr as a true black box.

Ground truth plate is fixed (VRJ7774).
Detection selection is based on IoU with ground truth corners (when provided),
matching the approach used in model_extraction and optimize_patch.
"""

import argparse
from typing import List, Optional
import torch
import numpy as np
import cv2

from fast_alpr import ALPR

from model_extraction import (
    BlackBoxModel,
    ALPRResult,
    optimize_patch_bb,
)
from optimize_patch import corners_to_bbox, compute_iou


GROUND_TRUTH_PLATE = "VRJ7774"


class LargeFastALPR(BlackBoxModel):
    """
    True black-box ALPR using fast-alpr high-level API.
    """

    def __init__(self, device: str = None):
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        print("Loading fast-alpr (true black box)...")

        self.alpr = ALPR(
            detector_model="yolo-v9-s-608-license-plate-end2end",
            ocr_model="cct-s-v1-global-model",
        )

        print(f"fast-alpr loaded on device: {self.device}")

    def evaluate(
        self,
        images: List[torch.Tensor],
        corners: Optional[List[torch.Tensor]] = None
    ) -> List[ALPRResult]:
        """
        Evaluate images using fast-alpr.

        Detection is chosen by:
        - IoU with ground truth corners (when corners provided)
        - Highest confidence (when corners not provided)
        """
        results: List[ALPRResult] = []

        for idx, image in enumerate(images):
            # torch [C,H,W] -> uint8 BGR numpy
            img_np = image.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)

            if img_np.ndim != 3 or img_np.shape[2] != 3:
                results.append(ALPRResult(text=None, confidence=0.0))
                continue

            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            try:
                predictions = self.alpr.predict(img_bgr)
            except Exception as e:
                print(f"ALPR inference failed: {e}")
                results.append(ALPRResult(text=None, confidence=0.0))
                continue

            if not predictions:
                results.append(ALPRResult(text=None, confidence=0.0))
                continue

            # Select best detection based on IoU or confidence
            best_pred = None

            if corners is not None and idx < len(corners):
                # IoU-based selection (matches model_extraction approach)
                gt_corners = corners[idx]
                target_box = corners_to_bbox(gt_corners)
                best_iou = -1.0

                for pred in predictions:
                    if pred.detection is None or pred.detection.bounding_box is None:
                        continue

                    # fast-alpr bbox: [x1, y1, x2, y2]
                    pred_box = torch.tensor([
                        pred.detection.bounding_box.x1,
                        pred.detection.bounding_box.y1,
                        pred.detection.bounding_box.x2,
                        pred.detection.bounding_box.y2
                    ], dtype=torch.float32)

                    iou = compute_iou(pred_box.unsqueeze(0), target_box.unsqueeze(0)).item()

                    if iou > best_iou:
                        best_iou = iou
                        best_pred = pred

            else:
                # Confidence-based selection (fallback)
                best_conf = -1.0
                for pred in predictions:
                    if pred.ocr is None:
                        continue
                    if pred.ocr.confidence > best_conf:
                        best_conf = pred.ocr.confidence
                        best_pred = pred

            if best_pred is None or best_pred.ocr is None:
                results.append(ALPRResult(text=None, confidence=0.0))
            else:
                results.append(
                    ALPRResult(
                        text=best_pred.ocr.text,
                        confidence=float(best_pred.ocr.confidence),
                    )
                )

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Black-box adversarial patch optimization (fast-alpr)"
    )
    parser.add_argument("--csv", type=str, default="preproc_labels.csv")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--blur-sigma", type=float, default=7.5,
                        help="Initial blur sigma suggestion (speeds up calibration)")

    args = parser.parse_args()

    print("=" * 60)
    print("Black-Box Adversarial Patch Optimization")
    print("Target: fast-alpr (true black box)")
    print("Ground truth plate:", GROUND_TRUTH_PLATE)
    print("Selection: IoU with ground truth corners")
    print("=" * 60)

    # Ground truth dictionary: same label for all indices
    # optimize_patch_bb only needs index -> string mapping
    import pandas as pd
    df = pd.read_csv(args.csv)
    ground_truth = {idx: GROUND_TRUTH_PLATE for idx in range(len(df))}

    black_box = LargeFastALPR(device=args.device)

    results = optimize_patch_bb(
        black_box=black_box,
        csv_path=args.csv,
        ground_truth_texts=ground_truth,
        num_epochs=args.epochs,
        device=args.device,
        learning_rate=args.lr,
        blur_sigma_init=args.blur_sigma,
        save_interval=args.save_interval,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("Optimization Summary")
    print("=" * 60)
    print(f"Initial blur sigma: {results['initial_blur_sigma']:.2f}")
    print(f"Final blur sigma: {results['blur_sigma']:.2f}")
    print(f"Epochs completed: {len(results['history']['epoch'])}/{args.epochs}")
    if results['history']['bb_success_rate']:
        print(f"Final black-box success rate: {results['history']['bb_success_rate'][-1]:.1%}")

    history_df = pd.DataFrame(results["history"])
    history_df.to_csv("bb_optimization_history.csv", index=False)

    print("\nHistory saved to bb_optimization_history.csv")
    print("Patches saved to bb_patches/ and bb_patches_final/")


if __name__ == "__main__":
    main()
