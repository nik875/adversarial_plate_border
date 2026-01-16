#!/usr/bin/env python3
"""
Black-box adversarial patch optimization using fast-alpr as a true black box.

This version uses ONLY the high-level fast_alpr.ALPR API and does not
touch model internals, ONNX, or torch inference.

Usage:
    python run_blackbox_optimization.py --csv preproc_labels.csv --epochs 100
"""

import argparse
from typing import List, Dict
from pathlib import Path

import torch
import numpy as np
import cv2

from fast_alpr import ALPR

from model_extraction import (
    BlackBoxModel,
    ALPRResult,
    optimize_patch_bb,
)


class LargeFastALPR(BlackBoxModel):
    """
    True black-box ALPR using fast-alpr high-level API.

    Models:
      - Detector: yolo-v9-s-608-license-plate-end2end
      - OCR: cct-s-v1-global-model
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

        print("Loading fast-alpr (black-box)...")

        self.alpr = ALPR(
            detector_model="yolo-v9-s-608-license-plate-end2end",
            ocr_model="cct-s-v1-global-model",
        )

        print(f"fast-alpr loaded on device: {self.device}")

    def evaluate(self, images: List[torch.Tensor]) -> List[ALPRResult]:
        """
        Evaluate images using fast-alpr as a black box.

        Args:
            images: list of torch.Tensor [C, H, W] in [0, 1]

        Returns:
            List[ALPRResult]
        """
        results: List[ALPRResult] = []

        for image in images:
            # Convert tensor → uint8 BGR numpy (fast-alpr contract)
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

            # Select best prediction by OCR confidence
            best_pred = None
            best_conf = 0.0

            for pred in predictions:
                if pred.ocr is None:
                    continue
                conf = float(pred.ocr.confidence)
                if conf > best_conf:
                    best_conf = conf
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


def load_ground_truth(csv_path: str) -> Dict[int, str]:
    """Load ground-truth plate strings from CSV."""
    import pandas as pd

    df = pd.read_csv(csv_path)

    text_col = None
    for col in ["plate_text", "text", "label", "plate"]:
        if col in df.columns:
            text_col = col
            break

    if text_col is None:
        raise ValueError(
            f"Could not find plate text column in {csv_path}. "
            f"Available columns: {list(df.columns)}"
        )

    gt = {}
    for idx, row in df.iterrows():
        gt[idx] = str(row[text_col]).upper().strip()

    print(f"Loaded {len(gt)} ground truth labels from '{text_col}' column")
    return gt


def main():
    parser = argparse.ArgumentParser(
        description="Black-box adversarial patch optimization (fast-alpr)"
    )
    parser.add_argument("--csv", type=str, default="preproc_labels.csv")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save-interval", type=int, default=10)

    args = parser.parse_args()

    print("=" * 60)
    print("Black-Box Adversarial Patch Optimization")
    print("Target: fast-alpr (true black box)")
    print("  - Detector: yolo-v9-s-608-license-plate-end2end")
    print("  - OCR: cct-s-v1-global-model")
    print("=" * 60)

    ground_truth = load_ground_truth(args.csv)

    black_box = LargeFastALPR(device=args.device)

    results = optimize_patch_bb(
        black_box=black_box,
        csv_path=args.csv,
        ground_truth_texts=ground_truth,
        num_epochs=args.epochs,
        device=args.device,
        learning_rate=args.lr,
        save_interval=args.save_interval,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("Optimization Summary")
    print("=" * 60)
    print(f"Initial blur sigma: {results['initial_blur_sigma']:.2f}")
    print(f"Final blur sigma: {results['blur_sigma']:.2f}")
    print(
        f"Final black-box success rate: "
        f"{results['history']['bb_success_rate'][-1]:.1%}"
    )

    import pandas as pd

    history_df = pd.DataFrame(results["history"])
    history_df.to_csv("bb_optimization_history.csv", index=False)

    print("\nHistory saved to bb_optimization_history.csv")
    print("Patches saved to bb_patches/ and bb_patches_final/")


if __name__ == "__main__":
    main()
