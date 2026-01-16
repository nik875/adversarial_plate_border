#!/usr/bin/env python3
"""
Black-box adversarial patch optimization using fast-alpr as a true black box.

Ground truth plate is fixed (VRJ7774).
Detection selection is based on minimum Levenshtein distance to ground truth,
NOT highest confidence.
"""

import argparse
from typing import List
import torch
import numpy as np
import cv2

from fast_alpr import ALPR
from Levenshtein import distance as levenshtein_distance

from model_extraction import (
    BlackBoxModel,
    ALPRResult,
    optimize_patch_bb,
)


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

    def evaluate(self, images: List[torch.Tensor]) -> List[ALPRResult]:
        """
        Evaluate images using fast-alpr.
        Detection is chosen by minimum Levenshtein distance to ground truth.
        """
        results: List[ALPRResult] = []

        for image in images:
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

            best_pred = None
            best_distance = float("inf")

            for pred in predictions:
                if pred.ocr is None or not pred.ocr.text:
                    continue

                pred_text = pred.ocr.text.upper()
                dist = levenshtein_distance(pred_text, GROUND_TRUTH_PLATE)

                if dist < best_distance:
                    best_distance = dist
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
    parser.add_argument("--epochs", type=int, default=100)
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
    print("Selection: minimum Levenshtein distance")
    print("=" * 60)

    # Ground truth dictionary: same label for all indices
    # optimize_patch_bb only needs index -> string mapping
    import pandas as pd
    df = pd.read_csv(args.csv)
    ground_truth = {idx: GROUND_TRUTH_PLATE for idx in range(len(df))}

    black_box = LargeFastALPR(device=args.device)

    try:
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
        print(
            f"Final black-box success rate: "
            f"{results['history']['bb_success_rate'][-1]:.1%}"
        )

        history_df = pd.DataFrame(results["history"])
        history_df.to_csv("bb_optimization_history.csv", index=False)

        print("\nHistory saved to bb_optimization_history.csv")
        print("Patches saved to bb_patches/ and bb_patches_final/")

    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("Training interrupted by user")
        print("=" * 60)

        if 'results' in locals() and results is not None:
            print(f"Saving training history up to epoch {len(results['history']['bb_success_rate'])}...")
            history_df = pd.DataFrame(results["history"])
            history_df.to_csv("bb_optimization_history_interrupted.csv", index=False)
            print("Interrupted history saved to bb_optimization_history_interrupted.csv")
            print(f"Initial blur sigma: {results['initial_blur_sigma']:.2f}")
            print(f"Last blur sigma: {results['blur_sigma']:.2f}")
            if results['history']['bb_success_rate']:
                print(f"Last black-box success rate: {results['history']['bb_success_rate'][-1]:.1%}")
        else:
            print("Training was interrupted before completion. No history to save.")


if __name__ == "__main__":
    main()
