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
        Evaluate images using fast-alpr with batch processing.

        Detection is chosen by:
        - IoU with ground truth corners (when corners provided)
        - Highest confidence (when corners not provided)
        """
        results: List[ALPRResult] = []
        batch_size = 32

        # Convert all images to BGR numpy arrays
        images_bgr = []
        valid_indices = []
        for idx, image in enumerate(images):
            # torch [C,H,W] -> uint8 BGR numpy
            img_np = image.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)

            if img_np.ndim != 3 or img_np.shape[2] != 3:
                results.append(ALPRResult(text=None, confidence=0.0))
                continue

            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            images_bgr.append(img_bgr)
            valid_indices.append(idx)

        if not images_bgr:
            return results

        # Process in batches
        for batch_start in range(0, len(images_bgr), batch_size):
            batch_end = min(batch_start + batch_size, len(images_bgr))
            batch_images = images_bgr[batch_start:batch_end]
            batch_indices = valid_indices[batch_start:batch_end]

            try:
                # Batch detection
                batch_predictions = self.alpr.detector.detector.predict(batch_images)
            except Exception as e:
                print(f"ALPR batch inference failed: {e}")
                for _ in batch_indices:
                    results.append(ALPRResult(text=None, confidence=0.0))
                continue

            # Collect all cropped plates from this batch for batch OCR
            batch_crops = []
            crop_metadata = []  # Track (image_idx, detection_idx) for each crop

            for local_idx, (pred_list, global_idx) in enumerate(zip(batch_predictions, batch_indices)):
                if not pred_list:
                    results.append(ALPRResult(text=None, confidence=0.0))
                    continue

                # Crop plates for OCR
                for det_idx, pred in enumerate(pred_list):
                    if pred.bounding_box is None:
                        continue

                    x1 = max(0, int(pred.bounding_box.x1))
                    y1 = max(0, int(pred.bounding_box.y1))
                    x2 = min(batch_images[local_idx].shape[1], int(pred.bounding_box.x2))
                    y2 = min(batch_images[local_idx].shape[0], int(pred.bounding_box.y2))

                    cropped = batch_images[local_idx][y1:y2, x1:x2]
                    if cropped.size > 0:
                        batch_crops.append(cropped)
                        crop_metadata.append((global_idx, det_idx, pred))

            if not batch_crops:
                # No valid detections in this batch
                for local_idx, global_idx in enumerate(batch_indices):
                    results.append(ALPRResult(text=None, confidence=0.0))
                continue

            # Batch OCR on all crops
            try:
                ocr_texts, ocr_confidences = self.alpr.ocr.ocr_model.run(batch_crops, return_confidence=True)
            except Exception as e:
                print(f"OCR inference failed: {e}")
                # Fill results with None for all images with detections
                processed = set()
                for global_idx, _, _ in crop_metadata:
                    if global_idx not in processed:
                        results.append(ALPRResult(text=None, confidence=0.0))
                        processed.add(global_idx)
                continue

            # Select best detection per image based on IoU or confidence
            image_results = {}  # global_idx -> (text, confidence, det_metadata)

            for crop_idx, (global_idx, det_idx, pred) in enumerate(crop_metadata):
                ocr_text = ocr_texts[crop_idx].strip('_')  # Remove padding
                ocr_conf = float(np.max(ocr_confidences[crop_idx])) if len(ocr_confidences) > crop_idx else 0.0

                if global_idx not in image_results:
                    image_results[global_idx] = (ocr_text, ocr_conf, pred)
                else:
                    # Keep better result based on selection criterion
                    if corners is not None and global_idx < len(corners):
                        # IoU-based selection
                        gt_corners = corners[global_idx]
                        target_box = corners_to_bbox(gt_corners)

                        # Compare current vs stored
                        pred_box_new = torch.tensor([
                            pred.bounding_box.x1, pred.bounding_box.y1,
                            pred.bounding_box.x2, pred.bounding_box.y2
                        ], dtype=torch.float32)
                        iou_new = compute_iou(pred_box_new.unsqueeze(0), target_box.unsqueeze(0)).item()

                        old_text, old_conf, old_pred = image_results[global_idx]
                        pred_box_old = torch.tensor([
                            old_pred.bounding_box.x1, old_pred.bounding_box.y1,
                            old_pred.bounding_box.x2, old_pred.bounding_box.y2
                        ], dtype=torch.float32)
                        iou_old = compute_iou(pred_box_old.unsqueeze(0), target_box.unsqueeze(0)).item()

                        if iou_new > iou_old:
                            image_results[global_idx] = (ocr_text, ocr_conf, pred)
                    else:
                        # Confidence-based selection
                        if ocr_conf > image_results[global_idx][1]:
                            image_results[global_idx] = (ocr_text, ocr_conf, pred)

            # Add results in order
            for global_idx in batch_indices:
                if global_idx in image_results:
                    text, conf, _ = image_results[global_idx]
                    results.append(ALPRResult(text=text if text else None, confidence=conf))
                else:
                    results.append(ALPRResult(text=None, confidence=0.0))

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
