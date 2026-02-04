#!/usr/bin/env python3
"""
Black-box adversarial patch optimization using fast-alpr as a true black box
and CMA-ES for searching the generator latent space.

Uses blackbox_constrained_search to optimize patches by searching over the
generator's latent codes.
"""

import argparse
from typing import Optional
import numpy as np
import torch

try:
    from fast_alpr import ALPR
except ImportError:
    ALPR = None

from blackbox_constrained_search import BaseBlackBoxOracle, BlackBoxPatchOptimizer


def polygon_iou(corners1: np.ndarray, corners2: np.ndarray) -> float:
    """
    Calculate IoU between two quadrilaterals.

    Args:
        corners1: [4, 2] array of (x, y) corners
        corners2: [4, 2] array of (x, y) corners

    Returns:
        IoU score in [0, 1]
    """
    try:
        from shapely.geometry import Polygon
        from shapely.validation import make_valid
    except ImportError:
        raise ImportError("shapely required for IoU calculation. Install with: pip install shapely")

    poly1 = Polygon(corners1)
    poly2 = Polygon(corners2)

    # Make valid in case of self-intersecting polygons
    if not poly1.is_valid:
        poly1 = make_valid(poly1)
    if not poly2.is_valid:
        poly2 = make_valid(poly2)

    intersection = poly1.intersection(poly2).area
    union = poly1.union(poly2).area

    if union == 0:
        return 0.0

    return intersection / union


def bbox_to_corners(detection) -> np.ndarray:
    """
    Convert detection result to corner array.

    Args:
        detection: DetectionResult object from fast-alpr with bounding_box attribute

    Returns:
        [4, 2] array of corners in format [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
    """
    bbox = detection.bounding_box
    return np.array([
        [bbox.x1, bbox.y1],
        [bbox.x2, bbox.y1],
        [bbox.x2, bbox.y2],
        [bbox.x1, bbox.y2]
    ], dtype=np.float32)


class FastALPROracle(BaseBlackBoxOracle):
    """Black-box oracle using fast-alpr for license plate detection and recognition."""

    def __init__(self, device: str = None, ocr_only: bool = False):
        """
        Initialize fast-alpr oracle.

        Args:
            device: Device to use (ignored, fast-alpr manages its own)
            ocr_only: If True, skip YOLO detector and run OCR-only
        """
        if ALPR is None:
            raise ImportError(
                "fast-alpr not installed. Install with: pip install fast-alpr"
            )

        print("Loading fast-alpr (true black box)...")
        self.ocr_only = ocr_only

        if ocr_only:
            self.alpr = ALPR(
                detector=None,
                ocr_model="cct-s-v1-global-model",
            )
            print("fast-alpr loaded (OCR-only mode)")
        else:
            self.alpr = ALPR(
                detector_model="yolo-v9-s-608-license-plate-end2end",
                ocr_model="cct-s-v1-global-model",
            )
            print("fast-alpr loaded (full pipeline)")

    def query(self, image: np.ndarray, corners: Optional[np.ndarray] = None) -> Optional[str]:
        """
        Query fast-alpr for license plate recognition.

        Args:
            image: RGB image [H, W, 3], uint8, range [0, 255]
                   In OCR-only mode: should be pre-cropped to plate region
            corners: Optional [4, 2] array of ground truth plate corners for IoU-based selection
                     (only used in full pipeline mode)

        Returns:
            Detected license plate text, or None if no detection/recognition
        """
        import cv2

        try:
            if self.ocr_only:
                # OCR-only mode: image should be pre-cropped, run OCR directly
                ocr_result = self.alpr.ocr.predict(image)
                return ocr_result.text if ocr_result is not None else None
            else:
                # Full pipeline mode: YOLO detection + OCR
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                predictions = self.alpr.predict(image_bgr)

                if not predictions:
                    return None

                # If corners provided, select detection with highest IoU to ground truth
                if corners is not None:
                    best_pred = None
                    best_iou = 0.0

                    for pred in predictions:
                        if pred.detection is None:
                            continue

                        # Convert detection bbox to corners
                        det_corners = bbox_to_corners(pred.detection)

                        # Calculate IoU with ground truth corners
                        iou = polygon_iou(det_corners, corners)

                        if iou > best_iou:
                            best_iou = iou
                            best_pred = pred

                    # If no detection has reasonable IoU, treat as no detection
                    if best_iou < 0.1:  # Threshold for minimum overlap
                        return None

                else:
                    # Fallback: use highest-confidence detection if no corners provided
                    best_pred = max(
                        predictions,
                        key=lambda p: p.ocr.confidence if p.ocr else 0.0,
                        default=None
                    )

                if best_pred is None or best_pred.ocr is None:
                    return None

                return best_pred.ocr.text
        except Exception as e:
            print(f"Warning: ALPR inference failed: {e}")
            return None


def main():
    parser = argparse.ArgumentParser(
        description="Black-box adversarial patch optimization using CMA-ES"
    )
    parser.add_argument(
        "--generator-checkpoint",
        required=True,
        help="Path to frozen generator checkpoint (.pt file)"
    )
    parser.add_argument(
        "--refinement-checkpoint",
        default=None,
        help="Path to refinement checkpoint (.pt file, optional)"
    )
    parser.add_argument(
        "--disable-refiner",
        action="store_true",
        help="Disable refinement network even if checkpoint provided"
    )
    parser.add_argument(
        "--generator-type",
        choices=["simple", "foundation"],
        default="simple",
        help="Generator architecture type (default: simple, auto-detected from checkpoint if mismatch detected)"
    )
    parser.add_argument(
        "--csv",
        default="preproc_labels.csv",
        help="CSV file with image paths and corners (default: preproc_labels.csv)"
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="Device to use for patch generation/refinement"
    )
    parser.add_argument(
        "--target-plate",
        default=None,
        help="Target plate for impersonation (None for disruption mode)"
    )
    parser.add_argument(
        "--sigma0",
        type=float,
        default=0.3,
        help="Initial CMA-ES standard deviation (default: 0.3)"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Maximum CMA-ES iterations (default: 100)"
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=None,
        help="CMA-ES population size (default: 4 + 3*log(latent_dim))"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for CMA-ES reproducibility (default: None)"
    )
    parser.add_argument(
        "--test-image-subset",
        type=int,
        default=None,
        help="Sample this many test images per iteration (default: use all)"
    )
    parser.add_argument(
        "--output-patch",
        default="optimized_patch_cmaes.png",
        help="Output path for optimized patch"
    )
    parser.add_argument(
        "--output-latent",
        default="optimized_latent_cmaes.npy",
        help="Output path for optimized latent code"
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Directory to save checkpoints at each best fitness (default: None)"
    )
    parser.add_argument(
        "--disable-homography",
        action="store_true",
        help="Disable homography-based insertion, use simple rectangular blending instead"
    )
    parser.add_argument(
        "--ocr-mode",
        action="store_true",
        help="Crop to border region (1.4x scaled corners) for OCR-only evaluation"
    )
    parser.add_argument(
        "--border-scale",
        type=float,
        default=1.4,
        help="Scale factor for border region when using --ocr-mode (default: 1.4)"
    )
    parser.add_argument(
        "--enable-plate-blur",
        action="store_true",
        help="Enable adaptive Gaussian blur on plate area (increases at fitness>0.95, decreases at fitness<0.8)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("BLACK-BOX ADVERSARIAL PATCH OPTIMIZATION (CMA-ES)")
    print("=" * 70)
    print(f"Generator checkpoint: {args.generator_checkpoint}")
    if args.refinement_checkpoint:
        print(f"Refinement checkpoint: {args.refinement_checkpoint}")
    print(f"CSV file: {args.csv}")
    print(f"Evaluation mode: {'OCR (cropped plates)' if args.ocr_mode else 'Standard (full images)'}")
    if args.ocr_mode:
        print(f"  Border scale: {args.border_scale}")
    print(f"Attack mode: {'Disruption' if args.target_plate is None else 'Impersonation'}")
    if args.target_plate:
        print(f"Target plate: {args.target_plate}")
    print(f"Patch insertion: {'Rectangular (simple blending)' if args.disable_homography else 'Homography (perspective-aware)'}")
    print(f"Adaptive plate blur: {'enabled (increases at fitness>0.95, decreases at fitness<0.8)' if args.enable_plate_blur else 'disabled'}")
    print(f"CMA-ES parameters: sigma0={args.sigma0}, max_iterations={args.max_iterations}, ", end="")
    print(f"population_size={args.population_size or 'auto'}, seed={args.seed or 'None'}")
    print("=" * 70)
    print()

    # Initialize optimizer
    print("Initializing optimizer...")
    refinement_checkpoint = None if args.disable_refiner else args.refinement_checkpoint
    optimizer = BlackBoxPatchOptimizer(
        generator_checkpoint=args.generator_checkpoint,
        refinement_checkpoint=refinement_checkpoint,
        generator_type=args.generator_type,
        device=args.device,
        csv_path=args.csv,
        target_plate=args.target_plate,
        disruption_mode=(args.target_plate is None),
        test_image_subset=args.test_image_subset,
        use_homography=not args.disable_homography,
        ocr_mode=args.ocr_mode,
        border_scale=args.border_scale,
        enable_plate_blur=args.enable_plate_blur
    )

    # Create fast-alpr oracle
    print("Initializing fast-alpr oracle...")
    oracle = FastALPROracle(ocr_only=args.ocr_mode)

    # Run optimization
    print("\n" + "=" * 70)
    best_z, best_fitness = optimizer.optimize(
        oracle,
        sigma0=args.sigma0,
        max_iterations=args.max_iterations,
        population_size=args.population_size,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir
    )

    # Save results
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    optimizer.save_patch(best_z, args.output_patch)
    np.save(args.output_latent, best_z)
    print(f"Latent code saved to: {args.output_latent}")

    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)
    print(f"Best fitness achieved: {best_fitness:.4f}")
    print(f"Optimized patch: {args.output_patch}")
    print(f"Optimized latent code: {args.output_latent}")
    print()


if __name__ == "__main__":
    main()
