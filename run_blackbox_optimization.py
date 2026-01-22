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


class FastALPROracle(BaseBlackBoxOracle):
    """Black-box oracle using fast-alpr for license plate detection and recognition."""

    def __init__(self, device: str = None):
        """
        Initialize fast-alpr oracle.

        Args:
            device: Device to use (ignored, fast-alpr manages its own)
        """
        if ALPR is None:
            raise ImportError(
                "fast-alpr not installed. Install with: pip install fast-alpr"
            )

        print("Loading fast-alpr (true black box)...")

        self.alpr = ALPR(
            detector_model="yolo-v9-s-608-license-plate-end2end",
            ocr_model="cct-s-v1-global-model",
        )

        print("fast-alpr loaded successfully")

    def query(self, image: np.ndarray) -> Optional[str]:
        """
        Query fast-alpr for license plate detection and recognition.

        Args:
            image: RGB image [H, W, 3], uint8, range [0, 255]

        Returns:
            Detected license plate text, or None if no detection
        """
        import cv2

        # Convert RGB to BGR for OpenCV/fast-alpr
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            predictions = self.alpr.predict(image_bgr)
        except Exception as e:
            print(f"Warning: ALPR inference failed: {e}")
            return None

        if not predictions:
            return None

        # Return text from highest-confidence detection
        best_pred = max(
            predictions,
            key=lambda p: p.ocr.confidence if p.ocr else 0.0,
            default=None
        )

        if best_pred is None or best_pred.ocr is None:
            return None

        return best_pred.ocr.text


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
        "--generator-type",
        choices=["simple", "foundation"],
        default="simple",
        help="Generator architecture type"
    )
    parser.add_argument(
        "--test-images-dir",
        required=True,
        help="Directory with test images and corner annotations"
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
        "--output-patch",
        default="optimized_patch_cmaes.png",
        help="Output path for optimized patch"
    )
    parser.add_argument(
        "--output-latent",
        default="optimized_latent_cmaes.npy",
        help="Output path for optimized latent code"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("BLACK-BOX ADVERSARIAL PATCH OPTIMIZATION (CMA-ES)")
    print("=" * 70)
    print(f"Generator checkpoint: {args.generator_checkpoint}")
    if args.refinement_checkpoint:
        print(f"Refinement checkpoint: {args.refinement_checkpoint}")
    print(f"Test images directory: {args.test_images_dir}")
    print(f"Mode: {'Disruption' if args.target_plate is None else 'Impersonation'}")
    if args.target_plate:
        print(f"Target plate: {args.target_plate}")
    print(f"CMA-ES parameters: sigma0={args.sigma0}, max_iterations={args.max_iterations}")
    print("=" * 70)
    print()

    # Initialize optimizer
    print("Initializing optimizer...")
    optimizer = BlackBoxPatchOptimizer(
        generator_checkpoint=args.generator_checkpoint,
        refinement_checkpoint=args.refinement_checkpoint,
        generator_type=args.generator_type,
        device=args.device,
        test_images_dir=args.test_images_dir,
        target_plate=args.target_plate,
        disruption_mode=(args.target_plate is None)
    )

    # Create fast-alpr oracle
    print("Initializing fast-alpr oracle...")
    oracle = FastALPROracle()

    # Run optimization
    print("\n" + "=" * 70)
    best_z, best_fitness = optimizer.optimize(
        oracle,
        sigma0=args.sigma0,
        max_iterations=args.max_iterations
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
