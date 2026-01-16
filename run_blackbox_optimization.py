#!/usr/bin/env python3
"""
Black-box adversarial patch optimization using larger fast-alpr models as target.

This script uses the larger fast-alpr models as a black-box target:
- Detection: yolo-v9-s-608-license-plate-end2end (vs small yolo-v9-t-384)
- OCR: cct-s-v1-global-model (vs small cct-xs-v1-global)

Usage:
    python run_blackbox_optimization.py --csv preproc_labels.csv --epochs 100
"""

import argparse
from pathlib import Path
from typing import List, Dict

import torch
import onnx
import onnx2torch
from open_image_models import LicensePlateDetector
from fast_plate_ocr import ONNXPlateRecognizer

from model_extraction import (
    BlackBoxModel,
    ALPRResult,
    optimize_patch_bb,
)


class LargeFastALPR(BlackBoxModel):
    """
    Black-box ALPR using larger fast-alpr models.

    Uses:
    - yolo-v9-s-608-license-plate-end2end (608x608 input, larger backbone)
    - cct-s-v1-global-model (larger OCR model)
    """

    def __init__(self, device: str = None, confidence_threshold: float = 0.25):
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device

        self.confidence_threshold = confidence_threshold

        print("Loading large fast-alpr models...")

        # Load larger detection model
        # This triggers download if not cached
        LicensePlateDetector(detection_model="yolo-v9-s-608-license-plate-end2end")

        detector_path = (
            Path.home() / ".cache/open-image-models/yolo-v9-s-608-license-plate-end2end"
            / "yolo-v9-s-608-license-plates-end2end.onnx"
        )

        if not detector_path.exists():
            raise FileNotFoundError(f"Detector not found: {detector_path}")

        print(f"  Loading detector: {detector_path.name}")
        detector_onnx = onnx.load(str(detector_path))
        self.detector = onnx2torch.convert(detector_onnx)
        self.detector.to(self.device)
        self.detector.eval()

        # Freeze detector
        for param in self.detector.parameters():
            param.requires_grad = False

        # Load larger OCR model
        # Use the fast_plate_ocr interface which handles model loading
        print("  Loading OCR: cct-s-v1-global-model")
        self.ocr = ONNXPlateRecognizer("cct-s-v1-global-model")

        # Also load as torch model for direct inference
        ocr_path = Path.home() / ".cache/fast-plate-ocr/cct-s-v1-global-model/cct_s_v1_global.onnx"
        if ocr_path.exists():
            ocr_onnx = onnx.load(str(ocr_path))
            self.ocr_torch = onnx2torch.convert(ocr_onnx)
            self.ocr_torch.to(self.device)
            self.ocr_torch.eval()
            for param in self.ocr_torch.parameters():
                param.requires_grad = False
        else:
            self.ocr_torch = None

        self.input_size = 608  # Larger model uses 608x608
        self.alphabet = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_'

        print(f"  Models loaded on {self.device}")

    def _preprocess_for_detection(self, image: torch.Tensor) -> torch.Tensor:
        """Preprocess image for 608x608 detection model."""
        import torch.nn.functional as F

        # image is [C, H, W] in [0, 1]
        C, H, W = image.shape

        # Calculate scaling to fit in 608x608
        scale = min(self.input_size / H, self.input_size / W)
        new_h, new_w = int(H * scale), int(W * scale)

        # Resize
        resized = F.interpolate(
            image.unsqueeze(0),
            size=(new_h, new_w),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

        # Pad to 608x608
        pad_h = self.input_size - new_h
        pad_w = self.input_size - new_w
        pad_top = pad_h // 2
        pad_left = pad_w // 2

        padded = F.pad(
            resized,
            (pad_left, pad_w - pad_left, pad_top, pad_h - pad_top),
            mode='constant',
            value=0.5  # Gray padding
        )

        return padded, (scale, pad_left, pad_top)

    def _crop_plate(self, image: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
        """Crop plate region for OCR."""
        import torch.nn.functional as F

        C, H, W = image.shape
        x1, y1, x2, y2 = box.int().tolist()

        # Clamp to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = image[:, y1:y2, x1:x2]

        # Resize to OCR input size (140x70 for cct-s)
        resized = F.interpolate(
            crop.unsqueeze(0),
            size=(70, 140),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

        return resized

    def _decode_ocr(self, logits: torch.Tensor) -> str:
        """Decode OCR logits to text."""
        probs = torch.softmax(logits, dim=-1)
        pred_indices = torch.argmax(probs, dim=-1).squeeze(0)

        text = ""
        for idx in pred_indices:
            char = self.alphabet[idx.item()]
            if char != '_':
                text += char

        return text

    def evaluate(self, images: List[torch.Tensor]) -> List[ALPRResult]:
        """
        Evaluate images through the large fast-alpr pipeline.

        Args:
            images: List of [C, H, W] tensors in [0, 1] range

        Returns:
            List of ALPRResult with detected text and confidence
        """
        results = []

        with torch.no_grad():
            for image in images:
                # Preprocess for detection
                prep_image, (scale, pad_x, pad_y) = self._preprocess_for_detection(image)
                prep_image = prep_image.to(self.device)

                # Run detection
                detections = self.detector(prep_image.unsqueeze(0))

                if len(detections) == 0:
                    results.append(ALPRResult(text=None, confidence=0.0))
                    continue

                # Find best detection by confidence
                best_det = None
                best_conf = 0.0

                for det in detections:
                    conf = det[6].item()
                    if conf > best_conf and conf >= self.confidence_threshold:
                        best_conf = conf
                        best_det = det

                if best_det is None:
                    results.append(ALPRResult(text=None, confidence=0.0))
                    continue

                # Transform box back to original image coordinates
                box = best_det[1:5].clone()
                box[0] = (box[0] - pad_x) / scale
                box[1] = (box[1] - pad_y) / scale
                box[2] = (box[2] - pad_x) / scale
                box[3] = (box[3] - pad_y) / scale

                # Crop plate from original image
                plate_crop = self._crop_plate(image, box)

                if plate_crop is None:
                    results.append(ALPRResult(text=None, confidence=best_conf))
                    continue

                # Run OCR
                if self.ocr_torch is not None:
                    # Use torch model directly
                    ocr_input = plate_crop.to(self.device)
                    # OCR expects NHWC format * 255
                    ocr_input = ocr_input.unsqueeze(0).permute(0, 2, 3, 1) * 255
                    ocr_output = self.ocr_torch(ocr_input)
                    text = self._decode_ocr(ocr_output)
                else:
                    # Fall back to ONNX interface
                    import numpy as np
                    plate_np = (plate_crop.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    text = self.ocr.run(plate_np)

                results.append(ALPRResult(text=text, confidence=best_conf))

        return results


def load_ground_truth(csv_path: str) -> Dict[int, str]:
    """Load ground truth plate texts from dataset CSV."""
    import pandas as pd

    df = pd.read_csv(csv_path)

    # Assuming 'plate_text' or 'text' column exists
    text_col = None
    for col in ['plate_text', 'text', 'label', 'plate']:
        if col in df.columns:
            text_col = col
            break

    if text_col is None:
        raise ValueError(f"Could not find plate text column in {csv_path}. "
                        f"Available columns: {list(df.columns)}")

    ground_truth = {}
    for idx, row in df.iterrows():
        text = str(row[text_col]).upper().strip()
        ground_truth[idx] = text

    print(f"Loaded {len(ground_truth)} ground truth labels from '{text_col}' column")
    return ground_truth


def main():
    parser = argparse.ArgumentParser(
        description='Black-box adversarial patch optimization against large fast-alpr'
    )
    parser.add_argument('--csv', type=str, default='preproc_labels.csv',
                        help='Path to dataset CSV')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of optimization epochs')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate for patch optimization')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda, mps, cpu)')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--confidence-threshold', type=float, default=0.25,
                        help='Detection confidence threshold')
    args = parser.parse_args()

    print("=" * 60)
    print("Black-Box Adversarial Patch Optimization")
    print("Target: Large fast-alpr models")
    print("  - Detection: yolo-v9-s-608-license-plate-end2end")
    print("  - OCR: cct-s-v1-global-model")
    print("=" * 60)

    # Load ground truth
    ground_truth = load_ground_truth(args.csv)

    # Create black-box model
    black_box = LargeFastALPR(
        device=args.device,
        confidence_threshold=args.confidence_threshold
    )

    # Run optimization
    results = optimize_patch_bb(
        black_box=black_box,
        csv_path=args.csv,
        ground_truth_texts=ground_truth,
        num_epochs=args.epochs,
        device=args.device,
        learning_rate=args.lr,
        save_interval=args.save_interval,
        verbose=True
    )

    # Print summary
    print("\n" + "=" * 60)
    print("Optimization Summary")
    print("=" * 60)
    print(f"Initial blur sigma: {results['initial_blur_sigma']:.2f}")
    print(f"Final blur sigma: {results['blur_sigma']:.2f}")
    print(f"Final black-box success rate: {results['history']['bb_success_rate'][-1]:.1%}")

    # Save history
    import pandas as pd
    history_df = pd.DataFrame(results['history'])
    history_df.to_csv('bb_optimization_history.csv', index=False)
    print("\nHistory saved to bb_optimization_history.csv")
    print("Patches saved to bb_patches/ and bb_patches_final/")


if __name__ == "__main__":
    main()
