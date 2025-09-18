#!/usr/bin/env python3
"""
Standalone license plate detection + OCR pipeline test script.
Tests the complete flow: Image → YOLO detection → OCR cropping → Text output
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import kornia
import kornia.geometry as K
from PIL import Image
import torchvision.transforms as T
import onnx
import onnx2torch
from open_image_models import LicensePlateDetector
import argparse


class LicensePlateOCRPipeline:
    def __init__(self, device=None):
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device

        print(f"Using device: {self.device}")

        # Image preprocessing for YOLO (384x384)
        self.yolo_transform = T.Compose([
            T.Resize((384, 384)),
            T.ToTensor()
        ])

        # OCR parameters
        self.ocr_input_shape = (64, 128, 3)  # Height, Width, Channels
        self.alphabet = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_'

        # Load models
        self.load_models()

    def load_models(self):
        """Load YOLO detection model and OCR model"""
        print("Loading YOLO detection model...")

        # Initialize detector to download models
        LicensePlateDetector(detection_model="yolo-v9-t-384-license-plate-end2end")

        # Get model paths
        model_cache_dir = Path.home() / ".cache/open-image-models/yolo-v9-t-384-license-plate-end2end"
        onnx_path = model_cache_dir / "yolo-v9-t-384-license-plates-end2end.onnx"

        ocr_path = Path.home() / ".cache/fast-plate-ocr/cct-xs-v1-global-model/cct_xs_v1_global.onnx"

        # Verify paths exist
        if not onnx_path.exists():
            raise FileNotFoundError(f"YOLO ONNX model not found at: {onnx_path}")
        if not ocr_path.exists():
            raise FileNotFoundError(f"OCR ONNX model not found at: {ocr_path}")

        print(f"Loading YOLO from: {onnx_path}")
        print(f"Loading OCR from: {ocr_path}")

        # Load and convert YOLO model
        yolo_onnx = onnx.load(str(onnx_path))
        self.yolo_model = onnx2torch.convert(yolo_onnx)
        self.yolo_model.to(self.device)
        self.yolo_model.eval()

        # Load and convert OCR model
        ocr_onnx = onnx.load(str(ocr_path))
        self.ocr_model = onnx2torch.convert(ocr_onnx)
        self.ocr_model.to(self.device)
        self.ocr_model.eval()

        # Disable gradients for inference
        for param in self.yolo_model.parameters():
            param.requires_grad = False
        for param in self.ocr_model.parameters():
            param.requires_grad = False

        print("Models loaded successfully")

    def load_image(self, image_path: str) -> tuple:
        """
        Load image and return both original and YOLO-preprocessed versions
        Returns: (original_tensor, yolo_tensor, original_size)
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load original image
        pil_image = Image.open(image_path).convert('RGB')
        original_size = pil_image.size  # (width, height)

        print(f"Loaded image: {image_path}")
        print(f"Original size: {original_size[0]}x{original_size[1]}")

        # Convert to tensor (keep original size)
        original_tensor = T.ToTensor()(pil_image)

        # YOLO preprocessing (resize to 384x384)
        yolo_tensor = self.yolo_transform(pil_image)

        return original_tensor, yolo_tensor, original_size

    def detect_license_plates(self, yolo_image: torch.Tensor) -> list:
        """
        Run YOLO detection on preprocessed image
        Returns: List of detections [x1, y1, x2, y2, confidence, class_id]
        """
        print("Running YOLO detection...")

        # Add batch dimension and move to device
        batch_input = yolo_image.unsqueeze(0).to(self.device)
        print(f"YOLO input shape: {batch_input.shape}")

        with torch.no_grad():
            detections = self.yolo_model(batch_input)

        print(f"Raw YOLO output type: {type(detections)}")

        # Parse detections
        parsed_detections = []
        detection_count = 0

        for detection in detections:
            detection_count += 1

            # Extract components
            x1, y1, x2, y2 = detection[1:5].cpu()
            confidence = detection[6].cpu().item()
            class_id = detection[5].cpu().item()

            parsed_detections.append({
                'bbox': [x1.item(), y1.item(), x2.item(), y2.item()],
                'confidence': confidence,
                'class_id': int(class_id)
            })

            print(f"Detection {detection_count}: bbox=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}], "
                  f"conf={confidence:.4f}, class={int(class_id)}")

        print(f"Total detections found: {len(parsed_detections)}")
        return parsed_detections

    def bbox_to_corners(self, bbox: list) -> torch.Tensor:
        """Convert bbox [x1,y1,x2,y2] to corner format for kornia"""
        x1, y1, x2, y2 = bbox
        corners = torch.tensor([[
            [x1, y1],  # top-left
            [x2, y1],  # top-right
            [x2, y2],  # bottom-right
            [x1, y2]   # bottom-left
        ]], dtype=torch.float32)
        return corners

    def scale_bbox_to_original(self, bbox: list, original_size: tuple) -> list:
        """
        Scale bbox from YOLO coordinates (384x384) back to original image coordinates
        """
        yolo_size = 384
        orig_width, orig_height = original_size

        # Calculate scale factors
        scale_x = orig_width / yolo_size
        scale_y = orig_height / yolo_size

        # Scale bbox
        x1, y1, x2, y2 = bbox
        scaled_bbox = [
            x1 * scale_x,
            y1 * scale_y,
            x2 * scale_x,
            y2 * scale_y
        ]

        print(f"Scaled bbox from {bbox} to {[round(x, 1) for x in scaled_bbox]}")
        return scaled_bbox

    def crop_for_ocr(self, original_image: torch.Tensor, bbox: list,
                     save_debug_image: bool = True, debug_prefix: str = "debug") -> torch.Tensor:
        """
        Crop license plate region and resize for OCR input
        """
        print(f"Cropping for OCR: bbox={[round(x, 1) for x in bbox]}")

        # Convert bbox to corners format
        corners = self.bbox_to_corners(bbox)
        print(f"Corners shape: {corners.shape}")

        # Add batch dimension to image if needed
        if original_image.dim() == 3:
            image_batch = original_image.unsqueeze(0)
        else:
            image_batch = original_image

        print(f"Image batch shape: {image_batch.shape}")
        print(f"Target OCR size: {self.ocr_input_shape[:2]}")

        # Save debug image of the bbox region (before resizing)
        if save_debug_image:
            try:
                x1, y1, x2, y2 = [int(x) for x in bbox]
                # Clamp coordinates to image bounds
                x1 = max(0, min(x1, original_image.shape[2]))
                x2 = max(0, min(x2, original_image.shape[2]))
                y1 = max(0, min(y1, original_image.shape[1]))
                y2 = max(0, min(y2, original_image.shape[1]))

                if x2 > x1 and y2 > y1:  # Valid bbox
                    # Extract bbox region directly
                    bbox_crop = original_image[:, y1:y2, x1:x2]

                    # Save original bbox crop
                    if bbox_crop.numel() > 0:
                        bbox_pil = T.ToPILImage()(bbox_crop)
                        bbox_pil.save(f"{debug_prefix}_bbox_crop.png")
                        print(f"Saved bbox crop: {debug_prefix}_bbox_crop.png ({bbox_crop.shape})")

            except Exception as e:
                print(f"Warning: Could not save bbox debug image: {e}")

        # Crop and resize using kornia
        try:
            cropped = kornia.geometry.crop_and_resize(
                image_batch,                    # [1, C, H, W]
                corners,                        # [1, 4, 2]
                self.ocr_input_shape[:2],      # (H, W) - target size (64, 128)
                mode='bilinear',
                align_corners=True
            )
            print(f"Cropped plate shape: {cropped.shape}")

            # Save debug image of final OCR input
            if save_debug_image:
                try:
                    ocr_input_pil = T.ToPILImage()(cropped.squeeze(0))
                    ocr_input_pil.save(f"{debug_prefix}_ocr_input.png")
                    print(f"Saved OCR input: {debug_prefix}_ocr_input.png")
                except Exception as e:
                    print(f"Warning: Could not save OCR input debug image: {e}")

            return cropped

        except Exception as e:
            print(f"Cropping failed: {e}")
            print(f"Image shape: {image_batch.shape}")
            print(f"Corners: {corners}")
            raise RuntimeError(f"Failed to crop license plate region: {e}")

    def run_ocr(self, cropped_plate: torch.Tensor) -> str:
        """
        Run OCR on cropped license plate image
        """
        print("Running OCR...")

        # Convert from [0,1] back to [0,255] range - OCR model expects integer values!
        # This is the key fix: OCR model was trained on [0,255] not normalized [0,1]
        ocr_input = cropped_plate * 255.0

        # Move to device and convert to NHWC format for OCR model
        ocr_input = ocr_input.to(self.device).permute(0, 2, 3, 1)  # NHWC
        print(f"OCR input shape: {ocr_input.shape}")
        print(f"OCR input range: [{ocr_input.min():.1f}, {ocr_input.max():.1f}]")

        with torch.no_grad():
            ocr_logits = self.ocr_model(ocr_input)

        print(f"OCR output shape: {ocr_logits.shape}")

        # Convert logits to text
        text = self.logits_to_text(ocr_logits)
        print(f"OCR result: '{text}'")

        return text

    def logits_to_text(self, logits: torch.Tensor) -> str:
        """Convert OCR logits to readable text"""
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1)

        # Get most likely character for each position
        pred_chars = torch.argmax(probs, dim=-1).squeeze(0)

        # Convert to string, filtering out padding character '_'
        text = ""
        for char_idx in pred_chars:
            char_idx = char_idx.item()
            if char_idx < len(self.alphabet):
                char = self.alphabet[char_idx]
                if char != '_':  # Skip padding character
                    text += char

        return text.strip()

    def process_image(self, image_path: str, confidence_threshold: float = 0.5) -> dict:
        """
        Complete pipeline: Load image → Detect → OCR → Return results
        """
        print(f"\n{'='*50}")
        print(f"Processing: {image_path}")
        print(f"{'='*50}")

        # Load image
        original_image, yolo_image, original_size = self.load_image(image_path)

        # YOLO detection
        detections = self.detect_license_plates(yolo_image)

        if not detections:
            return {
                'image_path': image_path,
                'detections': [],
                'ocr_results': [],
                'best_result': None
            }

        # Filter by confidence
        valid_detections = [d for d in detections if d['confidence'] >= confidence_threshold]
        print(f"Detections above {confidence_threshold} confidence: {len(valid_detections)}")

        if not valid_detections:
            print(f"No detections meet confidence threshold of {confidence_threshold}")
            return {
                'image_path': image_path,
                'detections': detections,
                'ocr_results': [],
                'best_result': None
            }

        # Process each valid detection with OCR
        ocr_results = []

        for i, detection in enumerate(valid_detections):
            print(f"\n--- Processing detection {i+1}/{len(valid_detections)} ---")

            try:
                # Scale bbox back to original image coordinates
                scaled_bbox = self.scale_bbox_to_original(detection['bbox'], original_size)

                # Crop for OCR
                debug_prefix = f"debug_detection_{i+1}"
                cropped_plate = self.crop_for_ocr(
                    original_image,
                    scaled_bbox,
                    save_debug_image=True,
                    debug_prefix=debug_prefix)

                # Run OCR
                ocr_text = self.run_ocr(cropped_plate)

                result = {
                    'detection_index': i,
                    'bbox_yolo_coords': detection['bbox'],
                    'bbox_original_coords': scaled_bbox,
                    'confidence': detection['confidence'],
                    'ocr_text': ocr_text
                }

                ocr_results.append(result)

            except Exception as e:
                print(f"Failed to process detection {i+1}: {e}")
                # Continue with other detections rather than failing completely
                ocr_results.append({
                    'detection_index': i,
                    'bbox_yolo_coords': detection['bbox'],
                    'bbox_original_coords': None,
                    'confidence': detection['confidence'],
                    'ocr_text': None,
                    'error': str(e)
                })

        # Find best result (highest confidence with successful OCR)
        best_result = None
        for result in ocr_results:
            if result.get('ocr_text') and result['ocr_text'].strip():
                if best_result is None or result['confidence'] > best_result['confidence']:
                    best_result = result

        return {
            'image_path': image_path,
            'original_size': original_size,
            'detections': detections,
            'ocr_results': ocr_results,
            'best_result': best_result
        }


def main():
    parser = argparse.ArgumentParser(description='Test license plate detection + OCR pipeline')
    parser.add_argument('image_path', help='Path to input image')
    parser.add_argument('--confidence', type=float, default=0.5,
                        help='Minimum detection confidence (default: 0.5)')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'],
                        help='Device to use (auto-detect if not specified)')

    args = parser.parse_args()

    try:
        # Initialize pipeline
        pipeline = LicensePlateOCRPipeline(device=args.device)

        # Process image
        results = pipeline.process_image(args.image_path, args.confidence)

        # Print final results
        print(f"\n{'='*50}")
        print("FINAL RESULTS")
        print(f"{'='*50}")

        print(f"Image: {results['image_path']}")
        if 'original_size' in results:
            print(f"Original size: {results['original_size'][0]}x{results['original_size'][1]}")

        print(f"Total detections: {len(results['detections'])}")
        print(f"OCR results: {len(results['ocr_results'])}")

        if results['best_result']:
            best = results['best_result']
            print(f"\nBEST RESULT:")
            print(f"  Text: '{best['ocr_text']}'")
            print(f"  Confidence: {best['confidence']:.4f}")
            print(
                f"  Bbox (original coords): {[round(x, 1) for x in best['bbox_original_coords']]}")
        else:
            print("\nNo successful OCR results found")

        # Print all OCR results
        if results['ocr_results']:
            print(f"\nALL OCR RESULTS:")
            for i, result in enumerate(results['ocr_results']):
                if result.get('error'):
                    print(f"  {i+1}: ERROR - {result['error']}")
                else:
                    print(f"  {i+1}: '{result['ocr_text']}' (conf: {result['confidence']:.4f})")

    except Exception as e:
        print(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
