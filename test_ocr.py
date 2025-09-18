#!/usr/bin/env python3
"""
Compare results between standalone pipeline and ALPR package on the same image
"""

import cv2
import numpy as np
from fast_alpr import ALPR
from PIL import Image
import os
import argparse


class ALPRComparison:
    def __init__(self):
        self.alpr = None

    def initialize_alpr(self):
        """Initialize ALPR with same models as standalone pipeline"""
        print("Initializing ALPR package...")
        self.alpr = ALPR(
            detector_model="yolo-v9-t-384-license-plate-end2end",
            ocr_model="cct-xs-v1-global-model",
        )
        print("ALPR initialized successfully")

    def load_image_for_alpr(self, image_path):
        """Load image in OpenCV format for ALPR"""
        # Handle different image formats
        file_ext = os.path.splitext(image_path)[1].lower()

        if file_ext in ['.heic', '.heif']:
            try:
                from pillow_heif import register_heif_opener
                register_heif_opener()
                pil_image = Image.open(image_path).convert('RGB')
                cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            except ImportError:
                raise RuntimeError("HEIC support not available. Install pillow-heif")
        else:
            cv_image = cv2.imread(image_path)

        if cv_image is None:
            raise ValueError(f"Could not load image: {image_path}")

        print(f"Loaded image for ALPR: {cv_image.shape}")
        return cv_image

    def run_alpr_detection(self, image_path):
        """Run ALPR detection and return results"""
        if self.alpr is None:
            self.initialize_alpr()

        cv_image = self.load_image_for_alpr(image_path)

        print("Running ALPR detection...")
        predictions = self.alpr.predict(cv_image)

        print(f"ALPR found {len(predictions)} predictions")

        results = []
        for i, pred in enumerate(predictions):
            detection = pred.detection
            ocr = pred.ocr
            bbox = detection.bounding_box

            result = {
                'index': i + 1,
                'text': ocr.text,
                'ocr_confidence': ocr.confidence,
                'detection_confidence': detection.confidence,
                'bbox': {
                    'x1': int(bbox.x1),
                    'y1': int(bbox.y1),
                    'x2': int(bbox.x2),
                    'y2': int(bbox.y2),
                    'width': int(bbox.x2 - bbox.x1),
                    'height': int(bbox.y2 - bbox.y1)
                }
            }

            results.append(result)

            print(f"  Result {i+1}:")
            print(f"    Text: '{result['text']}'")
            print(f"    OCR Confidence: {result['ocr_confidence']:.4f}")
            print(f"    Detection Confidence: {result['detection_confidence']:.4f}")
            print(
                f"    Bbox: [{result['bbox']['x1']}, {result['bbox']['y1']}, {result['bbox']['x2']}, {result['bbox']['y2']}]")
            print(f"    Size: {result['bbox']['width']}x{result['bbox']['height']}")

        return results

    def save_cropped_plates(self, image_path, results, output_prefix="alpr"):
        """Save cropped license plate regions for visual inspection"""
        cv_image = self.load_image_for_alpr(image_path)

        saved_crops = []
        for i, result in enumerate(results):
            bbox = result['bbox']
            x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']

            # Crop region
            cropped = cv_image[y1:y2, x1:x2]

            if cropped.size > 0:
                # Save original crop
                crop_path = f"{output_prefix}_crop_{i+1}.png"
                cv2.imwrite(crop_path, cropped)

                # Save resized to 128x64 (OCR input size, but different aspect ratio)
                resized = cv2.resize(cropped, (128, 64))
                resized_path = f"{output_prefix}_crop_{i+1}_resized_128x64.png"
                cv2.imwrite(resized_path, resized)

                saved_crops.append({
                    'index': i + 1,
                    'original_crop': crop_path,
                    'resized_crop': resized_path,
                    'original_size': (cropped.shape[1], cropped.shape[0]),
                    'text': result['text']
                })

                print(f"Saved crops for detection {i+1}: {crop_path}, {resized_path}")

        return saved_crops


def compare_results(image_path, confidence_threshold=0.5):
    """Compare ALPR package vs standalone pipeline on same image"""
    print(f"{'='*60}")
    print(f"COMPARING ALPR vs STANDALONE PIPELINE")
    print(f"{'='*60}")
    print(f"Image: {image_path}")
    print(f"Confidence threshold: {confidence_threshold}")
    print()

    # Run ALPR package
    print("1. RUNNING ALPR PACKAGE")
    print("-" * 30)
    alpr_comparison = ALPRComparison()

    try:
        alpr_results = alpr_comparison.run_alpr_detection(image_path)
        alpr_crops = alpr_comparison.save_cropped_plates(image_path, alpr_results, "alpr")
    except Exception as e:
        print(f"ALPR failed: {e}")
        alpr_results = []
        alpr_crops = []

    print()

    # Run standalone pipeline
    print("2. RUNNING STANDALONE PIPELINE")
    print("-" * 30)

    try:
        # Try to import from current directory or use absolute path
        import sys
        import importlib.util

        # Try different ways to import
        pipeline_module = None
        for module_name in ['pipeline_test', 'license_plate_pipeline']:
            try:
                pipeline_module = importlib.import_module(module_name)
                break
            except ImportError:
                continue

        if pipeline_module is None:
            # Try loading from file directly
            pipeline_file = 'pipeline_test.py'
            if os.path.exists(pipeline_file):
                spec = importlib.util.spec_from_file_location("pipeline_test", pipeline_file)
                pipeline_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(pipeline_module)
            else:
                raise ImportError("Cannot find pipeline_test.py file")

        pipeline = pipeline_module.LicensePlateOCRPipeline()
        pipeline_results = pipeline.process_image(image_path, confidence_threshold)

    except Exception as e:
        print(f"Standalone pipeline failed: {e}")
        print("Make sure pipeline_test.py is in the current directory")
        pipeline_results = {'ocr_results': [], 'best_result': None}

    print()

    # Compare results
    print("3. COMPARISON")
    print("-" * 30)

    print(f"ALPR Package Results: {len(alpr_results)}")
    for result in alpr_results:
        print(
            f"  '{result['text']}' (det_conf: {result['detection_confidence']:.4f}, ocr_conf: {result['ocr_confidence']:.4f})")

    print(f"\nStandalone Pipeline Results: {len(pipeline_results['ocr_results'])}")
    for result in pipeline_results['ocr_results']:
        if result.get('ocr_text'):
            print(f"  '{result['ocr_text']}' (conf: {result['confidence']:.4f})")
        else:
            print(f"  ERROR: {result.get('error', 'Unknown error')}")

    # Find matching detections (by overlap)
    print(f"\n4. DETAILED COMPARISON")
    print("-" * 30)

    if alpr_results and pipeline_results['ocr_results']:
        for alpr_result in alpr_results:
            alpr_bbox = alpr_result['bbox']
            alpr_center = ((alpr_bbox['x1'] + alpr_bbox['x2']) / 2,
                           (alpr_bbox['y1'] + alpr_bbox['y2']) / 2)

            # Find closest pipeline detection
            best_match = None
            min_distance = float('inf')

            for pipeline_result in pipeline_results['ocr_results']:
                if pipeline_result.get('bbox_original_coords'):
                    p_bbox = pipeline_result['bbox_original_coords']
                    p_center = ((p_bbox[0] + p_bbox[2]) / 2,
                                (p_bbox[1] + p_bbox[3]) / 2)

                    distance = ((alpr_center[0] - p_center[0])**2 +
                                (alpr_center[1] - p_center[1])**2)**0.5

                    if distance < min_distance:
                        min_distance = distance
                        best_match = pipeline_result

            print(f"\nALPR Detection:")
            print(f"  Text: '{alpr_result['text']}'")
            print(
                f"  Bbox: [{alpr_bbox['x1']}, {alpr_bbox['y1']}, {alpr_bbox['x2']}, {alpr_bbox['y2']}]")
            print(f"  Det Conf: {alpr_result['detection_confidence']:.4f}")
            print(f"  OCR Conf: {alpr_result['ocr_confidence']:.4f}")

            if best_match and min_distance < 100:  # Within 100 pixels
                print(f"  Pipeline Match:")
                print(f"    Text: '{best_match.get('ocr_text', 'ERROR')}'")
                if best_match.get('bbox_original_coords'):
                    pb = best_match['bbox_original_coords']
                    print(f"    Bbox: [{pb[0]:.1f}, {pb[1]:.1f}, {pb[2]:.1f}, {pb[3]:.1f}]")
                print(f"    Conf: {best_match['confidence']:.4f}")
                print(f"    Distance: {min_distance:.1f} pixels")

                # Text comparison
                if alpr_result['text'] == best_match.get('ocr_text', ''):
                    print(f"    ✓ TEXT MATCHES!")
                else:
                    print(f"    ✗ TEXT DIFFERS!")
            else:
                print(f"  ✗ No close pipeline match found")
    else:
        print("No results from one or both methods to compare")

    print(f"\n5. SUMMARY")
    print("-" * 30)
    print(f"ALPR found {len(alpr_results)} plates")
    print(f"Pipeline found {len(pipeline_results['ocr_results'])} plates")

    if alpr_results and pipeline_results['ocr_results']:
        alpr_texts = [r['text'] for r in alpr_results]
        pipeline_texts = [r.get('ocr_text', '')
                          for r in pipeline_results['ocr_results'] if r.get('ocr_text')]

        common_texts = set(alpr_texts) & set(pipeline_texts)
        if common_texts:
            print(f"✓ Common detections: {list(common_texts)}")
        else:
            print(f"✗ No common text detections")
            print(f"  ALPR texts: {alpr_texts}")
            print(f"  Pipeline texts: {pipeline_texts}")

    # Debug file locations
    print(f"\nDEBUG FILES:")
    print(f"  ALPR crops: alpr_crop_*.png")
    print(f"  Pipeline crops: debug_detection_*_bbox_crop.png, debug_detection_*_ocr_input.png")


def main():
    parser = argparse.ArgumentParser(description='Compare ALPR package vs standalone pipeline')
    parser.add_argument('image_path', help='Path to test image')
    parser.add_argument('--confidence', type=float, default=0.5,
                        help='Detection confidence threshold (default: 0.5)')

    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Error: Image not found: {args.image_path}")
        return

    compare_results(args.image_path, args.confidence)


if __name__ == "__main__":
    main()
