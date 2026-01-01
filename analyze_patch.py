#!/usr/bin/env python3

import os
import sys
import cv2
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Use non-interactive backend to prevent display blocking
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from PIL import Image
import torchvision.transforms as T
import kornia.geometry as K
from fast_alpr import ALPR
from typing import List, Dict, Tuple, Optional
import argparse
import warnings
warnings.filterwarnings("ignore")

# Register HEIC support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    print("HEIC support enabled")
except ImportError:
    print("Warning: pillow-heif not installed, HEIC files may not load")


class PatchEvaluator:
    """Evaluate adversarial patches against the original ALPR detection model"""

    def __init__(self, csv_path: str, patch_file: str, device: str = None,
                 impersonating_plate: str = None):
        """Initialize the patch evaluator

        Args:
            csv_path: Path to CSV file with image data
            patch_file: Path to patch image file
            device: Device to use for computation
            impersonating_plate: Target license plate number to track impersonation attempts
        """
        # Set device
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

        # Store impersonating plate number
        self.impersonating_plate = impersonating_plate
        if self.impersonating_plate is not None:
            print(f"Tracking impersonation attempts for plate: '{self.impersonating_plate}'")

        # Load dataset
        self.df = pd.read_csv(csv_path)
        if len(self.df) == 0:
            raise ValueError(f"No data found in CSV file: {csv_path}")

        print(f"Loaded {len(self.df)} images from dataset")

        # Handle different CSV formats
        if 'filename' in self.df.columns and 'processed_filename' not in self.df.columns:
            self.df['processed_filename'] = self.df['filename']

        # If alpr_conf is missing, we'll compute it during evaluation
        self.precompute_baseline = 'alpr_conf' not in self.df.columns
        if self.precompute_baseline:
            print("Note: alpr_conf not in CSV - will compute baseline confidence during evaluation")

        # Verify required columns exist
        required_columns = ['processed_filename', 'p1_x', 'p1_y', 'p2_x', 'p2_y',
                            'p3_x', 'p3_y', 'p4_x', 'p4_y', 'H00', 'H01', 'H02', 'H10', 'H11',
                            'H12', 'H20', 'H21', 'H22']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise KeyError(f"Missing required columns in CSV: {missing_columns}\n"
                           f"Available columns: {list(self.df.columns)}")

        # Load patch
        self.patch_tensor = self._load_patch(patch_file)
        self.patch_height, self.patch_width = self.patch_tensor.shape[1], self.patch_tensor.shape[2]
        print(f"Loaded patch: {self.patch_width}×{self.patch_height}")

        # Initialize ALPR detector (using fast_alpr)
        print("Loading ALPR detection model...")
        self.alpr = ALPR(
            detector_model="yolo-v9-s-608-license-plate-end2end",
            ocr_model="cct-xs-v1-global-model"
        )
        print("ALPR model loaded successfully")

        # Image preprocessing
        self.transform = T.Compose([T.ToTensor()])

    def _load_patch(self, patch_file: str) -> torch.Tensor:
        """Load patch from image file"""
        if not os.path.exists(patch_file):
            raise FileNotFoundError(f"Patch file not found: {patch_file}")

        try:
            patch_img = Image.open(patch_file).convert('RGB')
            print(f"Original patch image size: {patch_img.size}")

            # Convert to tensor and normalize to [0,1]
            patch_tensor = T.ToTensor()(patch_img).to(self.device)
            patch_tensor = torch.clamp(patch_tensor, 0, 1)

            print(f"Patch tensor shape: {patch_tensor.shape}")
            print(f"Patch value range: [{patch_tensor.min():.3f}, {patch_tensor.max():.3f}]")

            return patch_tensor

        except Exception as e:
            raise RuntimeError(
                f"Failed to load patch from {patch_file}: {type(e).__name__}: {str(e)}")

    def _load_image(self, image_path: str) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Load image at original resolution, return tensor and size"""
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            image = Image.open(image_path).convert('RGB')
            original_size = image.size  # (width, height)
            # Keep original resolution - don't resize
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            return image_tensor, original_size
        except Exception as e:
            raise RuntimeError(f"Failed to load image {image_path}: {type(e).__name__}: {str(e)}")

    def _extract_homography_matrix(self, row) -> torch.Tensor:
        """Extract homography matrix from CSV row"""
        try:
            H = np.array([
                [row['H00'], row['H01'], row['H02']],
                [row['H10'], row['H11'], row['H12']],
                [row['H20'], row['H21'], row['H22']]
            ], dtype=np.float32)

            if np.any(np.isnan(H)) or np.any(np.isinf(H)):
                raise ValueError(f"Invalid homography matrix contains NaN or Inf values: \n{H}")

            return torch.from_numpy(H).unsqueeze(0).to(self.device)
        except Exception as e:
            raise ValueError(f"Failed to extract homography matrix: {type(e).__name__}: {str(e)}\n"
                             f"Row data: H00={row.get('H00')}, H01={row.get('H01')}, ..., H22={row.get('H22')}")

    def _get_license_plate_corners(self, row, original_size: Tuple[int, int] = None) -> torch.Tensor:
        """Get license plate corner coordinates in original image space"""
        try:
            corners = torch.tensor([
                [row['p1_x'], row['p1_y']],
                [row['p2_x'], row['p2_y']],
                [row['p3_x'], row['p3_y']],
                [row['p4_x'], row['p4_y']]
            ], dtype=torch.float32, device=self.device).unsqueeze(0)

            # Corners are already in original image coordinates - no scaling needed

            if torch.any(torch.isnan(corners)):
                raise ValueError(f"Corner coordinates contain NaN values: {corners}")

            # Note: bounds check removed since we're working at original resolution
            # which can be any size (e.g., 3000x4000 for HEIC images)

            return corners
        except Exception as e:
            raise ValueError(f"Failed to extract corner coordinates: {type(e).__name__}: {str(e)}\n"
                             f"Row data: p1=({row.get('p1_x')},{row.get('p1_y')}), p2=({row.get('p2_x')},{row.get('p2_y')}), "
                             f"p3=({row.get('p3_x')},{row.get('p3_y')}), p4=({row.get('p4_x')},{row.get('p4_y')})")

    def _apply_patch_to_image(self, image: torch.Tensor, homography: torch.Tensor,
                              corners: torch.Tensor, border_scale: float = 1.4) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply patch as border around license plate using homography transformation"""
        batch_size = image.shape[0]
        # Get actual image dimensions (C, H, W format)
        img_h, img_w = image.shape[2], image.shape[3]

        try:
            # Get the 4 corners of the license plate
            plate_corners = corners[0]  # [4, 2]

            # Calculate center and create larger border quad
            center_x = plate_corners[:, 0].mean()
            center_y = plate_corners[:, 1].mean()
            center = torch.tensor([center_x, center_y], device=self.device)

            border_corners = center.unsqueeze(
                0) + (plate_corners - center.unsqueeze(0)) * border_scale
            border_corners = border_corners.unsqueeze(0)  # [1, 4, 2]

            # Create patch corner coordinates in patch space
            patch_h, patch_w = self.patch_height, self.patch_width
            src_corners = torch.tensor([
                [0, 0], [patch_w, 0], [patch_w, patch_h], [0, patch_h]
            ], dtype=torch.float32, device=self.device).unsqueeze(0)

            # Compute perspective transformation matrices
            M_border = K.get_perspective_transform(src_corners, border_corners)
            M_plate = K.get_perspective_transform(src_corners, corners)

            # Create and warp patch - use actual image dimensions
            patch_batch = self.patch_tensor.unsqueeze(0).repeat(batch_size, 1, 1, 1)
            warped_patch = K.warp_perspective(
                patch_batch, M_border, dsize=(img_h, img_w),
                mode='bilinear', padding_mode='zeros', align_corners=True
            )

            # Create masks - use actual image dimensions
            patch_mask = torch.ones(batch_size, 1, self.patch_height, self.patch_width,
                                    dtype=torch.float32, device=self.device)

            warped_border_mask = K.warp_perspective(
                patch_mask, M_border, dsize=(img_h, img_w),
                mode='bilinear', padding_mode='zeros', align_corners=True
            )

            warped_plate_mask = K.warp_perspective(
                patch_mask, M_plate, dsize=(img_h, img_w),
                mode='bilinear', padding_mode='zeros', align_corners=True
            )

            # Final mask: border area minus license plate area
            final_mask = torch.clamp(warped_border_mask - warped_plate_mask, 0, 1)
            final_mask = final_mask.expand(-1, 3, -1, -1)

            # Apply patch with cutout
            result_image = image * (1 - final_mask) + warped_patch * final_mask
            result_image = torch.clamp(result_image, 0, 1)

            return result_image, final_mask

        except Exception as e:
            raise RuntimeError(f"Patch application failed: {type(e).__name__}: {str(e)}\n"
                               f"Image shape: {image.shape}, Corners: {corners.shape}, Homography: {homography.shape}")

    def _detect_license_plates(self, image_tensor: torch.Tensor) -> List[Dict]:
        """Run ALPR detection on image using fast_alpr"""
        try:
            # Convert tensor to numpy array in RGB format
            image_np = image_tensor[0].permute(1, 2, 0).cpu().numpy()
            image_np = (image_np * 255).astype(np.uint8)

            # Convert RGB to BGR for OpenCV/ALPR (fast_alpr expects BGR format)
            cv_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            # Ensure the array is in the correct format (H, W, C)
            if cv_image.shape[2] != 3:
                raise ValueError(f"Expected 3-channel image, got shape: {cv_image.shape}")

            # Run ALPR detection using fast_alpr
            predictions = self.alpr.predict(cv_image)

            detections = []
            if predictions:
                for pred in predictions:
                    detection = pred.detection
                    ocr = pred.ocr
                    bbox = detection.bounding_box

                    detections.append({
                        'bbox': [float(bbox.x1), float(bbox.y1), float(bbox.x2), float(bbox.y2)],
                        'confidence': float(ocr.confidence),
                        'text': ocr.text,
                        'text_confidence': float(ocr.confidence)  # Same as confidence for fast_alpr
                    })

            return detections

        except Exception as e:
            # Provide detailed error information for debugging
            error_details = []
            error_details.append(f"Original error: {type(e).__name__}: {str(e)}")

            try:
                image_np = image_tensor[0].permute(1, 2, 0).cpu().numpy()
                image_np = (image_np * 255).astype(np.uint8)
                error_details.append(f"Image tensor shape: {image_tensor.shape}")
                error_details.append(f"Converted numpy shape: {image_np.shape}")
                error_details.append(f"Numpy dtype: {image_np.dtype}")
                error_details.append(f"Numpy value range: [{image_np.min()}, {image_np.max()}]")

                # Try conversion to BGR
                cv_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                error_details.append(f"BGR image shape: {cv_image.shape}")
                error_details.append(f"BGR dtype: {cv_image.dtype}")

            except Exception as conv_error:
                error_details.append(
                    f"Failed to analyze tensor: {type(conv_error).__name__}: {str(conv_error)}")

            full_error_msg = "\n".join(error_details)
            raise RuntimeError(f"ALPR detection failed:\n{full_error_msg}")

    def _compute_iou(self, box1: List[float], box2: List[float]) -> float:
        """Compute IoU between two bounding boxes"""
        try:
            x1_inter = max(box1[0], box2[0])
            y1_inter = max(box1[1], box2[1])
            x2_inter = min(box1[2], box2[2])
            y2_inter = min(box1[3], box2[3])

            if x2_inter <= x1_inter or y2_inter <= y1_inter:
                return 0.0

            inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)

            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union_area = area1 + area2 - inter_area

            if union_area <= 0:
                return 0.0

            return inter_area / union_area

        except Exception as e:
            raise ValueError(f"IoU computation failed: {type(e).__name__}: {str(e)}\n"
                             f"Box1: {box1}, Box2: {box2}")

    def _corners_to_bbox(self, corners: torch.Tensor) -> List[float]:
        """Convert corner coordinates to bounding box [x1, y1, x2, y2]"""
        corners_np = corners[0].cpu().numpy()  # [4, 2]
        x_coords = corners_np[:, 0]
        y_coords = corners_np[:, 1]
        return [float(x_coords.min()), float(y_coords.min()),
                float(x_coords.max()), float(y_coords.max())]

    def evaluate_patch(self, output_dir: str = "patch_evaluation_results") -> pd.DataFrame:
        """Evaluate patch on all images in dataset"""
        Path(output_dir).mkdir(exist_ok=True)

        results = []
        failed_images = []

        print(f"\nEvaluating patch on {len(self.df)} images...")

        # Process each image with progress bar
        for idx in tqdm(range(len(self.df)), desc="Processing images"):
            row = self.df.iloc[idx]
            image_path = row['processed_filename']

            # Get or compute original confidence
            if self.precompute_baseline:
                # Load original image and detect to get baseline
                try:
                    orig_image, orig_size = self._load_image(image_path)
                    orig_detections = self._detect_license_plates(orig_image)
                    corners = self._get_license_plate_corners(row, orig_size)
                    ground_truth_bbox = self._corners_to_bbox(corners)

                    # Find best detection
                    best_orig_detection = None
                    best_orig_iou = 0.0
                    for det in orig_detections:
                        iou = self._compute_iou(det['bbox'], ground_truth_bbox)
                        if iou > best_orig_iou:
                            best_orig_iou = iou
                            best_orig_detection = det

                    original_confidence = best_orig_detection['confidence'] if best_orig_detection else 0.0
                except Exception as e:
                    print(f"Error computing baseline for {image_path}: {e}")
                    original_confidence = 0.0
            else:
                original_confidence = float(row['alpr_conf'])

            try:
                # Load image and metadata
                image, orig_size = self._load_image(image_path)
                corners = self._get_license_plate_corners(row, orig_size)
                homography = self._extract_homography_matrix(row)
                ground_truth_bbox = self._corners_to_bbox(corners)

                # Apply patch to image
                patched_image, patch_mask = self._apply_patch_to_image(image, homography, corners)
                if patch_mask is None:
                    raise RuntimeError("Patch application returned None mask")

                # Run detection on patched image
                detections = self._detect_license_plates(patched_image)

                # Find best detection (highest IoU with ground truth)
                best_detection = None
                best_iou = 0.0

                for detection in detections:
                    iou = self._compute_iou(detection['bbox'], ground_truth_bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_detection = detection

                # Record results
                if best_detection is not None:
                    patched_confidence = best_detection['confidence']
                    confidence_change = patched_confidence - original_confidence
                    detection_text = best_detection['text']
                    text_confidence = best_detection['text_confidence']
                else:
                    patched_confidence = 0.0
                    confidence_change = -original_confidence
                    detection_text = ""
                    text_confidence = 0.0
                    best_iou = 0.0

                results.append({
                    'image_path': image_path,
                    'image_index': idx,
                    'original_confidence': original_confidence,
                    'patched_confidence': patched_confidence,
                    'confidence_change': confidence_change,
                    'confidence_change_pct': (confidence_change / original_confidence * 100) if original_confidence > 0 else 0,
                    'best_iou': best_iou,
                    'ground_truth_bbox': ground_truth_bbox,
                    'detected_bbox': best_detection['bbox'] if best_detection else None,
                    'detection_text': detection_text,
                    'text_confidence': text_confidence,
                    'num_detections': len(detections),
                    'patch_applied': True,
                    'error': None
                })

            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                print(f"\nERROR processing image {idx} ({image_path}): {error_msg}")

                failed_images.append({
                    'image_path': image_path,
                    'image_index': idx,
                    'error': error_msg
                })

                # Add failed result to maintain dataset consistency
                results.append({
                    'image_path': image_path,
                    'image_index': idx,
                    'original_confidence': original_confidence,
                    'patched_confidence': 0.0,
                    'confidence_change': -original_confidence,
                    'confidence_change_pct': -100.0,
                    'best_iou': 0.0,
                    'ground_truth_bbox': None,
                    'detected_bbox': None,
                    'detection_text': "",
                    'text_confidence': 0.0,
                    'num_detections': 0,
                    'patch_applied': False,
                    'error': error_msg
                })

        # Convert to DataFrame
        results_df = pd.DataFrame(results)

        # Save detailed results
        results_path = Path(output_dir) / "patch_evaluation_results.csv"
        results_df.to_csv(results_path, index=False)
        print(f"\nDetailed results saved to: {results_path}")

        # Save failed images summary
        if failed_images:
            failed_df = pd.DataFrame(failed_images)
            failed_path = Path(output_dir) / "failed_images.csv"
            failed_df.to_csv(failed_path, index=False)
            print(f"Failed images summary saved to: {failed_path}")
            print(f"WARNING: {len(failed_images)} images failed processing")

        return results_df

    def create_visualizations(self, results_df: pd.DataFrame,
                              output_dir: str = "patch_evaluation_results"):
        """Create comprehensive visualizations of patch evaluation results"""

        # Filter out failed images for analysis
        valid_results = results_df[results_df['patch_applied'] == True].copy()

        if len(valid_results) == 0:
            raise ValueError("No valid results found for visualization")

        print(f"\nCreating visualizations for {len(valid_results)} valid results...")

        # Set style
        plt.style.use('default')
        sns.set_palette("husl")

        # Create comprehensive figure - expand to 4x3 to fit OCR accuracy pie chart
        fig = plt.figure(figsize=(24, 18))

        # 1. Confidence change distribution
        ax1 = plt.subplot(3, 4, 1)
        sns.histplot(valid_results['confidence_change'], bins=30, kde=True, ax=ax1)
        ax1.axvline(0, color='red', linestyle='--', alpha=0.7, label='No change')
        ax1.set_xlabel('Confidence Change')
        ax1.set_ylabel('Count')
        ax1.set_title('Distribution of Confidence Changes')
        ax1.legend()

        # 2. Confidence change percentage
        ax2 = plt.subplot(3, 4, 2)
        sns.histplot(valid_results['confidence_change_pct'], bins=30, kde=True, ax=ax2)
        ax2.axvline(0, color='red', linestyle='--', alpha=0.7, label='No change')
        ax2.set_xlabel('Confidence Change (%)')
        ax2.set_ylabel('Count')
        ax2.set_title('Distribution of Confidence Changes (%)')
        ax2.legend()

        # 3. IoU distribution
        ax3 = plt.subplot(3, 4, 3)
        sns.histplot(valid_results['best_iou'], bins=30, kde=True, ax=ax3)
        ax3.axvline(0.5, color='orange', linestyle='--', alpha=0.7, label='IoU = 0.5')
        ax3.set_xlabel('Best IoU with Ground Truth')
        ax3.set_ylabel('Count')
        ax3.set_title('Distribution of Best IoU Scores')
        ax3.legend()

        # 4. OCR Accuracy Pie Chart - MODIFIED to include impersonating detection
        ax4 = plt.subplot(3, 4, 4)

        # Categorize by OCR accuracy (including impersonating plate if specified)
        def categorize_ocr_result(row):
            if row['best_iou'] == 0:
                return 'Eliminated'
            elif row['detection_text'] == 'VRJ7774':
                return 'Correct Read'
            elif self.impersonating_plate is not None and row['detection_text'] == self.impersonating_plate:
                return 'Impersonating Read'
            elif row['detection_text'] != '' and row['detection_text'] != 'VRJ7774':
                # Only count as incorrect if it's not the impersonating plate (already
                # handled above)
                if self.impersonating_plate is None or row['detection_text'] != self.impersonating_plate:
                    return 'Incorrect Read'
                else:
                    # This shouldn't happen due to logic above, but fail loudly if it does
                    raise ValueError(f"Unexpected OCR categorization state for row with detection_text='{row['detection_text']}' "
                                     f"and impersonating_plate='{self.impersonating_plate}'")
            else:
                return 'Detection No OCR'  # Detected but no text extracted

        valid_results['ocr_category'] = valid_results.apply(categorize_ocr_result, axis=1)
        ocr_counts = valid_results['ocr_category'].value_counts()

        # Colors for pie chart (including new impersonating category)
        ocr_colors = {
            'Eliminated': '#d62728',        # Red - best outcome for adversarial patch
            'Incorrect Read': '#ff7f0e',    # Orange - partial success
            'Impersonating Read': '#9467bd',  # Purple - successful impersonation
            'Detection No OCR': '#ffbb78',  # Light orange - detection but OCR failed
            'Correct Read': '#2ca02c'       # Green - patch failed, system worked correctly
        }

        colors_list = [ocr_colors.get(cat, '#gray') for cat in ocr_counts.index]

        wedges, texts, autotexts = ax4.pie(ocr_counts.values, labels=ocr_counts.index,
                                           autopct='%1.1f%%', colors=colors_list, startangle=90)

        # Create title based on whether impersonating is enabled
        title_text = 'OCR Accuracy After Patch Application\n(Target: VRJ7774'
        if self.impersonating_plate is not None:
            title_text += f', Impersonating: {self.impersonating_plate}'
        title_text += ')'
        ax4.set_title(title_text)

        # Make percentage text more readable
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)

        # 5. Original vs Patched Confidence
        ax5 = plt.subplot(3, 4, 5)
        ax5.scatter(valid_results['original_confidence'], valid_results['patched_confidence'],
                    alpha=0.6, s=20)
        max_conf = max(valid_results['original_confidence'].max(),
                       valid_results['patched_confidence'].max())
        ax5.plot([0, max_conf], [0, max_conf], 'r--', alpha=0.7, label='No change line')
        ax5.set_xlabel('Original Confidence')
        ax5.set_ylabel('Patched Confidence')
        ax5.set_title('Original vs Patched Confidence')
        ax5.legend()

        # 6. Confidence change vs Original confidence
        ax6 = plt.subplot(3, 4, 6)
        ax6.scatter(valid_results['original_confidence'], valid_results['confidence_change'],
                    alpha=0.6, s=20)
        ax6.axhline(0, color='red', linestyle='--', alpha=0.7, label='No change')
        ax6.set_xlabel('Original Confidence')
        ax6.set_ylabel('Confidence Change')
        ax6.set_title('Confidence Change vs Original Confidence')
        ax6.legend()

        # 7. IoU vs Confidence change
        ax7 = plt.subplot(3, 4, 7)
        ax7.scatter(valid_results['best_iou'], valid_results['confidence_change'],
                    alpha=0.6, s=20)
        ax7.axhline(0, color='red', linestyle='--', alpha=0.7, label='No change')
        ax7.axvline(0.5, color='orange', linestyle='--', alpha=0.7, label='IoU = 0.5')
        ax7.set_xlabel('Best IoU')
        ax7.set_ylabel('Confidence Change')
        ax7.set_title('IoU vs Confidence Change')
        ax7.legend()

        # 8. Number of detections distribution
        ax8 = plt.subplot(3, 4, 8)
        sns.countplot(data=valid_results, x='num_detections', ax=ax8)
        ax8.set_xlabel('Number of Detections')
        ax8.set_ylabel('Count')
        ax8.set_title('Distribution of Detection Counts')

        # 9. Success rate by confidence bins
        ax9 = plt.subplot(3, 4, 9)
        # Define success as IoU > 0.1 (some overlap with ground truth)
        valid_results['detected'] = valid_results['best_iou'] > 0.1
        conf_bins = pd.cut(valid_results['original_confidence'], bins=10)
        success_rates = valid_results.groupby(conf_bins)['detected'].mean()
        success_rates.plot(kind='bar', ax=ax9, rot=45)
        ax9.set_xlabel('Original Confidence Bins')
        ax9.set_ylabel('Detection Success Rate')
        ax9.set_title('Detection Success Rate by Original Confidence')

        # 10. OCR Results by Original Confidence
        ax10 = plt.subplot(3, 4, 10)
        ocr_by_conf = pd.crosstab(pd.cut(valid_results['original_confidence'], bins=5),
                                  valid_results['ocr_category'], normalize='index') * 100
        ocr_by_conf.plot(
            kind='bar', ax=ax10, rot=45, color=[
                ocr_colors.get(
                    col, 'gray') for col in ocr_by_conf.columns])
        ax10.set_xlabel('Original Confidence Bins')
        ax10.set_ylabel('Percentage (%)')
        ax10.set_title('OCR Results by Original Confidence')
        ax10.legend(title='OCR Result', bbox_to_anchor=(1.05, 1), loc='upper left')

        # 11. Summary statistics text
        ax11 = plt.subplot(3, 4, 11)
        ax11.axis('off')

        # Calculate summary stats including OCR accuracy
        total_images = len(valid_results)
        failed_images = len(results_df) - len(valid_results)
        avg_conf_change = valid_results['confidence_change'].mean()
        avg_conf_change_pct = valid_results['confidence_change_pct'].mean()
        median_iou = valid_results['best_iou'].median()
        detections_eliminated = (valid_results['best_iou'] == 0).sum()
        strong_reduction = (valid_results['confidence_change'] < -0.1).sum()

        # OCR-specific stats (including impersonating if specified)
        correct_reads = (valid_results['detection_text'] == 'VRJ7774').sum()
        incorrect_reads = ((valid_results['detection_text'] != 'VRJ7774') &
                           (valid_results['detection_text'] != '') &
                           (valid_results['best_iou'] > 0)).sum()

        # Calculate impersonating reads if specified
        impersonating_reads = 0
        if self.impersonating_plate is not None:
            impersonating_reads = (
                valid_results['detection_text'] == self.impersonating_plate).sum()
            # Subtract impersonating reads from incorrect reads since they're now separate
            incorrect_reads = ((valid_results['detection_text'] != 'VRJ7774') &
                               (valid_results['detection_text'] != self.impersonating_plate) &
                               (valid_results['detection_text'] != '') &
                               (valid_results['best_iou'] > 0)).sum()

        summary_text = f"""Patch Evaluation Summary

Total Images Processed: {total_images}
Failed Images: {failed_images}

Average Confidence Change: {avg_conf_change:.4f}
Average Confidence Change %: {avg_conf_change_pct:.1f}%

Median IoU: {median_iou:.3f}
Detections Eliminated: {detections_eliminated} ({detections_eliminated/total_images*100:.1f}%)
Strong Reduction (>0.1): {strong_reduction} ({strong_reduction/total_images*100:.1f}%)

OCR ACCURACY ANALYSIS:
Eliminated Detection: {detections_eliminated} ({detections_eliminated/total_images*100:.1f}%)
Correct Read "VRJ7774": {correct_reads} ({correct_reads/total_images*100:.1f}%)"""

        if self.impersonating_plate is not None:
            summary_text += f"""
Impersonating Read "{self.impersonating_plate}": {impersonating_reads} ({impersonating_reads/total_images*100:.1f}%)"""

        summary_text += f"""
Other Incorrect OCR: {incorrect_reads} ({incorrect_reads/total_images*100:.1f}%)

Patch Success Rate: {((detections_eliminated + incorrect_reads + impersonating_reads)/total_images*100):.1f}%"""

        ax11.text(0.05, 0.95, summary_text, transform=ax11.transAxes, fontsize=11,
                  verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        # 12. Detailed OCR breakdown text
        ax12 = plt.subplot(3, 4, 12)
        ax12.axis('off')

        # Show some examples of incorrect readings (excluding impersonating if specified)
        filter_condition = ((valid_results['detection_text'] != 'VRJ7774') &
                            (valid_results['detection_text'] != '') &
                            (valid_results['best_iou'] > 0))

        if self.impersonating_plate is not None:
            filter_condition = filter_condition & (
                valid_results['detection_text'] != self.impersonating_plate)

        incorrect_examples = valid_results[filter_condition]['detection_text'].value_counts().head(
            8)

        ocr_detail_text = f"""OCR Reading Details

TARGET PLATE: "VRJ7774\""""

        if self.impersonating_plate is not None:
            ocr_detail_text += f"""
IMPERSONATING: "{self.impersonating_plate}\""""

        ocr_detail_text += f"""

Most Common Other Misreadings:"""

        if len(incorrect_examples) > 0:
            for text, count in incorrect_examples.items():
                ocr_detail_text += f"\n  '{text}': {count} times"
        else:
            ocr_detail_text += f"\n  No other misreadings found!"

        ocr_detail_text += f"""

Detection Categories:
• RED (Eliminated): Complete detection failure
• GREEN (Correct): Patch failed, correct read"""

        if self.impersonating_plate is not None:
            ocr_detail_text += f"""
• PURPLE (Impersonating): Successfully read as "{self.impersonating_plate}\""""

        ocr_detail_text += f"""
• ORANGE (Other Incorrect): Wrong plate number
• LIGHT ORANGE: Detection but no OCR text

Patch Effectiveness:
• Total Disruption: {((detections_eliminated + incorrect_reads + impersonating_reads)/total_images*100):.1f}%
• Elimination Only: {(detections_eliminated/total_images*100):.1f}%"""

        if self.impersonating_plate is not None:
            ocr_detail_text += f"""
• Impersonation Success: {(impersonating_reads/total_images*100):.1f}%"""

        ocr_detail_text += f"""
• Other Misreading: {(incorrect_reads/total_images*100):.1f}%"""

        ax12.text(0.05, 0.95, ocr_detail_text, transform=ax12.transAxes, fontsize=10,
                  verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        plt.tight_layout()

        # Save visualization
        viz_path = Path(output_dir) / "patch_evaluation_visualization.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close figure to free memory
        print(f"Visualization saved to: {viz_path}")

        # Create additional focused plots
        self._create_focused_plots(valid_results, output_dir)

        # Create visual examples showing actual images with patches and detections
        self._create_visual_examples(valid_results, output_dir)

    def _create_focused_plots(self, results_df: pd.DataFrame, output_dir: str):
        """Create additional focused analysis plots"""

        # Effectiveness analysis
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Categorize results by effectiveness
        def categorize_effectiveness(row):
            if row['best_iou'] == 0:
                return 'Eliminated'
            elif row['confidence_change'] < -0.1:
                return 'Strong Reduction'
            elif row['confidence_change'] < -0.05:
                return 'Moderate Reduction'
            elif abs(row['confidence_change']) <= 0.05:
                return 'Minimal Impact'
            else:
                return 'Increased'

        results_df['effectiveness'] = results_df.apply(categorize_effectiveness, axis=1)

        # Effectiveness pie chart
        effectiveness_counts = results_df['effectiveness'].value_counts()
        ax1.pie(effectiveness_counts.values, labels=effectiveness_counts.index, autopct='%1.1f%%')
        ax1.set_title('Patch Effectiveness Distribution')

        # Confidence change by effectiveness category
        sns.boxplot(data=results_df, x='effectiveness', y='confidence_change', ax=ax2)
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_title('Confidence Change by Effectiveness Category')
        ax2.axhline(0, color='red', linestyle='--', alpha=0.7)

        plt.tight_layout()

        # Save focused plots
        focused_path = Path(output_dir) / "patch_effectiveness_analysis.png"
        plt.savefig(focused_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close figure to free memory
        print(f"Effectiveness analysis saved to: {focused_path}")

    def _create_visual_examples(self, results_df: pd.DataFrame, output_dir: str):
        """Create visual examples showing actual images with patches and detection results"""

        # Categorize results by effectiveness
        def categorize_effectiveness(row):
            if row['best_iou'] == 0:
                return 'Eliminated'
            elif row['confidence_change'] < -0.1:
                return 'Strong Reduction'
            elif row['confidence_change'] < -0.05:
                return 'Moderate Reduction'
            elif abs(row['confidence_change']) <= 0.05:
                return 'Minimal Impact'
            else:
                return 'Increased'

        results_df['effectiveness'] = results_df.apply(categorize_effectiveness, axis=1)

        # Select one representative example from each category
        categories = [
            'Eliminated',
            'Strong Reduction',
            'Moderate Reduction',
            'Minimal Impact',
            'Increased']
        selected_examples = {}

        for category in categories:
            category_data = results_df[results_df['effectiveness'] == category]
            if len(category_data) > 0:
                # Select the example closest to the median for that category
                if category == 'Eliminated':
                    # For eliminated, pick one with highest original confidence (most dramatic
                    # example)
                    selected = category_data.loc[category_data['original_confidence'].idxmax()]
                else:
                    # For others, pick median confidence change within category
                    median_change = category_data['confidence_change'].median()
                    selected = category_data.loc[(
                        category_data['confidence_change'] - median_change).abs().idxmin()]

                selected_examples[category] = selected

        if not selected_examples:
            print("No examples found for visual display")
            return

        print(f"Creating visual examples for {len(selected_examples)} performance categories...")

        # Create figure for visual examples
        n_examples = len(selected_examples)
        fig, axes = plt.subplots(2, max(3, (n_examples + 1) // 2), figsize=(20, 12))
        axes = axes.flatten()

        colors = {
            'Eliminated': 'red',
            'Strong Reduction': 'orange',
            'Moderate Reduction': 'yellow',
            'Minimal Impact': 'lightblue',
            'Increased': 'lightgreen'
        }

        for idx, (category, row) in enumerate(selected_examples.items()):
            if idx >= len(axes):
                break

            ax = axes[idx]

            try:
                # Load and process image
                image, orig_size = self._load_image(row['image_path'])
                corners = self._get_license_plate_corners(self.df.iloc[row['image_index']], orig_size)
                homography = self._extract_homography_matrix(self.df.iloc[row['image_index']])

                # Apply patch
                patched_image, patch_mask = self._apply_patch_to_image(image, homography, corners)

                # Convert to numpy for display
                display_img = patched_image[0].permute(1, 2, 0).cpu().numpy()
                display_img = np.clip(display_img, 0, 1)

                # Display image
                ax.imshow(display_img)
                ax.set_title(f'{category}\nOrig: {row["original_confidence"]:.3f} → Patch: {row["patched_confidence"]:.3f}\n'
                             f'Change: {row["confidence_change"]:+.3f} | IoU: {row["best_iou"]:.3f}',
                             fontsize=10, color=colors.get(category, 'black'))

                # Draw ground truth bounding box (green)
                if row['ground_truth_bbox'] is not None:
                    gt_bbox = row['ground_truth_bbox']
                    gt_rect = patches.Rectangle(
                        (gt_bbox[0], gt_bbox[1]),
                        gt_bbox[2] - gt_bbox[0],
                        gt_bbox[3] - gt_bbox[1],
                        linewidth=3, edgecolor='green', facecolor='none',
                        label='Ground Truth'
                    )
                    ax.add_patch(gt_rect)

                # Draw detected bounding box (red) if exists
                if row['detected_bbox'] is not None and row['best_iou'] > 0:
                    det_bbox = row['detected_bbox']
                    det_rect = patches.Rectangle(
                        (det_bbox[0], det_bbox[1]),
                        det_bbox[2] - det_bbox[0],
                        det_bbox[3] - det_bbox[1],
                        linewidth=2, edgecolor='red', facecolor='none', linestyle='--',
                        label=f'Detection ({row["patched_confidence"]:.3f})'
                    )
                    ax.add_patch(det_rect)

                    # Add detection text if available
                    if row['detection_text']:
                        # Color-code the text based on what was detected
                        text_color = 'red'
                        if row['detection_text'] == 'VRJ7774':
                            text_color = 'green'  # Correct read
                        elif self.impersonating_plate is not None and row['detection_text'] == self.impersonating_plate:
                            text_color = 'purple'  # Successful impersonation

                        ax.text(det_bbox[0], det_bbox[1] - 10,
                                f'"{row["detection_text"]}"',
                                color=text_color, fontsize=8, weight='bold',
                                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
                elif row['best_iou'] == 0:
                    # Add "NO DETECTION" text
                    ax.text(0.5, 0.95, 'NO DETECTION', transform=ax.transAxes,
                            ha='center', va='top', color='red', fontsize=12, weight='bold',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))

                # Add legend for first subplot only
                if idx == 0:
                    legend_elements = [
                        patches.Patch(color='green', label='Ground Truth'),
                        patches.Patch(color='red', label='ALPR Detection')
                    ]
                    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

                ax.set_xticks([])
                ax.set_yticks([])

            except Exception as e:
                # Show error in subplot
                ax.text(0.5, 0.5, f'Error loading example:\n{type(e).__name__}\n{str(e)}',
                        transform=ax.transAxes, ha='center', va='center',
                        bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
                ax.set_title(f'{category} - ERROR', color='red')
                ax.set_xticks([])
                ax.set_yticks([])
                print(f"Error creating visual example for {category}: {e}")

        # Hide unused subplots
        for idx in range(len(selected_examples), len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle('Representative Examples by Performance Category\n'
                     'Green Box = Ground Truth, Red Dashed = ALPR Detection',
                     fontsize=14, y=0.98)
        plt.tight_layout()

        # Save visual examples
        examples_path = Path(output_dir) / "patch_visual_examples.png"
        plt.savefig(examples_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close figure to free memory
        print(f"Visual examples saved to: {examples_path}")

        # Save detailed info about selected examples
        examples_info = []
        for category, row in selected_examples.items():
            examples_info.append({
                'category': category,
                'image_path': row['image_path'],
                'image_index': row['image_index'],
                'original_confidence': row['original_confidence'],
                'patched_confidence': row['patched_confidence'],
                'confidence_change': row['confidence_change'],
                'best_iou': row['best_iou'],
                'detection_text': row['detection_text']
            })

        examples_df = pd.DataFrame(examples_info)
        examples_info_path = Path(output_dir) / "visual_examples_info.csv"
        examples_df.to_csv(examples_info_path, index=False)
        print(f"Visual examples info saved to: {examples_info_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate adversarial patches on ALPR detection')
    parser.add_argument('--csv', required=True, help='Path to CSV file with image data')
    parser.add_argument('--patch', required=True, help='Path to patch image file')
    parser.add_argument('--output', default='patch_evaluation_results',
                        help='Output directory for results and visualizations')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default=None,
                        help='Device to use for computation')
    parser.add_argument('--impersonating', type=str, default=None,
                        help='License plate number to track for impersonation attempts (e.g., "ABC123")')
    args = parser.parse_args()

    try:
        print("=== Adversarial Patch Evaluation ===")
        print(f"CSV file: {args.csv}")
        print(f"Patch file: {args.patch}")
        print(f"Output directory: {args.output}")
        if args.impersonating:
            print(f"Tracking impersonation attempts for: {args.impersonating}")

        # Initialize evaluator
        evaluator = PatchEvaluator(
            csv_path=args.csv,
            patch_file=args.patch,
            device=args.device,
            impersonating_plate=args.impersonating
        )

        # Run evaluation
        results_df = evaluator.evaluate_patch(output_dir=args.output)

        # Create visualizations
        evaluator.create_visualizations(results_df, output_dir=args.output)

        print(f"\n=== Evaluation Complete ===")
        print(f"Results saved in: {args.output}/")
        print(f"- patch_evaluation_results.csv: Detailed results")
        print(f"- patch_evaluation_visualization.png: Main analysis plots")
        print(f"- patch_effectiveness_analysis.png: Effectiveness breakdown")

        # Print quick summary
        valid_results = results_df[results_df['patch_applied'] == True]
        if len(valid_results) > 0:
            avg_change = valid_results['confidence_change'].mean()
            eliminated = (valid_results['best_iou'] == 0).sum()
            print(f"\nQuick Summary:")
            print(f"- Average confidence change: {avg_change:.4f}")
            print(
                f"- Detections eliminated: {eliminated}/{len(valid_results)} ({eliminated/len(valid_results)*100:.1f}%)")

            # Add impersonation summary if enabled
            if args.impersonating:
                impersonating_count = (valid_results['detection_text'] == args.impersonating).sum()
                print(
                    f"- Successful impersonations as '{args.impersonating}': {impersonating_count}/{len(valid_results)} ({impersonating_count/len(valid_results)*100:.1f}%)")

    except Exception as e:
        print(f"\nFATAL ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
