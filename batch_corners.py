#!/usr/bin/env python3
"""
License Plate Corner Detection - Batch Processing Version with YOLOv8 .pt

This version supports batch processing for faster inference:
1. YOLOv8 .pt file (custom license plate detector) with native batch support
2. SAM for precise corner extraction

Usage:
    # Single image
    python sam_plate_corners_batch.py --image photo.jpg --yolo-model plates.pt --visualize
    
    # Batch processing multiple images
    python sam_plate_corners_batch.py --input-dir ./plates --yolo-model plates.pt --batch-size 8 --output corners.csv
"""

import os
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2
import torch
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches as mpl_patches

# SAM imports
try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    print("ERROR: segment-anything not installed!")
    print("Install with: pip install git+https://github.com/facebookresearch/segment-anything.git")
    exit(1)

# YOLO imports for license plate detection
try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics not installed!")
    print("Install with: pip install ultralytics")
    exit(1)


class LicensePlateCornerDetectorBatch:
    """
    Detect license plates with YOLOv8 .pt (batch processing) and extract precise corners with SAM
    """
    
    def __init__(
        self,
        yolo_model: str = "yolov8n.pt",
        sam_checkpoint: str = "sam_vit_h_4b8939.pth",
        sam_model_type: str = "vit_h",
        device: str = None,
        conf_threshold: float = 0.25,
        batch_size: int = 4
    ):
        """
        Initialize detector
        
        Args:
            yolo_model: Path to YOLOv8 .pt model file (use license plate trained model)
            sam_checkpoint: Path to SAM checkpoint
            sam_model_type: SAM model type (vit_h, vit_l, vit_b)
            device: Device to use (cuda/mps/cpu)
            conf_threshold: Confidence threshold for detections
            batch_size: Number of images to process in parallel
        """
        # Setup device
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
        
        self.conf_threshold = conf_threshold
        self.batch_size = batch_size
        
        # Load YOLOv8 license plate model
        self.load_yolo_model(yolo_model)
        
        # Load SAM model
        print(f"Loading SAM model: {sam_model_type}")
        if not Path(sam_checkpoint).exists():
            print(f"ERROR: SAM checkpoint not found at {sam_checkpoint}")
            print("Download from: https://github.com/facebookresearch/segment-anything#model-checkpoints")
            raise FileNotFoundError(f"SAM checkpoint not found: {sam_checkpoint}")
            
        sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
        sam.to(device=self.device)
        self.sam_predictor = SamPredictor(sam)
        
        print("✓ Models loaded successfully!")

    def load_yolo_model(self, yolo_model: str):
        """Load YOLOv8 .pt model file"""
        print(f"Loading YOLO model: {yolo_model}")
        
        if not Path(yolo_model).exists():
            print(f"ERROR: YOLO model not found at {yolo_model}")
            print("\nPlease provide a license plate detection model (.pt file)")
            print("Options:")
            print("  1. Train your own:")
            print("     yolo train data=license_plates.yaml model=yolov8n.pt epochs=100")
            print("  2. Download pre-trained model:")
            print("     https://github.com/MuhammadMoinFaisal/LicensePlateDetection-YOLOv8")
            print("  3. Use Roboflow datasets:")
            print("     https://universe.roboflow.com/search?q=license%20plate")
            raise FileNotFoundError(f"YOLO model not found: {yolo_model}")
        
        try:
            self.yolo = YOLO(yolo_model)
            self.yolo.to(self.device)
            
            # Get model info
            names = self.yolo.names if hasattr(self.yolo, 'names') else {}
            print(f"✓ YOLO model loaded successfully")
            print(f"  Classes: {names if names else 'Generic detection'}")
            print(f"  Device: {self.device}")
            
        except Exception as e:
            print(f"ERROR loading YOLO model: {e}")
            raise
    
    def detect_plates_batch(self, image_paths: List[str]) -> List[List[Dict]]:
        """
        Batch detect license plates using YOLOv8
        
        Args:
            image_paths: List of image file paths to process
            
        Returns:
            List of detection lists, one per image in batch
        """
        # YOLOv8 native batch inference
        results = self.yolo(
            image_paths,
            conf=self.conf_threshold,
            verbose=False,
            device=self.device
        )
        
        all_detections = []
        
        # Parse results for each image
        for result in results:
            detections = []
            
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    # Get class name if available
                    class_name = result.names[cls] if hasattr(result, 'names') else str(cls)
                    
                    detections.append({
                        'box': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': conf,
                        'class': cls,
                        'class_name': class_name
                    })
            
            all_detections.append(detections)
        
        return all_detections
    
    def extract_corners_from_mask(
        self, 
        mask: np.ndarray,
        method: str = 'minrect'
    ) -> np.ndarray:
        """
        Extract 4 corners from a binary mask
        
        Args:
            mask: Binary mask (H, W)
            method: Corner extraction method ('contour', 'minrect', 'convex')
            
        Returns:
            Corner coordinates (4, 2) in order: [top-left, top-right, bottom-right, bottom-left]
        """
        # Ensure mask is uint8
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        if method == 'minrect':
            # Use minimum area rectangle (best for rotated plates)
            rect = cv2.minAreaRect(largest_contour)
            corners = cv2.boxPoints(rect).astype(np.float32)
            
        elif method == 'contour':
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            if len(approx) == 4:
                corners = approx.reshape(4, 2)
            else:
                # Fallback to minrect
                rect = cv2.minAreaRect(largest_contour)
                corners = cv2.boxPoints(rect).astype(np.float32)
                
        elif method == 'convex':
            # Use convex hull
            hull = cv2.convexHull(largest_contour)
            epsilon = 0.02 * cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, epsilon, True)
            
            if len(approx) == 4:
                corners = approx.reshape(4, 2)
            else:
                rect = cv2.minAreaRect(hull)
                corners = cv2.boxPoints(rect).astype(np.float32)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Sort corners: top-left, top-right, bottom-right, bottom-left
        corners = self._order_corners(corners)
        
        return corners
    
    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """Order corners in consistent order: TL, TR, BR, BL"""
        # Sort by y coordinate
        sorted_by_y = corners[corners[:, 1].argsort()]
        
        # Top two points
        top_points = sorted_by_y[:2]
        # Bottom two points
        bottom_points = sorted_by_y[2:]
        
        # Sort top points by x (left to right)
        top_left, top_right = top_points[top_points[:, 0].argsort()]
        
        # Sort bottom points by x (left to right)
        bottom_left, bottom_right = bottom_points[bottom_points[:, 0].argsort()]
        
        return np.array([top_left, top_right, bottom_right, bottom_left])
    
    def segment_plate_sam(
        self,
        image: np.ndarray,
        box: List[float],
        use_points: bool = False
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Segment license plate using SAM with box or point prompts
        
        Args:
            image: Input image (BGR format)
            box: Bounding box [x1, y1, x2, y2]
            use_points: If True, use center point prompt; if False, use box prompt
            
        Returns:
            (mask, corners) or (None, None) if segmentation fails
        """
        # Convert BGR to RGB for SAM
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Set image for SAM
        self.sam_predictor.set_image(image_rgb)
        
        if use_points:
            # Use center point of bounding box as prompt
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            point_coords = np.array([[center_x, center_y]])
            point_labels = np.array([1])  # 1 = foreground
            
            masks, scores, _ = self.sam_predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True
            )
        else:
            # Use box prompt
            box_array = np.array(box)
            
            masks, scores, _ = self.sam_predictor.predict(
                box=box_array,
                multimask_output=True
            )
        
        # Select best mask
        if len(masks) == 0:
            return None, None
            
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        
        # Extract corners from mask
        corners = self.extract_corners_from_mask(best_mask, method='minrect')
        
        return best_mask, corners
    
    def process_from_csv(
        self,
        csv_path: str,
        output_csv: str = "plate_corners.csv",
        visualize: bool = True,
        vis_dir: str = "visualizations",
        use_points: bool = False,
        filename_column: str = None
    ) -> pd.DataFrame:
        """
        Batch process images from CSV file containing filenames
        
        Args:
            csv_path: Path to CSV file with image filenames
            output_csv: Path to output CSV file
            visualize: Create visualizations
            vis_dir: Directory for visualizations
            use_points: Use point prompts for SAM
            filename_column: Column name containing filenames (auto-detect if None)
            
        Returns:
            DataFrame with all results
        """
        print(f"Loading image paths from CSV: {csv_path}")
        
        # Read CSV
        df_input = pd.read_csv(csv_path)
        
        # Auto-detect filename column if not specified
        if filename_column is None:
            # Try common column names
            common_names = ['filename', 'image', 'image_path', 'path', 'file', 'image_file']
            
            # If only one column, use that
            if len(df_input.columns) == 1:
                filename_column = df_input.columns[0]
                print(f"  Using only column: '{filename_column}'")
            else:
                # Try to find a matching column name
                for name in common_names:
                    if name in df_input.columns:
                        filename_column = name
                        print(f"  Auto-detected column: '{filename_column}'")
                        break
                
                if filename_column is None:
                    print(f"\nAvailable columns: {list(df_input.columns)}")
                    raise ValueError(
                        f"Could not auto-detect filename column. "
                        f"Please specify with --filename-column"
                    )
        else:
            if filename_column not in df_input.columns:
                print(f"\nAvailable columns: {list(df_input.columns)}")
                raise ValueError(f"Column '{filename_column}' not found in CSV")
            print(f"  Using specified column: '{filename_column}'")
        
        # Get image paths
        image_paths = df_input[filename_column].tolist()
        
        # Filter out any NaN or empty values
        image_paths = [str(p) for p in image_paths if pd.notna(p) and str(p).strip()]
        
        print(f"  Found {len(image_paths)} image paths in CSV")
        
        # Process using the batch processing logic
        return self._process_image_list(
            image_paths=image_paths,
            output_csv=output_csv,
            visualize=visualize,
            vis_dir=vis_dir,
            use_points=use_points
        )
    
    def process_directory_batch(
        self,
        input_dir: str,
        output_csv: str = "plate_corners.csv",
        visualize: bool = True,
        vis_dir: str = "visualizations",
        use_points: bool = False,
        image_extensions: List[str] = None
    ) -> pd.DataFrame:
        """
        Batch process all images in directory
        
        Args:
            input_dir: Directory containing input images
            output_csv: Path to output CSV file
            visualize: Create visualizations
            vis_dir: Directory for visualizations
            use_points: Use point prompts for SAM
            image_extensions: List of image extensions to process
            
        Returns:
            DataFrame with all results
        """
        if image_extensions is None:
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            
        # Find all images
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(Path(input_dir).glob(f"*{ext}"))
            image_paths.extend(Path(input_dir).glob(f"*{ext.upper()}"))
            
        image_paths = sorted(set(str(p) for p in image_paths))
        
        print(f"Found {len(image_paths)} images in {input_dir}")
        
        # Process using shared logic
        return self._process_image_list(
            image_paths=image_paths,
            output_csv=output_csv,
            visualize=visualize,
            vis_dir=vis_dir,
            use_points=use_points
        )
    
    def _process_image_list(
        self,
        image_paths: List[str],
        output_csv: str,
        visualize: bool,
        vis_dir: str,
        use_points: bool
    ) -> pd.DataFrame:
        """
        Internal method: Process a list of image paths
        
        Args:
            image_paths: List of image file paths
            output_csv: Path to output CSV file
            visualize: Create visualizations
            vis_dir: Directory for visualizations
            use_points: Use point prompts for SAM
            
        Returns:
            DataFrame with all results
        """
        
        print(f"Processing with batch size: {self.batch_size}")
        
        all_results = []
        
        # Process in batches
        num_batches = (len(image_paths) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
            batch_start = batch_idx * self.batch_size
            batch_end = min(batch_start + self.batch_size, len(image_paths))
            batch_paths = image_paths[batch_start:batch_end]
            
            # YOLO batch inference
            batch_detections = self.detect_plates_batch(batch_paths)
            
            # Process each image's detections with SAM
            for img_path, detections in zip(batch_paths, batch_detections):
                # Load image for SAM processing
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Warning: Could not load {img_path}")
                    continue
                
                # Process each detection
                for det_idx, detection in enumerate(detections):
                    box = detection['box']
                    conf = detection['confidence']
                    class_name = detection['class_name']
                    
                    # Segment with SAM
                    mask, corners = self.segment_plate_sam(img, box, use_points=use_points)
                    
                    if corners is not None:
                        all_results.append({
                            'image_path': img_path,
                            'detection_idx': det_idx,
                            'yolo_box': box,
                            'yolo_confidence': conf,
                            'yolo_class': class_name,
                            'corners': corners,
                            'mask': mask,
                            'image': img  # Store for visualization
                        })
                
                # Visualize if requested
                if visualize and len([r for r in all_results if r['image_path'] == img_path]) > 0:
                    img_results = [r for r in all_results if r['image_path'] == img_path]
                    self._visualize_results(img, img_results, img_path, vis_dir)
        
        # Convert to DataFrame
        if all_results:
            df_data = []
            for result in all_results:
                corners = result['corners']
                df_data.append({
                    'image_path': result['image_path'],
                    'detection_idx': result['detection_idx'],
                    'yolo_x1': result['yolo_box'][0],
                    'yolo_y1': result['yolo_box'][1],
                    'yolo_x2': result['yolo_box'][2],
                    'yolo_y2': result['yolo_box'][3],
                    'yolo_confidence': result['yolo_confidence'],
                    'yolo_class': result['yolo_class'],
                    'top_left_x': corners[0, 0],
                    'top_left_y': corners[0, 1],
                    'top_right_x': corners[1, 0],
                    'top_right_y': corners[1, 1],
                    'bottom_right_x': corners[2, 0],
                    'bottom_right_y': corners[2, 1],
                    'bottom_left_x': corners[3, 0],
                    'bottom_left_y': corners[3, 1],
                })
            
            df = pd.DataFrame(df_data)
            
            # Save to CSV
            df.to_csv(output_csv, index=False)
            print(f"\n✓ Results saved to: {output_csv}")
            print(f"  Total detections: {len(df)}")
            
            # Show class distribution
            if 'yolo_class' in df.columns:
                print(f"\nDetected classes:")
                for class_name, count in df['yolo_class'].value_counts().items():
                    print(f"  {class_name}: {count}")
            
            return df
        else:
            print("⚠ No plates detected in any images!")
            return pd.DataFrame()
    
    def process_from_csv(
        self,
        csv_path: str,
        output_csv: str = "plate_corners.csv",
        visualize: bool = True,
        vis_dir: str = "visualizations",
        use_points: bool = False,
        filename_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Process images from a CSV file containing image paths
        
        Args:
            csv_path: Path to CSV file with image filenames
            output_csv: Path to output CSV file
            visualize: Create visualizations
            vis_dir: Directory for visualizations
            use_points: Use point prompts for SAM
            filename_column: Column name containing filenames (auto-detect if None)
            
        Returns:
            DataFrame with all results
        """
        # Load CSV
        if not Path(csv_path).exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        input_df = pd.read_csv(csv_path)
        
        print(f"Loaded CSV with {len(input_df)} rows and columns: {list(input_df.columns)}")
        
        # Auto-detect filename column if not specified
        if filename_column is None:
            # Common column names for image paths
            possible_columns = ['image_path', 'filepath', 'file_path', 'filename', 
                              'image', 'path', 'img_path', 'image_file']
            
            # Check if any common column exists
            for col in possible_columns:
                if col in input_df.columns:
                    filename_column = col
                    print(f"Auto-detected filename column: '{filename_column}'")
                    break
            
            # If still not found, use first column
            if filename_column is None:
                filename_column = input_df.columns[0]
                print(f"Using first column as filename: '{filename_column}'")
        
        # Validate column exists
        if filename_column not in input_df.columns:
            raise ValueError(
                f"Column '{filename_column}' not found in CSV. "
                f"Available columns: {list(input_df.columns)}"
            )
        
        # Get image paths
        image_paths = input_df[filename_column].tolist()
        
        # Filter out NaN/None values
        image_paths = [str(p) for p in image_paths if pd.notna(p)]
        
        print(f"Found {len(image_paths)} image paths in column '{filename_column}'")
        print(f"Processing with batch size: {self.batch_size}")
        
        all_results = []
        
        # Process in batches
        num_batches = (len(image_paths) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
            batch_start = batch_idx * self.batch_size
            batch_end = min(batch_start + self.batch_size, len(image_paths))
            batch_paths = image_paths[batch_start:batch_end]
            
            # Verify paths exist
            valid_batch_paths = []
            for img_path in batch_paths:
                if Path(img_path).exists():
                    valid_batch_paths.append(img_path)
                else:
                    print(f"Warning: File not found: {img_path}")
            
            if not valid_batch_paths:
                continue
            
            # YOLO batch inference
            batch_detections = self.detect_plates_batch(valid_batch_paths)
            
            # Process each image's detections with SAM
            for img_path, detections in zip(valid_batch_paths, batch_detections):
                # Load image for SAM processing
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Warning: Could not load {img_path}")
                    continue
                
                # Process each detection
                for det_idx, detection in enumerate(detections):
                    box = detection['box']
                    conf = detection['confidence']
                    class_name = detection['class_name']
                    
                    # Segment with SAM
                    mask, corners = self.segment_plate_sam(img, box, use_points=use_points)
                    
                    if corners is not None:
                        all_results.append({
                            'image_path': img_path,
                            'detection_idx': det_idx,
                            'yolo_box': box,
                            'yolo_confidence': conf,
                            'yolo_class': class_name,
                            'corners': corners,
                            'mask': mask,
                            'image': img
                        })
                
                # Visualize if requested
                if visualize and len([r for r in all_results if r['image_path'] == img_path]) > 0:
                    img_results = [r for r in all_results if r['image_path'] == img_path]
                    self._visualize_results(img, img_results, img_path, vis_dir)
        
        # Convert to DataFrame
        if all_results:
            df_data = []
            for result in all_results:
                corners = result['corners']
                df_data.append({
                    'image_path': result['image_path'],
                    'detection_idx': result['detection_idx'],
                    'yolo_x1': result['yolo_box'][0],
                    'yolo_y1': result['yolo_box'][1],
                    'yolo_x2': result['yolo_box'][2],
                    'yolo_y2': result['yolo_box'][3],
                    'yolo_confidence': result['yolo_confidence'],
                    'yolo_class': result['yolo_class'],
                    'top_left_x': corners[0, 0],
                    'top_left_y': corners[0, 1],
                    'top_right_x': corners[1, 0],
                    'top_right_y': corners[1, 1],
                    'bottom_right_x': corners[2, 0],
                    'bottom_right_y': corners[2, 1],
                    'bottom_left_x': corners[3, 0],
                    'bottom_left_y': corners[3, 1],
                })
            
            df = pd.DataFrame(df_data)
            
            # Save to CSV
            df.to_csv(output_csv, index=False)
            print(f"\n✓ Results saved to: {output_csv}")
            print(f"  Total detections: {len(df)}")
            
            # Show class distribution
            if 'yolo_class' in df.columns:
                print(f"\nDetected classes:")
                for class_name, count in df['yolo_class'].value_counts().items():
                    print(f"  {class_name}: {count}")
            
            return df
        else:
            print("⚠ No plates detected in any images!")
            return pd.DataFrame()
    
    def _visualize_results(
        self,
        image: np.ndarray,
        results: List[Dict],
        image_path: str,
        output_dir: str
    ):
        """Create visualization of detection results"""
        os.makedirs(output_dir, exist_ok=True)
        
        fig, axes = plt.subplots(1, len(results) + 1, figsize=(6 * (len(results) + 1), 6))
        if len(results) == 0:
            return
        
        if len(results) == 0:
            axes = [axes]
        elif not isinstance(axes, np.ndarray):
            axes = [axes]
        
        # Original image with all detections
        ax = axes[0]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image_rgb)
        ax.set_title('YOLO Detections + SAM Corners')
        ax.axis('off')
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for idx, result in enumerate(results):
            color = colors[idx % len(colors)]
            
            # Draw YOLO box
            x1, y1, x2, y2 = result['yolo_box']
            rect = mpl_patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color, facecolor='none',
                linestyle='--', label=f'YOLO {idx}'
            )
            ax.add_patch(rect)
            
            # Draw SAM corners
            corners = result['corners']
            corners_closed = np.vstack([corners, corners[0]])
            ax.plot(corners_closed[:, 0], corners_closed[:, 1], 
                   color=color, linewidth=3, marker='o', markersize=8,
                   label=f'SAM {idx}')
            
            # Individual plate view
            if idx + 1 < len(axes):
                ax_detail = axes[idx + 1]
                
                # Show mask overlay
                mask_overlay = image_rgb.copy()
                mask_overlay[result['mask']] = mask_overlay[result['mask']] * 0.5 + np.array([255, 0, 0]) * 0.5
                
                ax_detail.imshow(mask_overlay.astype(np.uint8))
                
                # Draw corners
                corners = result['corners']
                corners_closed = np.vstack([corners, corners[0]])
                ax_detail.plot(corners_closed[:, 0], corners_closed[:, 1],
                              color='yellow', linewidth=3, marker='o', markersize=10)
                
                # Label corners
                labels = ['TL', 'TR', 'BR', 'BL']
                for corner, label in zip(corners, labels):
                    ax_detail.text(corner[0], corner[1] - 10, label,
                                  color='yellow', fontsize=12, weight='bold',
                                  bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
                
                ax_detail.set_title(f'Plate {idx} ({result["yolo_class"]}, {result["yolo_confidence"]:.2f})')
                ax_detail.axis('off')
        
        axes[0].legend(loc='best', fontsize=8)
        
        plt.tight_layout()
        
        # Save visualization
        base_name = Path(image_path).stem
        output_path = os.path.join(output_dir, f"{base_name}_corners.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()


def download_sam_checkpoint(model_type: str = "vit_h") -> str:
    """Download SAM checkpoint if not present"""
    checkpoints = {
        'vit_h': 'sam_vit_h_4b8939.pth',
        'vit_l': 'sam_vit_l_0b3195.pth',
        'vit_b': 'sam_vit_b_01ec64.pth'
    }
    
    urls = {
        'vit_h': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
        'vit_l': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
        'vit_b': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth'
    }
    
    checkpoint_path = checkpoints[model_type]
    
    if Path(checkpoint_path).exists():
        print(f"Using existing checkpoint: {checkpoint_path}")
        return checkpoint_path
    
    print(f"Downloading SAM {model_type} checkpoint...")
    import urllib.request
    
    url = urls[model_type]
    urllib.request.urlretrieve(url, checkpoint_path)
    
    print(f"✓ Downloaded to: {checkpoint_path}")
    return checkpoint_path


def main():
    parser = argparse.ArgumentParser(
        description='Extract precise license plate corners using YOLOv8 .pt + SAM (Batch Processing)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Batch process directory with YOLOv8 license plate model
  python sam_plate_corners_batch.py --input-dir ./plates --yolo-model license_plate.pt --batch-size 8 --output corners.csv
  
  # Process from CSV file with image filenames
  python sam_plate_corners_batch.py --input-csv image_list.csv --yolo-model license_plate.pt --output results.csv
  
  # CSV with specific column name
  python sam_plate_corners_batch.py --input-csv dataset.csv --filename-column image_path --yolo-model plates.pt
  
  # Single image processing
  python sam_plate_corners_batch.py --image plate.jpg --yolo-model license_plate.pt --visualize
  
  # Use faster SAM model for speed
  python sam_plate_corners_batch.py --input-dir ./plates --yolo-model plates.pt --sam-model vit_b --batch-size 16
  
  # No visualizations for maximum speed
  python sam_plate_corners_batch.py --input-csv images.csv --yolo-model plates.pt --batch-size 32 --no-visualize

CSV FILE FORMAT:
  Option 1 - Single column with filenames:
    filename
    image1.jpg
    image2.jpg
    /path/to/image3.png
  
  Option 2 - Multiple columns (auto-detects filename column):
    image_path,label,split
    img1.jpg,car,train
    img2.jpg,truck,val

IMPORTANT: You must provide a YOLOv8 .pt model trained on license plates!
  Train your own: yolo train data=plates.yaml model=yolov8n.pt epochs=100
  Or download pre-trained from GitHub/Roboflow
        """
    )
    
    # Input/output
    parser.add_argument('--image', type=str, help='Single image to process')
    parser.add_argument('--input-dir', type=str, help='Directory of images to process')
    parser.add_argument('--input-csv', type=str, help='CSV file with image filenames (one column)')
    parser.add_argument('--filename-column', type=str, default=None,
                       help='Column name in CSV containing filenames (auto-detect if not specified)')
    parser.add_argument('--output', type=str, default='plate_corners.csv',
                       help='Output CSV file (default: plate_corners.csv)')
    parser.add_argument('--vis-dir', type=str, default='visualizations',
                       help='Directory for visualizations (default: visualizations)')
    
    # Model configuration
    parser.add_argument('--yolo-model', type=str, required=True,
                       help='Path to YOLOv8 .pt model file (REQUIRED - must be license plate detector)')
    parser.add_argument('--sam-checkpoint', type=str, default=None,
                       help='SAM checkpoint path (auto-download if not specified)')
    parser.add_argument('--sam-model', type=str, default='vit_h',
                       choices=['vit_h', 'vit_l', 'vit_b'],
                       help='SAM model type (default: vit_h, use vit_b for speed)')
    parser.add_argument('--device', type=str, choices=['cuda', 'mps', 'cpu'],
                       help='Device to use (auto-detect if not specified)')
    
    # Detection options
    parser.add_argument('--conf-threshold', type=float, default=0.25,
                       help='YOLO confidence threshold (default: 0.25)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for parallel processing (default: 8)')
    parser.add_argument('--use-points', action='store_true',
                       help='Use point prompts for SAM instead of box prompts')
    
    # Visualization
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualizations')
    parser.add_argument('--no-visualize', action='store_true',
                       help='Skip visualizations (faster)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.image and not args.input_dir and not args.input_csv:
        parser.error("Must specify either --image, --input-dir, or --input-csv")
    
    # Download SAM checkpoint if needed
    if args.sam_checkpoint is None:
        args.sam_checkpoint = download_sam_checkpoint(args.sam_model)
    
    # Determine visualization setting
    visualize = args.visualize or (not args.no_visualize and args.image is not None)
    
    # Initialize detector
    print("\n" + "="*70)
    print("LICENSE PLATE CORNER DETECTION - BATCH PROCESSING")
    print("="*70)
    
    detector = LicensePlateCornerDetectorBatch(
        yolo_model=args.yolo_model,
        sam_checkpoint=args.sam_checkpoint,
        sam_model_type=args.sam_model,
        device=args.device,
        conf_threshold=args.conf_threshold,
        batch_size=args.batch_size
    )
    
    print("="*70 + "\n")
    
    # Process
    if args.image:
        print(f"Processing single image: {args.image}\n")
        
        # For single image, process with batch size 1
        batch_detections = detector.detect_plates_batch([args.image])
        
        # Load image for SAM
        image = cv2.imread(args.image)
        if image is None:
            print(f"ERROR: Could not load image: {args.image}")
            exit(1)
        
        results = []
        for det_idx, detection in enumerate(batch_detections[0]):
            box = detection['box']
            conf = detection['confidence']
            class_name = detection['class_name']
            
            mask, corners = detector.segment_plate_sam(image, box, use_points=args.use_points)
            
            if corners is not None:
                results.append({
                    'image_path': args.image,
                    'detection_idx': det_idx,
                    'yolo_box': box,
                    'yolo_confidence': conf,
                    'yolo_class': class_name,
                    'corners': corners,
                    'mask': mask
                })
        
        if results:
            print(f"✓ Found {len(results)} plate(s)")
            for idx, result in enumerate(results):
                print(f"\nPlate {idx}:")
                print(f"  Class: {result['yolo_class']}")
                print(f"  YOLO confidence: {result['yolo_confidence']:.3f}")
                print(f"  Corners:")
                labels = ['Top-left', 'Top-right', 'Bottom-right', 'Bottom-left']
                for label, corner in zip(labels, result['corners']):
                    print(f"    {label:12s}: ({corner[0]:.1f}, {corner[1]:.1f})")
            
            if visualize:
                detector._visualize_results(image, results, args.image, args.vis_dir)
                print(f"\n✓ Visualization saved to: {args.vis_dir}")
        else:
            print("⚠ No plates detected!")
            print("  Try lowering --conf-threshold or check if YOLO model is correct")
            
    elif args.input_dir:
        print(f"Processing directory: {args.input_dir}\n")
        df = detector.process_directory_batch(
            args.input_dir,
            output_csv=args.output,
            visualize=not args.no_visualize,
            vis_dir=args.vis_dir,
            use_points=args.use_points
        )
        
        if not df.empty:
            print(f"\n✓ Summary:")
            print(f"  Images with plates: {df['image_path'].nunique()}")
            print(f"  Total plates: {len(df)}")
            print(f"  Avg confidence: {df['yolo_confidence'].mean():.3f}")
            
            if not args.no_visualize:
                print(f"  Visualizations: {args.vis_dir}")
            
            print(f"\n✓ Processing complete!")
            print(f"  Use corners.csv for adversarial patch training")
    
    elif args.input_csv:
        print(f"Processing from CSV: {args.input_csv}\n")
        df = detector.process_from_csv(
            csv_path=args.input_csv,
            output_csv=args.output,
            visualize=not args.no_visualize,
            vis_dir=args.vis_dir,
            use_points=args.use_points,
            filename_column=args.filename_column
        )
        
        if not df.empty:
            print(f"\n✓ Summary:")
            print(f"  Images with plates: {df['image_path'].nunique()}")
            print(f"  Total plates: {len(df)}")
            print(f"  Avg confidence: {df['yolo_confidence'].mean():.3f}")
            
            if not args.no_visualize:
                print(f"  Visualizations: {args.vis_dir}")
            
            print(f"\n✓ Processing complete!")
            print(f"  Use {args.output} for adversarial patch training")


if __name__ == "__main__":
    main()