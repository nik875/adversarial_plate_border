#!/usr/bin/env python3
"""
License Plate Corner Detection using FastALPR + SAM

This script combines:
1. FastALPR for accurate license plate detection
2. SAM (Segment Anything Model) for precise segmentation and corner extraction

Features:
- Batch processing of images
- FastALPR-based plate detection (more accurate than YOLO for plates)
- Precise corner extraction from SAM masks
- CSV export with corner coordinates
- Visualization of results
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

# ONNX imports for FastALPR
try:
    import onnx
    import onnx2torch
except ImportError:
    print("ERROR: onnx or onnx2torch not installed!")
    print("Install with: pip install onnx onnx2torch")
    exit(1)


class LicensePlateCornerDetector:
    """o
    Detect license plates with FastALPR and extract precise corners with SAM
    """
    
    def __init__(
        self,
        sam_checkpoint: str = "sam_vit_h_4b8939.pth",
        sam_model_type: str = "vit_h",
        device: str = None,
        conf_threshold: float = 0.25
    ):
        """
        Initialize detector
        
        Args:
            sam_checkpoint: Path to SAM checkpoint
            sam_model_type: SAM model type (vit_h, vit_l, vit_b)
            device: Device to use (cuda/mps/cpu)
            conf_threshold: Confidence threshold for detections (FastALPR uses different scaling)
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
        
        # Load FastALPR ONNX model
        self.load_alpr_model()
        
        # Load SAM model
        print(f"Loading SAM model: {sam_model_type}")
        if not Path(sam_checkpoint).exists():
            print(f"ERROR: SAM checkpoint not found at {sam_checkpoint}")
            print("Download from: https://github.com/facebookresearch/segment-anything#model-checkpoints")
            raise FileNotFoundError(f"SAM checkpoint not found: {sam_checkpoint}")
            
        sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
        sam.to(device=self.device)
        self.sam_predictor = SamPredictor(sam)
        
        print("Models loaded successfully!")
        
    def load_alpr_model(self):
        """Load and convert FastALPR YOLO model from ONNX to PyTorch"""
        print("Loading FastALPR YOLO model from ONNX...")
        
        # Get ONNX model path from cache
        model_cache_dir = Path.home() / ".cache/open-image-models/yolo-v9-t-384-license-plate-end2end"
        onnx_path = model_cache_dir / "yolo-v9-t-384-license-plates-end2end.onnx"
        
        if not onnx_path.exists():
            print(f"ERROR: ONNX model not found at: {onnx_path}")
            print("Make sure to initialize FastALPR first with:")
            print("  from fast_plate_ocr.alpr import ALPR")
            print("  ALPR(detection_model='yolo-v9-t-384-license-plate-end2end')")
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
        
        # Load and convert ONNX model to PyTorch
        try:
            onnx_model = onnx.load(str(onnx_path))
            self.model = onnx2torch.convert(onnx_model)
            self.model.to(self.device)
            self.model.eval()
            
            # Disable gradients for inference
            for param in self.model.parameters():
                param.requires_grad = False
                
            print(f"FastALPR ONNX model loaded successfully")
        except Exception as e:
            print(f"ERROR loading ONNX model: {e}")
            raise
        
        
    def detect_plates_fastalpr(self, image: np.ndarray) -> List[Dict]:
        """
        Detect license plates using FastALPR ONNX model
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of detections with boxes and confidence scores
        """
        # Prepare image for ONNX model (same preprocessing as YOLO)
        # Resize to 384x384 (YOLOv9 input size for license plates)
        img_h, img_w = image.shape[:2]
        target_size = 384
        
        # Resize while maintaining aspect ratio
        scale = target_size / max(img_h, img_w)
        new_h, new_w = int(img_h * scale), int(img_w * scale)
        
        img_resized = cv2.resize(image, (new_w, new_h))
        
        # Pad to target size
        pad_h = target_size - new_h
        pad_w = target_size - new_w
        img_padded = cv2.copyMakeBorder(
            img_resized, 0, pad_h, 0, pad_w,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        
        # Convert BGR to RGB and normalize
        img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, 384, 384]
        
        # Run inference
        try:
            with torch.no_grad():
                img_tensor = img_tensor.to(self.device)
                model_output = self.model(img_tensor)
        except Exception as e:
            print(f"Model inference error: {e}")
            return []
        
        detections = []
        
        # Parse YOLO output (format: [x1, y1, x2, y2, conf, class_id, ...])
        # ONNX model output is typically [batch, num_detections, 6+]
        try:
            if isinstance(model_output, torch.Tensor):
                detections_tensor = model_output
            elif isinstance(model_output, (list, tuple)):
                # Handle multiple outputs
                detections_tensor = model_output[0] if len(model_output) > 0 else None
            else:
                detections_tensor = model_output
            
            if detections_tensor is None:
                return []
            
            detections_tensor = detections_tensor.cpu().numpy()
            
            # Handle different output shapes
            if len(detections_tensor.shape) == 2:
                # Shape: [num_detections, 6+]
                detections_array = detections_tensor
            elif len(detections_tensor.shape) == 3:
                # Shape: [batch, num_detections, 6+]
                detections_array = detections_tensor[0]  # Take first batch
            else:
                print(f"Unexpected ONNX output shape: {detections_tensor.shape}")
                return []
            
            # Scale coordinates back to original image size
            scale_x = img_w / new_w if new_w > 0 else 1.0
            scale_y = img_h / new_h if new_h > 0 else 1.0
            
            for detection in detections_array:
                if len(detection) < 6:
                    continue
                
                x1, y1, x2, y2 = detection[1:5]
                conf = float(detection[6])
                cls = int(detection[5]) if len(detection) > 5 else 0
                
                # Filter by confidence threshold
                if conf < self.conf_threshold:
                    continue
                
                # Scale back to original image coordinates
                x1 = float(x1 * scale_x)
                y1 = float(y1 * scale_y)
                x2 = float(x2 * scale_x)
                y2 = float(y2 * scale_y)
                
                # Clamp to image bounds
                x1 = max(0, min(x1, img_w))
                y1 = max(0, min(y1, img_h))
                x2 = max(0, min(x2, img_w))
                y2 = max(0, min(y2, img_h))
                
                detections.append({
                    'box': [x1, y1, x2, y2],
                    'confidence': conf,
                    'class': cls
                })
        except Exception as e:
            print(f"Error parsing ONNX output: {e}")
            return []
        
        return detections
    
    def extract_corners_from_mask(
        self, 
        mask: np.ndarray,
        method: str = 'contour'
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
        
        if method == 'contour':
            # Find contours
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
                
            # Get largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            # If we get exactly 4 points, use them
            if len(approx) == 4:
                corners = approx.reshape(4, 2)
            else:
                # Use minimum area rectangle
                rect = cv2.minAreaRect(largest_contour)
                corners = cv2.boxPoints(rect).astype(np.float32)
                
        elif method == 'minrect':
            # Find contours and use minimum area rectangle
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
                
            largest_contour = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(largest_contour)
            corners = cv2.boxPoints(rect).astype(np.float32)
            
        elif method == 'convex':
            # Use convex hull then approximate
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
                
            largest_contour = max(contours, key=cv2.contourArea)
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
        """
        Order corners in consistent order: TL, TR, BR, BL
        
        Args:
            corners: Unordered corners (4, 2)
            
        Returns:
            Ordered corners (4, 2)
        """
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
    
    def process_image(
        self,
        image_path: str,
        use_points: bool = False,
        visualize: bool = False,
        output_dir: Optional[str] = None
    ) -> List[Dict]:
        """
        Process single image: detect plates and extract corners
        
        Args:
            image_path: Path to input image
            use_points: Use point prompts instead of box prompts for SAM
            visualize: Create visualization
            output_dir: Directory to save visualizations
            
        Returns:
            List of results with corners for each detected plate
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        # Detect plates with FastALPR
        detections = self.detect_plates_fastalpr(image)
        
        results = []
        
        for det_idx, detection in enumerate(detections):
            box = detection['box']
            conf = detection['confidence']
            
            # Segment with SAM
            mask, corners = self.segment_plate_sam(image, box, use_points=use_points)
            
            if corners is not None:
                results.append({
                    'image_path': image_path,
                    'detection_idx': det_idx,
                    'fastalpr_box': box,
                    'fastalpr_confidence': conf,
                    'corners': corners,
                    'mask': mask
                })
        
        # Visualize if requested
        if visualize and output_dir and results:
            self._visualize_results(image, results, image_path, output_dir)
            
        return results
    
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
            axes = [axes]
        
        # Original image with all detections
        ax = axes[0]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image_rgb)
        ax.set_title('FastALPR Detections')
        ax.axis('off')
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for idx, result in enumerate(results):
            color = colors[idx % len(colors)]
            
            # Draw FastALPR box
            x1, y1, x2, y2 = result['fastalpr_box']
            rect = mpl_patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color, facecolor='none',
                linestyle='--', label=f'FastALPR {idx}'
            )
            ax.add_patch(rect)
            
            # Draw SAM corners
            corners = result['corners']
            corners_closed = np.vstack([corners, corners[0]])  # Close the polygon
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
                
                ax_detail.set_title(f'Plate {idx} (Conf: {result["fastalpr_confidence"]:.2f})')
                ax_detail.axis('off')
        
        axes[0].legend(loc='best', fontsize=8)
        
        plt.tight_layout()
        
        # Save visualization
        base_name = Path(image_path).stem
        output_path = os.path.join(output_dir, f"{base_name}_corners.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def process_directory(
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
            
        image_paths = sorted(set(image_paths))
        
        print(f"Found {len(image_paths)} images in {input_dir}")
        
        all_results = []
        
        for image_path in tqdm(image_paths, desc="Processing images"):
            try:
                results = self.process_image(
                    str(image_path),
                    use_points=use_points,
                    visualize=visualize,
                    output_dir=vis_dir if visualize else None
                )
                all_results.extend(results)
                
            except Exception as e:
                print(f"\nError processing {image_path}: {e}")
                continue
        
        # Convert to DataFrame
        if all_results:
            df_data = []
            for result in all_results:
                corners = result['corners']
                df_data.append({
                    'image_path': result['image_path'],
                    'detection_idx': result['detection_idx'],
                    'fastalpr_x1': result['fastalpr_box'][0],
                    'fastalpr_y1': result['fastalpr_box'][1],
                    'fastalpr_x2': result['fastalpr_box'][2],
                    'fastalpr_y2': result['fastalpr_box'][3],
                    'fastalpr_confidence': result['fastalpr_confidence'],
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
            print(f"\nResults saved to: {output_csv}")
            print(f"Total detections: {len(df)}")
            
            return df
        else:
            print("No plates detected in any images!")
            return pd.DataFrame()


def download_sam_checkpoint(model_type: str = "vit_h") -> str:
    """
    Download SAM checkpoint if not present
    
    Args:
        model_type: SAM model type (vit_h, vit_l, vit_b)
        
    Returns:
        Path to checkpoint
    """
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
    
    print(f"Downloaded to: {checkpoint_path}")
    return checkpoint_path


def main():
    parser = argparse.ArgumentParser(
        description='Extract precise license plate corners using FastALPR + SAM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single image
  python sam_plate_corners_fastalpr.py --image plate.jpg --visualize
  
  # Batch process directory
  python sam_plate_corners_fastalpr.py --input-dir ./plates --output corners.csv
  
  # Use lightweight SAM model
  python sam_plate_corners_fastalpr.py --input-dir ./plates --sam-model vit_b
  
  # Use point prompts instead of box prompts
  python sam_plate_corners_fastalpr.py --input-dir ./plates --use-points
        """
    )
    
    # Input/output
    parser.add_argument('--image', type=str, help='Single image to process')
    parser.add_argument('--input-dir', type=str, help='Directory of images to process')
    parser.add_argument('--output', type=str, default='plate_corners.csv',
                       help='Output CSV file (default: plate_corners.csv)')
    parser.add_argument('--vis-dir', type=str, default='visualizations',
                       help='Directory for visualizations (default: visualizations)')
    
    # Model configuration
    parser.add_argument('--sam-checkpoint', type=str, default=None,
                       help='SAM checkpoint path (auto-download if not specified)')
    parser.add_argument('--sam-model', type=str, default='vit_h',
                       choices=['vit_h', 'vit_l', 'vit_b'],
                       help='SAM model type (default: vit_h)')
    parser.add_argument('--device', type=str, choices=['cuda', 'mps', 'cpu'],
                       help='Device to use (auto-detect if not specified)')
    
    # Detection options
    parser.add_argument('--conf-threshold', type=float, default=0.25,
                       help='FastALPR confidence threshold (default: 0.25)')
    parser.add_argument('--use-points', action='store_true',
                       help='Use point prompts for SAM instead of box prompts')
    
    # Visualization
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualizations')
    parser.add_argument('--no-visualize', action='store_true',
                       help='Skip visualizations (faster)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.image and not args.input_dir:
        parser.error("Must specify either --image or --input-dir")
    
    # Download SAM checkpoint if needed
    if args.sam_checkpoint is None:
        args.sam_checkpoint = download_sam_checkpoint(args.sam_model)
    
    # Determine visualization setting
    visualize = args.visualize or (not args.no_visualize and args.image is not None)
    
    # Initialize detector
    detector = LicensePlateCornerDetector(
        sam_checkpoint=args.sam_checkpoint,
        sam_model_type=args.sam_model,
        device=args.device,
        conf_threshold=args.conf_threshold
    )
    
    # Process
    if args.image:
        print(f"\nProcessing single image: {args.image}")
        results = detector.process_image(
            args.image,
            use_points=args.use_points,
            visualize=visualize,
            output_dir=args.vis_dir if visualize else None
        )
        
        if results:
            print(f"\nFound {len(results)} plate(s)")
            for idx, result in enumerate(results):
                print(f"\nPlate {idx}")
                print(f"  FastALPR confidence: {result['fastalpr_confidence']:.3f}")
                print(f"  Corners:")
                labels = ['Top-left', 'Top-right', 'Bottom-right', 'Bottom-left']
                for label, corner in zip(labels, result['corners']):
                    print(f"    {label:12s}: ({corner[0]:.1f}, {corner[1]:.1f})")
            
            if visualize:
                print(f"\nVisualization saved to: {args.vis_dir}")
        else:
            print("No plates detected!")
            
    elif args.input_dir:
        print(f"\nProcessing directory: {args.input_dir}")
        df = detector.process_directory(
            args.input_dir,
            output_csv=args.output,
            visualize=not args.no_visualize,
            vis_dir=args.vis_dir,
            use_points=args.use_points
        )
        
        if not df.empty:
            print(f"\nSummary:")
            print(f"  Images with plates: {df['image_path'].nunique()}")
            print(f"  Total plates: {len(df)}")
            print(f"  Avg confidence: {df['fastalpr_confidence'].mean():.3f}")
            
            if not args.no_visualize:
                print(f"\nVisualizations saved to: {args.vis_dir}")


if __name__ == "__main__":
    main()