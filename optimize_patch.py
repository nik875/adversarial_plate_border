#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import kornia
import kornia.geometry as K
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import onnx
import onnx2torch
from open_image_models import LicensePlateDetector
from typing import Tuple, List, Optional
import logging
import warnings
import argparse
warnings.filterwarnings("ignore")


class AdversarialPatchTrainer:
    def __init__(self,
                 csv_path: str,
                 patch_width: int = 64,
                 patch_height: int = 32,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):

        self.device = torch.device(device)
        self.patch_width = patch_width
        self.patch_height = patch_height

        # Load dataset
        self.df = pd.read_csv(csv_path)
        print(f"Loaded {len(self.df)} images")

        # Initialize adversarial patch with Xavier initialization
        self.patch = nn.Parameter(
            torch.randn(3, patch_height, patch_width, device=self.device) * 0.1)

        # Load YOLO model
        self.load_yolo_model()

        # Image preprocessing
        self.transform = T.Compose([T.ToTensor()])

        # Track statistics
        self.epoch_stats = []

    def load_yolo_model(self):
        """Load and convert YOLO model to PyTorch"""
        print("Loading YOLO model...")
        detector = LicensePlateDetector(detection_model="yolo-v9-t-384-license-plate-end2end")

        # Get ONNX model path
        model_cache_dir = Path.home() / ".cache/open-image-models/yolo-v9-t-384-license-plate-end2end"
        onnx_path = model_cache_dir / "yolo-v9-t-384-license-plates-end2end.onnx"

        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found at: {onnx_path}")

        onnx_model = onnx.load(str(onnx_path))
        self.model = onnx2torch.convert(onnx_model)
        self.model.to(self.device)
        self.model.eval()

        # Disable gradients for model parameters to save memory
        for param in self.model.parameters():
            param.requires_grad = False

    def load_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image"""
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert('RGB')
        image = image.resize((384, 384))
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        return image_tensor

    def extract_homography_matrix(self, row) -> torch.Tensor:
        """Extract homography matrix from CSV row"""
        H = np.array([
            [row['H00'], row['H01'], row['H02']],
            [row['H10'], row['H11'], row['H12']],
            [row['H20'], row['H21'], row['H22']]
        ], dtype=np.float32)

        if np.any(np.isnan(H)) or np.any(np.isinf(H)):
            raise ValueError("Invalid homography matrix")

        return torch.from_numpy(H).unsqueeze(0).to(self.device)

    def get_license_plate_corners(self, row) -> torch.Tensor:
        """Get license plate corner coordinates"""
        corners = torch.tensor([
            [row['p1_x'], row['p1_y']],
            [row['p2_x'], row['p2_y']],
            [row['p3_x'], row['p3_y']],
            [row['p4_x'], row['p4_y']]
        ], dtype=torch.float32, device=self.device).unsqueeze(0)

        if torch.any(torch.isnan(corners)) or torch.any(corners < 0) or torch.any(corners > 384):
            raise ValueError("Invalid corner coordinates")

        return corners

    def apply_patch_to_image(self, image: torch.Tensor, homography: torch.Tensor,
                             corners: torch.Tensor, border_scale: float = 1.4) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply adversarial patch as a border around the license plate using homography transformation"""
        batch_size = image.shape[0]

        try:
            # Normalize patch to [0, 1] range
            patch_normalized = torch.tanh(self.patch) * 0.5 + 0.5

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

            # Create and warp patch
            patch_batch = patch_normalized.unsqueeze(0).repeat(batch_size, 1, 1, 1)
            warped_patch = K.warp_perspective(
                patch_batch, M_border, dsize=(384, 384),
                mode='bilinear', padding_mode='zeros', align_corners=True
            )

            # Create masks
            patch_mask = torch.ones(batch_size, 1, self.patch_height, self.patch_width,
                                    dtype=torch.float32, device=self.device)

            warped_border_mask = K.warp_perspective(
                patch_mask, M_border, dsize=(384, 384),
                mode='bilinear', padding_mode='zeros', align_corners=True
            )

            warped_plate_mask = K.warp_perspective(
                patch_mask, M_plate, dsize=(384, 384),
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
            print(f"Warning: Patch application failed: {e}")
            return image, None

    def compute_detection_loss_full_image(self, model_output, ground_truth_box: torch.Tensor,
                                          target_confidence: float = 0.1) -> torch.Tensor:
        """Compute loss for full image detection"""
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        num_matched_detections = 0

        if isinstance(model_output, (list, tuple)):
            outputs = model_output
        else:
            outputs = [model_output]

        for output in outputs:
            if output is None or output.numel() == 0:
                continue

            # End-to-end YOLO format: [num_detections, 7]
            if output.dim() == 2 and output.shape[-1] >= 7:
                pred_boxes = output[:, 1:5]  # x1, y1, x2, y2
                confidences = output[:, 6]   # confidence

                if pred_boxes.shape[0] > 0:
                    ious = self.compute_box_iou(pred_boxes, ground_truth_box.unsqueeze(0))
                    ious = ious.squeeze(-1)

                    best_match_idx = ious.argmax()
                    best_iou = ious[best_match_idx]

                    # Only apply loss if IoU > threshold
                    if best_iou > 0.1:
                        matched_confidence = confidences[best_match_idx]
                        detection_loss = matched_confidence
                        total_loss = total_loss + detection_loss
                        num_matched_detections += 1

        # Return average loss or small default if no matches
        if num_matched_detections > 0:
            total_loss = total_loss / num_matched_detections
        else:
            total_loss = torch.tensor(0.001, device=self.device, requires_grad=True)

        return total_loss

    def evaluate_detection_performance_full_image(
            self, model_output, ground_truth_box: torch.Tensor) -> float:
        """Evaluate detection performance for full image"""
        max_matched_score = 0.0

        if isinstance(model_output, (list, tuple)):
            outputs = model_output
        else:
            outputs = [model_output]

        for output in outputs:
            if output is not None and output.dim() == 2 and output.shape[-1] >= 7:
                pred_boxes = output[:, 1:5]
                confidences = output[:, 6]

                if pred_boxes.shape[0] > 0:
                    ious = self.compute_box_iou(pred_boxes, ground_truth_box.unsqueeze(0))
                    ious = ious.squeeze(-1)

                    best_match_idx = ious.argmax()
                    best_iou = ious[best_match_idx]

                    if best_iou > 0.1:
                        matched_score = confidences[best_match_idx].item()
                        max_matched_score = max(max_matched_score, matched_score)

        return max_matched_score

    def compute_box_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Compute IoU between two sets of boxes"""
        area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        boxes1 = boxes1.unsqueeze(1)  # [N, 1, 4]
        boxes2 = boxes2.unsqueeze(0)  # [1, M, 4]

        # Intersection coordinates
        inter_x1 = torch.max(boxes1[..., 0], boxes2[..., 0])
        inter_y1 = torch.max(boxes1[..., 1], boxes2[..., 1])
        inter_x2 = torch.min(boxes1[..., 2], boxes2[..., 2])
        inter_y2 = torch.min(boxes1[..., 3], boxes2[..., 3])

        # Intersection area
        inter_width = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_height = torch.clamp(inter_y2 - inter_y1, min=0)
        inter_area = inter_width * inter_height

        # Union area and IoU
        union_area = area1.unsqueeze(1) + area2.unsqueeze(0) - inter_area
        iou = inter_area / (union_area + 1e-8)
        return iou

    def train_epoch(self, optimizer: torch.optim.Optimizer, epoch: int,
                    debug_first_image=False) -> Tuple[float, float, int]:
        """Train for one epoch"""
        total_loss = 0.0
        successful_images = 0
        detection_scores = []

        # Shuffle dataset
        shuffled_df = self.df.sample(frac=1, random_state=epoch).reset_index(drop=True)
        train_size = int(0.8 * len(shuffled_df))
        train_df = shuffled_df[:train_size]

        desc = f"Epoch {epoch+1} - Training"
        with tqdm(range(len(train_df)), desc=desc, leave=False) as pbar:
            for idx in pbar:
                row = train_df.iloc[idx]

                try:
                    # Load and preprocess image
                    image = self.load_image(row['processed_filename'])
                    homography = self.extract_homography_matrix(row)
                    corners = self.get_license_plate_corners(row)

                    # Apply adversarial patch to full image
                    patched_image, patch_mask = self.apply_patch_to_image(
                        image, homography, corners)
                    if patch_mask is None:
                        continue

                    # Create ground truth box from corners
                    x_coords = corners[0, :, 0]
                    y_coords = corners[0, :, 1]
                    ground_truth_box = torch.tensor([
                        x_coords.min(), y_coords.min(), x_coords.max(), y_coords.max()
                    ], device=self.device)

                    # Run YOLO detection on full patched image
                    detection_output = self.model(patched_image)

                    # DEBUG PRINTS FOR FIRST IMAGE
                    if debug_first_image and idx == 0:
                        print(f"\n{'='*80}")
                        print(f"DEBUG: FIRST IMAGE OF EPOCH {epoch+1}")
                        print(f"{'='*80}")
                        print(f"Image: {row['processed_filename']}")
                        print(f"Ground truth box: {ground_truth_box}")
                        print(f"Processing full 384x384 image (no ROI extraction)")

                        # Print raw model output structure
                        print(
                            f"\nModel output type: {type(detection_output)} (length: {len(detection_output)})")
                        for i, output in enumerate(detection_output):
                            print(f"  Output[{i}] shape: {output.shape}")
                            print(f"  Output[{i}] dtype: {output.dtype}")
                            print(f"  Output[{i}] device: {output.device}")

                            # Print full tensor values
                            print(f"  Output[{i}] full tensor:")
                            print(f"    {output}")

                            # Parse and explain the values
                            print(
                                f"\n  PARSING Output[{i}] (format: [num_detections, 7]):")
                            print(f"    Tensor columns explanation:")
                            print(
                                f"      Column 0: ? (value {output[0]})")
                            print(f"      Columns 1-4: Bounding box [x1, y1, x2, y2]")
                            print(f"      Column 5: Class ID")
                            print(f"      Column 6: Confidence score")

                            print(f"\n    Detection {i}:")
                            print(f"      Raw values: {output}")
                            print(
                                f"      Bounding box: [{output[1]:.2f}, {output[2]:.2f}, {output[3]:.2f}, {output[4]:.2f}]")
                            print(f"      Class ID: {output[5]:.1f}")
                            print(f"      Confidence: {output[6]:.6f}")

                            # Calculate IoU with ground truth
                            pred_box = output[1:5]
                            iou = self.compute_box_iou(
                                pred_box.unsqueeze(0), ground_truth_box.unsqueeze(0))
                            print(
                                f"      IoU with ground truth: {iou.item():.6f}")
                            print(
                                f"      Will be used in loss? {'YES' if iou.item() > 0.1 else 'NO'} (threshold: 0.1)")

                        print(f"{'='*80}")
                        print(f"END DEBUG FOR FIRST IMAGE")
                        print(f"{'='*80}\n")

                    # Compute loss and update statistics
                    loss = self.compute_detection_loss_full_image(
                        detection_output, ground_truth_box)
                    loss.backward()

                    total_loss += loss.item()
                    successful_images += 1

                    detection_score = self.evaluate_detection_performance_full_image(
                        detection_output, ground_truth_box)
                    detection_scores.append(detection_score)

                    # Update progress bar
                    if successful_images > 0:
                        avg_loss = total_loss / successful_images
                        avg_detection = np.mean(detection_scores)

                        pbar.set_postfix({
                            'Loss': f'{avg_loss:.4f}',
                            'Det': f'{avg_detection:.3f}',
                            'Success': f'{successful_images}/{idx+1}',
                            'Patch_Std': f'{self.patch.std().item():.3f}'
                        })

                except Exception as e:
                    # Fail loudly with full error details
                    print(f"\nFATAL ERROR processing {row.get('processed_filename', 'unknown')}:")
                    print(f"Error type: {type(e).__name__}")
                    print(f"Error message: {str(e)}")
                    print(f"Row data: {row.to_dict()}")
                    raise e

        # Apply accumulated gradients
        if successful_images > 0:
            torch.nn.utils.clip_grad_norm_([self.patch], max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = total_loss / successful_images if successful_images > 0 else float('inf')
        avg_detection_score = np.mean(detection_scores) if detection_scores else 1.0

        return avg_loss, avg_detection_score, successful_images

    def validate(self, epoch: int) -> Tuple[float, int]:
        """Validation pass on held-out data"""
        detection_scores = []
        successful_validations = 0

        shuffled_df = self.df.sample(frac=1, random_state=epoch).reset_index(drop=True)
        train_size = int(0.8 * len(shuffled_df))
        val_df = shuffled_df[train_size:]

        with torch.no_grad():
            for idx in range(min(30, len(val_df))):
                row = val_df.iloc[idx]

                try:
                    image = self.load_image(row['processed_filename'])
                    homography = self.extract_homography_matrix(row)
                    corners = self.get_license_plate_corners(row)

                    # Apply adversarial patch to full image
                    patched_image, patch_mask = self.apply_patch_to_image(
                        image, homography, corners)
                    if patch_mask is None:
                        continue

                    # Create ground truth box from corners
                    x_coords = corners[0, :, 0]
                    y_coords = corners[0, :, 1]
                    ground_truth_box = torch.tensor([
                        x_coords.min(), y_coords.min(), x_coords.max(), y_coords.max()
                    ], device=self.device)

                    # Run YOLO detection on full patched image
                    detection_output = self.model(patched_image)
                    detection_score = self.evaluate_detection_performance_full_image(
                        detection_output, ground_truth_box)
                    detection_scores.append(detection_score)
                    successful_validations += 1

                except Exception:
                    continue

        avg_val_score = np.mean(detection_scores) if detection_scores else 1.0
        return avg_val_score, successful_validations

    def save_patch(self, epoch: int, save_dir: str = "patches"):
        """Save current patch state"""
        Path(save_dir).mkdir(exist_ok=True)

        with torch.no_grad():
            patch_img = torch.tanh(self.patch) * 0.5 + 0.5
            patch_img = patch_img.detach().cpu()
            patch_pil = T.ToPILImage()(patch_img)

            patch_pil.save(f"{save_dir}/patch_epoch_{epoch:04d}.png")

            torch.save({
                'patch': self.patch.detach().cpu(),
                'epoch': epoch,
                'patch_size': (self.patch_height, self.patch_width)
            }, f"{save_dir}/patch_epoch_{epoch:04d}.pt")

    def train(self, num_epochs: int = 100, learning_rate: float = 0.01,
              save_interval: int = 10, early_stop_patience: int = 15):
        """Main training loop"""

        optimizer = optim.AdamW([self.patch], lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )

        history = {
            'loss': [], 'detection_score': [], 'val_score': [],
            'learning_rate': [], 'successful_images': []
        }

        best_loss = float('inf')
        patience_counter = 0

        print(f"\nStarting adversarial patch training")
        print(f"   Dataset: {len(self.df)} images")
        print(f"   Patch size: {self.patch_height}×{self.patch_width}")
        print(f"   Device: {self.device}")
        print(f"   Epochs: {num_epochs}")
        print(f"   Initial LR: {learning_rate}")
        print(f"   Processing: Full 384x384 images only")
        print("-" * 60)

        for epoch in range(num_epochs):
            # Training and validation
            train_loss, train_detection_score, successful_imgs = self.train_epoch(optimizer, epoch)
            val_detection_score, val_imgs = self.validate(epoch)

            # Learning rate scheduling
            scheduler.step(train_loss)
            current_lr = optimizer.param_groups[0]['lr']

            # Record history
            history['loss'].append(train_loss)
            history['detection_score'].append(train_detection_score)
            history['val_score'].append(val_detection_score)
            history['learning_rate'].append(current_lr)
            history['successful_images'].append(successful_imgs)

            # Calculate detection reduction
            initial_detection = history['val_score'][0] if len(history['val_score']) > 0 else 1.0
            detection_reduction = (1 - val_detection_score / initial_detection) * 100

            # Print epoch summary
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Loss: {train_loss:.4f} | "
                  f"Train Det: {train_detection_score:.3f} | "
                  f"Val Det: {val_detection_score:.3f} | "
                  f"Reduction: {detection_reduction:+.1f}% | "
                  f"LR: {current_lr:.2e} | "
                  f"Success: {successful_imgs}/{int(0.8*len(self.df))}")

            # Save best model
            if val_detection_score < best_loss:
                best_loss = val_detection_score
                patience_counter = 0
                self.save_patch(epoch, "best_patches")
                print(f"   New best loss: {best_loss:.4f}")
            else:
                patience_counter += 1

            # Periodic saves
            if (epoch + 1) % save_interval == 0:
                self.save_patch(epoch, "checkpoint_patches")

            # Early stopping
            if patience_counter >= early_stop_patience:
                print(f"   Early stopping: No improvement for {early_stop_patience} epochs")
                break

            # Check for convergence
            if len(history['loss']) >= 20:
                recent_losses = history['loss'][-20:]
                if (max(recent_losses) - min(recent_losses)) < 0.0001:
                    print("   Converged: Loss stabilized")
                    break

        print(f"\nTraining completed!")
        print(f"   Best loss: {best_loss:.4f}")
        final_reduction = (1 - history['val_score'][-1] / history['val_score'][0]) * 100
        print(f"   Detection reduction: {final_reduction:.1f}%")

        return history


def test_detection_visualization(csv_path: str, output_path: str = "test_detection.png"):
    """Test mode: visualize YOLO detections on a single image without patch"""
    print("Running test mode - visualizing detections on sample image...")

    df = pd.read_csv(csv_path)
    if len(df) == 0:
        raise ValueError("No images found in dataset")

    row = df.iloc[0]
    print(f"Testing on: {row['processed_filename']}")

    trainer = AdversarialPatchTrainer(csv_path, patch_width=64, patch_height=32)

    # Load and process the test image
    image = trainer.load_image(row['processed_filename'])
    corners = trainer.get_license_plate_corners(row)

    # Run YOLO detection on full image
    with torch.no_grad():
        detection_output = trainer.model(image)

    # Create ground truth box
    x_coords = corners[0, :, 0]
    y_coords = corners[0, :, 1]
    ground_truth_box = torch.tensor([
        x_coords.min(), y_coords.min(), x_coords.max(), y_coords.max()
    ], device=trainer.device)

    print(f"\nDetection Results:")
    print(f"Ground truth box: {ground_truth_box}")

    # Parse YOLO output
    if isinstance(detection_output, (list, tuple)):
        outputs = detection_output
    else:
        outputs = [detection_output]

    all_detections = []
    for i, output in enumerate(outputs):
        if output is not None and output.dim() == 2 and output.shape[0] > 0:
            pred_boxes = output[:, 1:5]
            confidences = output[:, 6]
            class_ids = output[:, 5]

            ious = trainer.compute_box_iou(pred_boxes, ground_truth_box.unsqueeze(0)).squeeze(-1)

            print(f"\nOutput {i}:")
            for j in range(len(pred_boxes)):
                print(f"  Detection {j}: Box={pred_boxes[j]}, Conf={confidences[j]:.4f}, "
                      f"Class={class_ids[j]:.0f}, IoU={ious[j]:.4f}")

                all_detections.append({
                    'box': pred_boxes[j].cpu().numpy(),
                    'confidence': confidences[j].item(),
                    'class_id': int(class_ids[j].item()),
                    'iou': ious[j].item()
                })

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Original image with ground truth
    original_img = image[0].permute(1, 2, 0).cpu().numpy()
    ax1.imshow(original_img)
    ax1.set_title('Full Image with Ground Truth & Detections')

    # Draw ground truth box
    gt_rect = patches.Rectangle(
        (ground_truth_box[0], ground_truth_box[1]),
        ground_truth_box[2] - ground_truth_box[0],
        ground_truth_box[3] - ground_truth_box[1],
        linewidth=3, edgecolor='green', facecolor='none', label='Ground Truth'
    )
    ax1.add_patch(gt_rect)

    # Draw all detections
    colors = ['red', 'blue', 'orange', 'purple', 'brown']
    for i, det in enumerate(all_detections):
        box = det['box']
        color = colors[i % len(colors)]

        rect = patches.Rectangle(
            (box[0], box[1]), box[2] - box[0], box[3] - box[1],
            linewidth=2, edgecolor=color, facecolor='none',
            label=f'Det {i}: {det["confidence"]:.3f}'
        )
        ax1.add_patch(rect)

        ax1.text(box[0], box[1] - 5, f'{det["confidence"]:.3f}',
                 color=color, fontsize=10, weight='bold')

    ax1.legend()

    # Analysis panel
    ax2.axis('off')
    ax2.set_title('Detection Analysis', pad=20)

    analysis_text = f"Dataset: {len(df)} images\n"
    analysis_text += f"Test image: {Path(row['processed_filename']).name}\n"
    analysis_text += f"Image size: 384x384 (full image processing)\n\n"

    analysis_text += f"Ground Truth Box:\n"
    analysis_text += f"  [{ground_truth_box[0]:.1f}, {ground_truth_box[1]:.1f}, "
    analysis_text += f"{ground_truth_box[2]:.1f}, {ground_truth_box[3]:.1f}]\n\n"

    analysis_text += f"YOLO Detections: {len(all_detections)}\n"
    for i, det in enumerate(all_detections):
        analysis_text += f"  {i}: Conf={det['confidence']:.4f}, IoU={det['iou']:.4f}\n"

    if all_detections:
        best_det = max(all_detections, key=lambda x: x['iou'])
        analysis_text += f"\nBest Match:\n"
        analysis_text += f"  Confidence: {best_det['confidence']:.4f}\n"
        analysis_text += f"  IoU: {best_det['iou']:.4f}\n"
        analysis_text += f"  Class: {best_det['class_id']}\n"

        if best_det['iou'] > 0.1:
            analysis_text += f"  Status: MATCHED (IoU > 0.1)\n"
        else:
            analysis_text += f"  Status: NO MATCH (IoU ≤ 0.1)\n"
    else:
        analysis_text += f"\nNo detections found!\n"

    ax2.text(0.05, 0.95, analysis_text, transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nVisualization saved to: {output_path}")
    print(f"Total detections: {len(all_detections)}")
    if all_detections:
        best_conf = max(det['confidence'] for det in all_detections)
        best_iou = max(det['iou'] for det in all_detections)
        print(f"Best confidence: {best_conf:.4f}")
        print(f"Best IoU: {best_iou:.4f}")


def debug_patch_application(csv_path: str, output_path: str = "debug_patch.png"):
    """Debug mode: Apply a blank white patch and visualize the result"""
    print("Running debug patch mode - applying blank white patch...")

    df = pd.read_csv(csv_path)
    if len(df) == 0:
        raise ValueError("No images found in dataset")

    row = df.iloc[0]
    print(f"Testing on: {row['processed_filename']}")

    trainer = AdversarialPatchTrainer(csv_path, patch_width=64, patch_height=32)

    # Create blank white patch
    white_patch = torch.ones(3, trainer.patch_height, trainer.patch_width, device=trainer.device)
    original_patch = trainer.patch.clone()
    trainer.patch.data = white_patch

    # Load and process the test image
    image = trainer.load_image(row['processed_filename'])
    corners = trainer.get_license_plate_corners(row)
    homography = trainer.extract_homography_matrix(row)

    # Apply white patch to full image
    patched_image, patch_mask = trainer.apply_patch_to_image(image, homography, corners)

    # Create ground truth box
    x_coords = corners[0, :, 0]
    y_coords = corners[0, :, 1]
    ground_truth_box = torch.tensor([
        x_coords.min(), y_coords.min(), x_coords.max(), y_coords.max()
    ], device=trainer.device)

    # Run YOLO on both original and patched full images
    with torch.no_grad():
        original_output = trainer.model(image)
        patched_output = trainer.model(patched_image)

    # Parse outputs
    def parse_yolo_output(output, gt_box):
        if isinstance(output, (list, tuple)):
            outputs = output
        else:
            outputs = [output]

        all_dets = []
        for out in outputs:
            if out is not None and out.dim() == 2 and out.shape[0] > 0:
                pred_boxes = out[:, 1:5]
                confidences = out[:, 6]
                class_ids = out[:, 5]
                ious = trainer.compute_box_iou(pred_boxes, gt_box.unsqueeze(0)).squeeze(-1)

                for j in range(len(pred_boxes)):
                    all_dets.append({
                        'box': pred_boxes[j].cpu().numpy(),
                        'confidence': confidences[j].item(),
                        'class_id': int(class_ids[j].item()),
                        'iou': ious[j].item()
                    })
        return all_dets

    original_dets = parse_yolo_output(original_output, ground_truth_box)
    patched_dets = parse_yolo_output(patched_output, ground_truth_box)

    # Find best matches
    best_original = max(original_dets, key=lambda x: x['iou']) if original_dets else None
    best_patched = max(patched_dets, key=lambda x: x['iou']) if patched_dets else None

    print(f"\nOriginal full image:")
    print(
        f"  Best detection: Conf={best_original['confidence']:.4f}, IoU={best_original['iou']:.4f}" if best_original else "  No detections")

    print(f"\nWith white patch on full image:")
    print(
        f"  Best detection: Conf={best_patched['confidence']:.4f}, IoU={best_patched['iou']:.4f}" if best_patched else "  No detections")

    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Original image
    orig_img = image[0].permute(1, 2, 0).cpu().numpy()
    ax1.imshow(orig_img)
    ax1.set_title('Original Full Image')

    # Draw ground truth
    gt_rect = patches.Rectangle(
        (ground_truth_box[0], ground_truth_box[1]),
        ground_truth_box[2] - ground_truth_box[0],
        ground_truth_box[3] - ground_truth_box[1],
        linewidth=3, edgecolor='green', facecolor='none', label='Ground Truth'
    )
    ax1.add_patch(gt_rect)

    # Draw best original detection
    if best_original and best_original['iou'] > 0.1:
        box = best_original['box']
        orig_rect = patches.Rectangle(
            (box[0], box[1]), box[2] - box[0], box[3] - box[1],
            linewidth=2, edgecolor='blue', facecolor='none',
            label=f'YOLO: {best_original["confidence"]:.3f}'
        )
        ax1.add_patch(orig_rect)
    ax1.legend()

    # Patched image
    patched_img = patched_image[0].permute(1, 2, 0).detach().cpu().numpy()
    ax2.imshow(patched_img)
    ax2.set_title('Full Image With White Patch')

    # Draw ground truth
    gt_rect2 = patches.Rectangle(
        (ground_truth_box[0], ground_truth_box[1]),
        ground_truth_box[2] - ground_truth_box[0],
        ground_truth_box[3] - ground_truth_box[1],
        linewidth=3, edgecolor='green', facecolor='none', label='Ground Truth'
    )
    ax2.add_patch(gt_rect2)

    # Draw best patched detection
    if best_patched and best_patched['iou'] > 0.1:
        box = best_patched['box']
        patch_rect = patches.Rectangle(
            (box[0], box[1]), box[2] - box[0], box[3] - box[1],
            linewidth=2, edgecolor='red', facecolor='none',
            label=f'YOLO: {best_patched["confidence"]:.3f}'
        )
        ax2.add_patch(patch_rect)
    ax2.legend()

    # Show patch mask
    if patch_mask is not None:
        mask_img = patch_mask[0, 0].cpu().numpy()
        ax3.imshow(mask_img, cmap='gray')
        ax3.set_title('Patch Mask (White = Patch Area)')
    else:
        ax3.text(
            0.5,
            0.5,
            'Patch application failed',
            ha='center',
            va='center',
            transform=ax3.transAxes)
        ax3.set_title('Patch Mask - FAILED')

    # Analysis
    ax4.axis('off')
    ax4.set_title('Debug Analysis', pad=20)

    analysis_text = f"White Patch Debug Results\n\n"
    analysis_text += f"Image: {Path(row['processed_filename']).name}\n"
    analysis_text += f"Processing: Full 384x384 images\n"
    analysis_text += f"Ground truth: [{ground_truth_box[0]:.1f}, {ground_truth_box[1]:.1f}, "
    analysis_text += f"{ground_truth_box[2]:.1f}, {ground_truth_box[3]:.1f}]\n\n"

    analysis_text += f"ORIGINAL FULL IMAGE:\n"
    analysis_text += f"  Detections: {len(original_dets)}\n"
    if best_original:
        analysis_text += f"  Best: Conf={best_original['confidence']:.4f}, IoU={best_original['iou']:.4f}\n"
    else:
        analysis_text += f"  Best: None found\n"

    analysis_text += f"\nWITH WHITE PATCH ON FULL IMAGE:\n"
    analysis_text += f"  Detections: {len(patched_dets)}\n"
    if best_patched:
        analysis_text += f"  Best: Conf={best_patched['confidence']:.4f}, IoU={best_patched['iou']:.4f}\n"
    else:
        analysis_text += f"  Best: None found\n"

    if best_original and best_patched:
        conf_change = best_patched['confidence'] - best_original['confidence']
        analysis_text += f"\nCONFIDENCE CHANGE: {conf_change:+.4f}\n"
        if abs(conf_change) < 0.01:
            analysis_text += f"  Status: MINIMAL IMPACT\n"
        elif conf_change < -0.05:
            analysis_text += f"  Status: PATCH WORKING\n"
        else:
            analysis_text += f"  Status: UNEXPECTED CHANGE\n"

    analysis_text += f"\nPatch application: {'SUCCESS' if patch_mask is not None else 'FAILED'}\n"

    ax4.text(0.05, 0.95, analysis_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Restore original patch
    trainer.patch.data = original_patch

    print(f"\nDebug visualization saved to: {output_path}")
    if best_original and best_patched:
        print(f"Confidence change: {best_original['confidence']:.4f} -> {best_patched['confidence']:.4f} "
              f"({best_patched['confidence'] - best_original['confidence']:+.4f})")


def main():
    parser = argparse.ArgumentParser(description='Adversarial Patch Training')
    parser.add_argument('--test', action='store_true',
                        help='Test mode: visualize detections on single image without patch')
    parser.add_argument('--debug-patch', action='store_true',
                        help='Debug mode: apply blank white patch and visualize impact')
    parser.add_argument('--output', default='test_detection.png',
                        help='Output path for test/debug visualization')
    args = parser.parse_args()

    # Configuration
    CSV_PATH = "preproc_labels.csv"
    PATCH_WIDTH = 64
    PATCH_HEIGHT = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.1

    if args.test:
        try:
            test_detection_visualization(CSV_PATH, args.output)
        except Exception as e:
            print(f"Test failed: {e}")
            raise
        return

    if args.debug_patch:
        try:
            debug_patch_application(CSV_PATH, args.output)
        except Exception as e:
            print(f"Debug patch failed: {e}")
            raise
        return

    # Normal training mode
    try:
        trainer = AdversarialPatchTrainer(
            csv_path=CSV_PATH,
            patch_width=PATCH_WIDTH,
            patch_height=PATCH_HEIGHT
        )

        history = trainer.train(
            num_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            save_interval=10,
            early_stop_patience=15
        )

        # Plot training results
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Loss curve
        ax1.plot(history['loss'], 'b-', label='Training Loss')
        ax1.set_title('Training Loss Over Time')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Detection scores
        ax2.plot(history['detection_score'], 'r-', label='Train Detection', alpha=0.7)
        ax2.plot(history['val_score'], 'g-', label='Val Detection', linewidth=2)
        ax2.set_title('Detection Scores (Lower = Better Attack)')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Detection Score')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Learning rate
        ax3.semilogy(history['learning_rate'], 'purple', label='Learning Rate')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate (log scale)')
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        # Final adversarial patch
        final_patch = torch.tanh(trainer.patch) * 0.5 + 0.5
        final_patch_np = final_patch.detach().cpu().permute(1, 2, 0).numpy()
        ax4.imshow(final_patch_np)
        ax4.set_title(f'Final Adversarial Patch ({PATCH_WIDTH}×{PATCH_HEIGHT})')
        ax4.axis('off')

        plt.tight_layout()
        plt.savefig('adversarial_training_results.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("\nResults saved to 'adversarial_training_results.png'")
        print("Patch checkpoints saved in 'patches/' directory")

    except Exception as e:
        print(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
