#!/usr/bin/env python3
import os
from typing import Tuple, List, Optional
import logging
import warnings
import argparse
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
import numpy as np
from tqdm import tqdm
import kornia
import kornia.geometry as K
import torchvision.transforms as T
import matplotlib.pyplot as plt
from matplotlib import patches
import onnx
import onnx2torch
from open_image_models import LicensePlateDetector
from dataset import create_dataloaders
warnings.filterwarnings("ignore")


PATCH_WIDTH = 256
PATCH_HEIGHT = 128


class AdversarialPatchTrainer:
    def __init__(self,
                 csv_path: str,
                 device: str = None,
                 grad_accumulate: int = None,
                 match_detection: bool = False,
                 impersonation_target: str = None):

        # Image preprocessing
        self.transform = T.Compose([T.ToTensor()])

        self.grad_accumulate = grad_accumulate
        self.match_detection = match_detection
        self.impersonation_target = impersonation_target

        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device
        self.patch_width = PATCH_WIDTH
        self.patch_height = PATCH_HEIGHT

        self.train_loader, self.val_loader = create_dataloaders(csv_path, transform=self.transform,
                                                                preload=False, batch_size=1,
                                                                n_jobs=1)

        # Initialize adversarial patch with Xavier initialization
        self.patch = nn.Parameter(
            torch.randn(3, self.patch_height, self.patch_width, device=self.device) * 0.1)

        # Load YOLO model
        self.load_yolo_model()

        # Track statistics
        self.epoch_stats = []

    def load_yolo_model(self):
        """Load and convert YOLO model to PyTorch"""
        print("Loading YOLO model...")
        LicensePlateDetector(detection_model="yolo-v9-t-384-license-plate-end2end")

        # Get ONNX model path
        model_cache_dir = \
            Path.home() / ".cache/open-image-models/yolo-v9-t-384-license-plate-end2end"
        onnx_path = model_cache_dir / "yolo-v9-t-384-license-plates-end2end.onnx"
        ocr_path = \
            Path.home() / ".cache/fast-plate-ocr/cct-xs-v1-global-model/cct_xs_v1_global.onnx"

        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found at: {onnx_path}")

        onnx_model = onnx.load(str(onnx_path))
        self.model = onnx2torch.convert(onnx_model)
        self.model.to(self.device)
        self.model.eval()

        ocr_model = onnx.load(str(ocr_path))
        self.ocr_input_shape = (64, 128, 3)
        alphabet = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_'

        # Set OCR target based on mode
        if self.impersonation_target:
            # For impersonation: target is the text we want OCR to read
            self.ocr_target = self.text_to_target_tensor(self.impersonation_target, 9, alphabet)
        else:
            # Original behavior: target is the text we want to prevent OCR from reading
            self.ocr_target = self.text_to_target_tensor('VRJ7774', 9, alphabet)

        self.ocr = onnx2torch.convert(ocr_model).to(self.device)
        self.ocr_loss = self.focal_cce_loss(len(alphabet))
        self.detection_baseline, self.ocr_baseline = self.calculate_baseline_loss()
        self.ocr.eval()

        # Disable gradients for model parameters to save memory
        for param in self.model.parameters():
            param.requires_grad = False

    def text_to_target_tensor(self, plate_text: str, max_slots: int, alphabet: str):
        """Convert 'ABC123' -> one-hot tensor [batch, seq_len, vocab_size]"""
        # Pad with '_' to max_slots
        padded = (plate_text + '_' * max_slots)[:max_slots]

        # Convert to indices
        indices = [alphabet.index(char) for char in padded]

        # One-hot encode
        target = torch.zeros(1, max_slots, len(alphabet))
        for i, idx in enumerate(indices):
            target[0, i, idx] = 1.0

        return target.to(self.device)

    def get_patch_bounding_box(self, corners: torch.Tensor,
                               border_scale: float = 1.4) -> torch.Tensor:
        """Calculate bounding box of the patch area (border around license plate)"""
        # corners is [4, 2] when called from partial_loss (no batch dim)
        plate_corners = corners  # [4, 2]

        # Calculate center and create larger border quad (same logic as in apply_patch_to_image)
        center_x = plate_corners[:, 0].mean()
        center_y = plate_corners[:, 1].mean()
        center = torch.tensor([center_x, center_y], device=self.device)

        border_corners = center.unsqueeze(0) + (plate_corners - center.unsqueeze(0)) * border_scale

        # Calculate bounding box of the border corners
        min_x = torch.min(border_corners[:, 0])
        max_x = torch.max(border_corners[:, 0])
        min_y = torch.min(border_corners[:, 1])
        max_y = torch.max(border_corners[:, 1])

        return torch.stack([min_x, min_y, max_x, max_y])

    def apply_patch_to_image(self, image: torch.Tensor,
                             corners: torch.Tensor,
                             border_scale: float = 1.4) \
            -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply adversarial patch as border around license plate using homography"""
        batch_size = image.shape[0]

        # Extract image dimensions dynamically
        image_height, image_width = image.shape[2], image.shape[3]
        # FIXED: kornia produces output with dimensions swapped, so we need to swap them back
        # NOTE: This produces output matching input spatial dims
        dsize = (image_height, image_width)

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

        # Create and warp patch using dynamic image size
        patch_batch = patch_normalized.unsqueeze(0).repeat(batch_size, 1, 1, 1)

        warped_patch = K.warp_perspective(
            patch_batch, M_border, dsize=dsize,  # Dynamic size
            mode='bilinear', padding_mode='zeros', align_corners=True
        )

        # Create masks using dynamic image size
        patch_mask = torch.ones(batch_size, 1, self.patch_height, self.patch_width,
                                dtype=torch.float32, device=self.device)

        warped_border_mask = K.warp_perspective(
            patch_mask, M_border, dsize=dsize,  # Dynamic size
            mode='bilinear', padding_mode='zeros', align_corners=True
        )

        warped_plate_mask = K.warp_perspective(
            patch_mask, M_plate, dsize=dsize,  # Dynamic size
            mode='bilinear', padding_mode='zeros', align_corners=True
        )

        # Final mask: border area minus license plate area
        final_mask = torch.clamp(warped_border_mask - warped_plate_mask, 0, 1)
        final_mask = final_mask.expand(-1, 3, -1, -1)

        # Validate all tensors have matching spatial dimensions
        if image.shape[2:] != final_mask.shape[2:]:
            raise RuntimeError(
                f"Image spatial dims {image.shape[2:]} != mask spatial dims {final_mask.shape[2:]}")
        if image.shape[2:] != warped_patch.shape[2:]:
            raise RuntimeError(
                f"Image spatial dims {image.shape[2:]} != warped patch spatial dims {warped_patch.shape[2:]}")

        # Apply patch with cutout
        result_image = image * (1 - final_mask) + warped_patch * final_mask
        result_image = torch.clamp(result_image, 0, 1)

        return result_image, final_mask

    def focal_cce_loss(
        self,
        vocabulary_size: int,
        alpha: float = 0.25,
        gamma: float = 2.0,
        label_smoothing: float = 0.01,
    ):
        """
        Categorical focal cross-entropy loss - exact PyTorch replica of Keras version.
        """
        def cce(y_true, y_pred):
            """
            Computes the focal categorical cross-entropy loss.

            Args:
                y_true: One-hot encoded ground truth [batch, seq_len, vocab_size]
                y_pred: Model predictions (logits or probabilities) [batch, seq_len, vocab_size]
            """
            # Exact replica: flatten both inputs to (-1, vocabulary_size)
            y_true = y_true.reshape(-1, vocabulary_size)
            y_pred = y_pred.reshape(-1, vocabulary_size)

            # Ensure y_pred are probabilities (Keras uses from_logits=False)
            if y_pred.max() > 1.0 or y_pred.min() < 0.0:
                y_pred = F.softmax(y_pred, dim=-1)

            # Apply label smoothing to y_true (if specified)
            if label_smoothing > 0.0:
                y_true = y_true * (1.0 - label_smoothing) + label_smoothing / vocabulary_size

            # Compute focal loss
            # p_t is the probability of the true class
            p_t = torch.sum(y_true * y_pred, dim=-1)  # [batch*seq]

            # Focal weight: (1 - p_t)^gamma
            focal_weight = (1.0 - p_t) ** gamma

            # Cross entropy: -log(p_t)
            ce_loss = -torch.log(p_t + 1e-8)  # Small epsilon to prevent log(0)

            # Combine: focal_loss = alpha * focal_weight * ce_loss
            focal_loss = alpha * focal_weight * ce_loss

            # Return mean (exact replica behavior)
            return torch.mean(focal_loss)

        return cce

    def invert_bbox(self, corners, transform):
        """Invert the given transformation to bring the corners back to original image"""
        r, dw, dh = transform
        # Undo operations in REVERSE order
        corners = corners.clone()  # Make a copy to avoid view issues
        corners[::2] = corners[::2] - dw
        corners[1::2] = corners[1::2] - dh
        corners = corners / r
        return corners

    def bbox_to_corners(self, bbox, device=None):
        x1, y1, x2, y2 = bbox
        corners = torch.tensor([[
            [x1, y1],  # top-left
            [x2, y1],  # top-right
            [x2, y2],  # bottom-right
            [x1, y2]   # bottom-left
        ]], device=device or self.device)  # [1, 4, 2]
        return corners

    def corners_to_bbox(self, corners):
        min_x = torch.min(corners[:, 0])
        max_x = torch.max(corners[:, 0])
        min_y = torch.min(corners[:, 1])
        max_y = torch.max(corners[:, 1])
        return torch.stack([min_x, min_y, max_x, max_y])

    def boxes_IoU(self, boxes1, boxes2):
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

        # Union area = area1 + area2 - intersection_area
        union_area = area1 + area2 - inter_area

        # Intersection area / ground truth area
        return inter_area / (union_area + 1e-8)

    def partial_loss(self, batch, use_ocr_baseline=True):
        # Load original image (no patch)
        prep_image = batch['prep_image'].to(self.device)
        corners = batch['new_corners'].to(self.device)

        if self.match_detection:
            # Use patch bounding box as target instead of ground truth
            target_box = self.get_patch_bounding_box(corners)
        else:
            # Use ground truth license plate box as target
            target_box = self.corners_to_bbox(corners)

        # Run YOLO detection on full patched image
        model_output = self.model(prep_image.unsqueeze(0))

        best_detection = None
        det_loss = 0.0

        for detection in model_output:
            pred_box = detection[1:5]
            conf = detection[6]
            IoU = self.boxes_IoU(pred_box.unsqueeze(0), target_box.unsqueeze(0))

            if self.match_detection:
                # For match_detection, we want to MAXIMIZE IoU with patch area
                # So we use negative IoU as loss (minimizing negative = maximizing positive)
                this_det_loss = -IoU * conf
            else:
                # Original behavior: minimize IoU with ground truth
                this_det_loss = IoU * conf

            if self.match_detection:
                # For match_detection, we want the detection with highest IoU*conf
                if -this_det_loss > -det_loss:  # Higher IoU*conf is better
                    det_loss = this_det_loss
                    best_detection = detection
            else:
                # Original behavior: detection with highest IoU*conf
                if this_det_loss > det_loss:
                    det_loss = this_det_loss
                    best_detection = detection

        ocr_loss = 0.0
        if best_detection is not None:
            pred_box = best_detection[1:5]
            orig_projection = self.invert_bbox(pred_box.to('cpu'), batch['transform'])
            corners_box = self.bbox_to_corners(orig_projection, device='cpu')

            # Use crop_and_resize instead
            cropped_plate = kornia.geometry.crop_and_resize(
                batch['orig_image'].unsqueeze(0),                           # [1, C, H, W]
                corners_box,                     # [1, 4, 2] - corner coordinates
                self.ocr_input_shape[:2],       # (H, W) tuple
                mode='bilinear',
                align_corners=True
            ).to(self.device)  # Only move small image to gpu

            ocr_input = cropped_plate.permute(0, 2, 3, 1) * 255  # NHWC
            ocr_output = self.ocr(ocr_input)
            ocr_loss = self.ocr_loss(self.ocr_target, ocr_output)

            if use_ocr_baseline:
                if not hasattr(self, 'ocr_baseline'):
                    raise ValueError('Must call calculate_baseline_loss before using!')

                if self.impersonation_target:
                    # For impersonation: minimize loss (reward good OCR performance)
                    # Use normal scaling: lower loss = better
                    ocr_loss = ocr_loss / self.ocr_baseline
                else:
                    # Original behavior: maximize loss (reward bad OCR performance)
                    # Use inverted scaling: higher loss = better (lower total loss)
                    ocr_loss = self.ocr_baseline / ocr_loss

        return det_loss, ocr_loss

    def calculate_baseline_loss(self) -> float:
        """
        Calculate baseline OCR loss across entire dataset using ground truth boxes.

        Returns:
            Average OCR loss across all images and plates
        """
        total_ocr_loss = 0.0
        total_det_loss = 0.0
        total_plates = 0

        desc = "Calculating baseline OCR loss"
        with tqdm(self.train_loader, desc=desc, leave=False) as pbar:
            with torch.no_grad():
                for batch in pbar:
                    batch = {k: v[0] for k, v in batch.items()}  # Remove batch dim
                    det_loss, ocr_loss = self.partial_loss(batch, use_ocr_baseline=False)
                    total_det_loss += det_loss
                    total_ocr_loss += ocr_loss
                    total_plates += 1

                    # Update progress bar
                    avg_ocr = total_ocr_loss / total_plates
                    avg_det = total_det_loss / total_plates
                    pbar.set_postfix({
                        'Avg_Detection_Loss': f'{avg_det.item():.4f}',
                        'Avg_OCR_Loss': f'{avg_ocr.item():.4f}'
                    })

        return total_det_loss / total_plates, total_ocr_loss / total_plates

    def compute_loss_full_image(self, batch: dict, use_ocr_baseline=True) -> torch.Tensor:
        """Compute loss for full image detection"""

        batch = {k: v[0] for k, v in batch.items()}  # Remove batch dim

        # Apply adversarial patch to YOLO input
        patched_image, _ = self.apply_patch_to_image(
            batch['prep_image'].to(self.device).unsqueeze(0),
            batch['new_corners'].to(self.device).unsqueeze(0)
        )
        # Overwrite batch's image with patched version
        batch['prep_image'] = patched_image.squeeze()

        # Apply adversarial patch to full original image
        patched_image, _ = self.apply_patch_to_image(
            batch['orig_image'].to(self.device).unsqueeze(0),
            batch['orig_corners'].to(self.device).unsqueeze(0)
        )
        # Overwrite batch's image with patched version
        batch['orig_image'] = patched_image.squeeze()

        det_loss, ocr_loss = self.partial_loss(batch, use_ocr_baseline=use_ocr_baseline)
        return (det_loss + ocr_loss) / 2

    def train_epoch(self, optimizer: torch.optim.Optimizer, epoch: int) -> float:
        """Train for one epoch with gradient accumulation"""
        total_loss = 0.0
        accumulation_loss = 0.0
        step_count = 0
        num_updates = 0

        # Determine update frequency
        update_every = len(
            self.train_loader) if self.grad_accumulate is None else self.grad_accumulate
        effective_batch_size = update_every

        desc = f"Epoch {epoch+1} - Training (AccumSteps={update_every})"
        with tqdm(enumerate(self.train_loader), desc=desc, leave=False,
                  total=len(self.train_loader)) as pbar:

            for idx, batch in pbar:
                # Compute loss and scale by accumulation steps
                loss = self.compute_loss_full_image(batch)
                scaled_loss = loss / effective_batch_size

                # Backward pass (accumulate gradients)
                scaled_loss.backward()

                # Track losses
                accumulation_loss += loss.item()
                step_count += 1

                # Update model every update_every steps (or at the very end if None)
                if step_count % update_every == 0:
                    # Apply accumulated gradients
                    torch.nn.utils.clip_grad_norm_([self.patch], max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                    # Add accumulated loss to total
                    total_loss += accumulation_loss
                    num_updates += 1

                    # Memory cleanup after update
                    del loss, scaled_loss
                    if self.device == 'cuda':
                        torch.cuda.empty_cache()
                    elif self.device == 'mps':
                        torch.mps.empty_cache()

                    # Update progress bar with correct averaging
                    if self.grad_accumulate is None:
                        # Single update at end - show total progress
                        avg_loss = total_loss / (num_updates * update_every)
                        pbar.set_postfix({'Loss': f"{avg_loss:.4f}", 'Mode': 'End-Update'})
                    else:
                        # Regular accumulation updates
                        avg_loss = total_loss / (num_updates * self.grad_accumulate)
                        pbar.set_postfix({'Loss': f"{avg_loss:.4f}", 'Updates': num_updates})

                    # Reset accumulation tracking
                    accumulation_loss = 0.0
                else:
                    # Just cleanup current tensors, keep gradients
                    del loss, scaled_loss

                    # Show accumulation progress
                    current_batch_avg = accumulation_loss / (step_count % update_every)
                    pbar.set_postfix({
                        'AccumLoss': f"{current_batch_avg:.4f}",
                        'Progress': f"{step_count % update_every}/{update_every}"
                    })

            # Handle remaining accumulated gradients (should only happen with regular
            # grad_accumulate, not None)
            if step_count % update_every != 0 and self.grad_accumulate is not None:
                torch.nn.utils.clip_grad_norm_([self.patch], max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

                # Add remaining accumulated loss
                total_loss += accumulation_loss
                num_updates += 1

                # Final memory cleanup
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
                elif self.device == 'mps':
                    torch.mps.empty_cache()

        # Return average loss per batch (fixed calculation)
        total_batches_processed = num_updates * \
            update_every if self.grad_accumulate else len(self.train_loader)
        return total_loss / total_batches_processed

    def validate(self) -> Tuple[float, int]:
        """Validation pass on held-out data"""
        losses = []

        with torch.no_grad():
            for batch in self.val_loader:
                loss = self.compute_loss_full_image(batch)
                losses.append(loss.detach().cpu().item())

        return np.mean(losses)

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

        history = {'loss': [], 'val_score': [], 'learning_rate': []}

        best_loss = float('inf')
        patience_counter = 0

        print("\nStarting adversarial patch training")
        print(f"   Dataset: {len(self.train_loader) + len(self.val_loader)} images")
        print(f"   Patch size: {self.patch_height}×{self.patch_width}")
        print(f"   Device: {self.device}")
        print(f"   Epochs: {num_epochs}")
        print(f"   Initial LR: {learning_rate}")
        print(f"   Match detection: {self.match_detection}")
        print(
            f"   Impersonation target: {self.impersonation_target or 'None (penalize correct reading)'}")
        print("   Processing: Full 384x384 images only")
        print("-" * 60)

        for epoch in range(num_epochs):
            # Training and validation
            train_loss = self.train_epoch(optimizer, epoch)
            val_detection_score = self.validate()

            # Learning rate scheduling
            scheduler.step(train_loss)
            current_lr = optimizer.param_groups[0]['lr']

            # Record history
            history['loss'].append(train_loss)
            history['val_score'].append(val_detection_score)
            history['learning_rate'].append(current_lr)

            # Calculate detection reduction
            initial_detection = history['val_score'][0] if len(history['val_score']) > 0 else 1.0
            detection_reduction = (1 - val_detection_score / initial_detection) * 100

            # Print epoch summary
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Loss: {train_loss:.4f} | "
                  f"Val Det: {val_detection_score:.3f} | "
                  f"Reduction: {detection_reduction:+.1f}% | "
                  f"LR: {current_lr:.2e} | ")

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

        print("\nTraining completed!")
        print(f"   Best loss: {best_loss:.4f}")
        final_reduction = (1 - history['val_score'][-1] / history['val_score'][0]) * 100
        print(f"   Detection reduction: {final_reduction:.1f}%")

        return history


# Add this new function for loading patches:
def load_patch_from_file(patch_file: str, target_height: int, target_width: int, device):
    """Load and prepare a patch from an image file"""
    from PIL import Image
    import torchvision.transforms as transforms

    if not os.path.exists(patch_file):
        raise FileNotFoundError(f"Patch file not found: {patch_file}")

    try:
        # Load image
        patch_img = Image.open(patch_file).convert('RGB')
        print(f"Loaded patch image: {patch_img.size}")

        # Resize to target dimensions
        transform = transforms.Compose([
            transforms.Resize((target_height, target_width)),
            transforms.ToTensor()
        ])

        patch_tensor = transform(patch_img).to(device)

        # Ensure values are in [0, 1] range
        patch_tensor = torch.clamp(patch_tensor, 0, 1)

        print(f"Patch tensor shape: {patch_tensor.shape}")
        print(f"Patch value range: [{patch_tensor.min():.3f}, {patch_tensor.max():.3f}]")

        return patch_tensor

    except Exception as e:
        raise RuntimeError(f"Failed to load patch from {patch_file}: {str(e)}")


def test_detection_visualization(csv_path: str, output_path: str = "test_detection.png", **kwargs):
    """Test mode: visualize YOLO detections on a single image without patch using dataloader"""
    print("Running test mode - visualizing detections on sample image...")

    trainer = AdversarialPatchTrainer(csv_path, **kwargs)

    # Use the existing dataloader system - get first batch
    batch = next(iter(trainer.train_loader))
    batch = {k: v[0] for k, v in batch.items()}  # Remove batch dim like in training

    print(f"Testing on image from dataloader batch")

    # Get the preprocessed image and corners from dataloader
    prep_image = batch['prep_image'].to(trainer.device)
    corners = batch['new_corners'].to(trainer.device)
    ground_truth = trainer.corners_to_bbox(corners)  # Use same format as partial_loss

    # Run YOLO detection on full image using existing method
    with torch.no_grad():
        model_output = trainer.model(prep_image.unsqueeze(0))

    print(f"\nDetection Results:")
    print(f"Ground truth box: {ground_truth}")

    # Parse detections and calculate IoUs like in partial_loss
    all_detections = []
    for detection in model_output:
        pred_box = detection[1:5]
        conf = detection[6]
        class_id = detection[5]
        IoU = trainer.boxes_IoU(pred_box.unsqueeze(0), ground_truth.unsqueeze(0)).squeeze()

        # Convert back to original image coordinates if needed
        orig_projection = trainer.invert_bbox(pred_box.to('cpu'), batch['transform'])

        print(f"  Detection: Box={pred_box}, Conf={conf:.4f}, Class={class_id:.0f}, IoU={IoU:.4f}")

        all_detections.append({
            'box': pred_box.cpu().numpy(),
            'orig_box': orig_projection.numpy(),
            'confidence': conf.item(),
            'class_id': int(class_id.item()),
            'iou': IoU.item()
        })

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Original preprocessed image with detections
    prep_img = prep_image.permute(1, 2, 0).detach().cpu().numpy()
    ax1.imshow(prep_img)
    ax1.set_title('Preprocessed Image (384x384) with Detections')

    # Draw ground truth box
    ground_truth = ground_truth.detach().cpu().numpy()
    gt_rect = patches.Rectangle(
        (ground_truth[0], ground_truth[1]),
        ground_truth[2] - ground_truth[0],
        ground_truth[3] - ground_truth[1],
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

    analysis_text = f"Using new dataloader system\n"
    analysis_text += f"Dataset: {len(trainer.train_loader) + len(trainer.val_loader)} batches\n"
    analysis_text += f"Image size: 384x384 (preprocessed)\n"
    analysis_text += f"Match detection: {trainer.match_detection}\n"
    analysis_text += f"Impersonation target: {trainer.impersonation_target or 'None'}\n\n"

    analysis_text += f"Ground Truth Box:\n"
    analysis_text += f"  Top-left: [{ground_truth[0]:.1f}, {ground_truth[1]:.1f}]\n"
    analysis_text += f"  Bottom-right: [{ground_truth[2]:.1f}, {ground_truth[3]:.1f}]\n\n"

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


def debug_patch_application(csv_path: str, patch_file: str = None,
                            output_path: str = "debug_patch.png", **kwargs):
    """Debug mode: apply patch and visualize impact on detection"""
    print("Running debug patch mode...")

    # Initialize trainer with new options
    trainer = AdversarialPatchTrainer(csv_path, **kwargs)

    # Get random batch from dataloader
    import random
    batch = next(iter(trainer.train_loader))
    batch = {k: v[0] for k, v in batch.items()}  # Remove batch dim like in training

    print(f"Selected random image from {len(trainer.train_loader)} available batches")
    print(f"Match detection: {trainer.match_detection}")
    print(f"Impersonation target: {trainer.impersonation_target or 'None'}")

    # Get image and targets
    prep_image = batch['prep_image'].to(trainer.device)
    corners = batch['new_corners'].to(trainer.device)

    if trainer.match_detection:
        target_box = trainer.get_patch_bounding_box(corners)
        target_name = "Patch Bounding Box"
    else:
        target_box = trainer.corners_to_bbox(corners)
        target_name = "Ground Truth Box"

    print(f"Image shape: {prep_image.shape}")
    print(f"{target_name}: {target_box}")

    # Run detection on original image
    print("Running detection on original image...")
    with torch.no_grad():
        original_output = trainer.model(prep_image.unsqueeze(0))

    # Parse original detections using same logic as partial_loss
    original_detections = []
    best_original_iou = 0.0
    best_original_conf = 0.0
    best_original_detection = None

    for detection in original_output:
        pred_box = detection[1:5]
        conf = detection[6]
        class_id = detection[5]
        iou = trainer.boxes_IoU(pred_box.unsqueeze(0), target_box.unsqueeze(0)).squeeze()

        det_info = {
            'box': pred_box.cpu().numpy(),
            'confidence': conf.item(),
            'class_id': int(class_id.item()),
            'iou': iou.item()
        }
        original_detections.append(det_info)

        if iou.item() > best_original_iou:
            best_original_iou = iou.item()
            best_original_detection = det_info
        if conf.item() > best_original_conf:
            best_original_conf = conf.item()

    print(f"Original detections found: {len(original_detections)}")

    # Load or create patch
    if patch_file:
        print(f"Loading patch from: {patch_file}")
        patch_tensor = load_patch_from_file(patch_file, PATCH_HEIGHT, PATCH_WIDTH, trainer.device)
        # Convert to parameter format that matches trainer.patch (inverse of tanh normalization)
        # patch = tanh^(-1)((patch_tensor * 2) - 1) but clamp to avoid inf
        patch = torch.arctanh(torch.clamp(patch_tensor * 2 - 1, -0.99, 0.99))
    else:
        print("Using default white patch")
        # Create white patch (arctanh(0.99) ≈ 2.6, so this will give white after tanh normalization)
        patch = torch.full((3, PATCH_HEIGHT, PATCH_WIDTH), 2.6, device=trainer.device)

    # Temporarily replace trainer's patch for visualization
    original_patch = trainer.patch.data.clone()
    trainer.patch.data = patch

    # Apply patch using trainer's method
    print("Applying patch to image...")
    patched_image, patch_mask = trainer.apply_patch_to_image(
        prep_image.unsqueeze(0), corners.unsqueeze(0))

    if patched_image is None:
        raise RuntimeError("apply_patch_to_image returned None")
    if patch_mask is None:
        print("Warning: No patch mask generated (patch application may have failed)")

    patched_image = patched_image.squeeze(0)

    # Run detection on patched image
    print("Running detection on patched image...")
    with torch.no_grad():
        patched_output = trainer.model(patched_image.unsqueeze(0))

    # Parse patched detections
    patched_detections = []
    best_patched_iou = 0.0
    best_patched_conf = 0.0
    best_patched_detection = None

    for detection in patched_output:
        pred_box = detection[1:5]
        conf = detection[6]
        class_id = detection[5]
        iou = trainer.boxes_IoU(pred_box.unsqueeze(0), target_box.unsqueeze(0)).squeeze()

        det_info = {
            'box': pred_box.cpu().numpy(),
            'confidence': conf.item(),
            'class_id': int(class_id.item()),
            'iou': iou.item()
        }
        patched_detections.append(det_info)

        if iou.item() > best_patched_iou:
            best_patched_iou = iou.item()
            best_patched_detection = det_info
        if conf.item() > best_patched_conf:
            best_patched_conf = conf.item()

    print(f"Patched detections found: {len(patched_detections)}")

    # Restore original patch
    trainer.patch.data = original_patch

    # Calculate impact metrics
    if trainer.match_detection:
        # For match_detection, higher IoU is better (we want to maximize it)
        iou_change = best_patched_iou - best_original_iou
        iou_effectiveness = f"{iou_change:+.1f}% change"
    else:
        # Original behavior: lower IoU is better (we want to minimize it)
        iou_reduction = ((best_original_iou - best_patched_iou) /
                         best_original_iou * 100) if best_original_iou > 0 else 0
        iou_effectiveness = f"{iou_reduction:.1f}% reduction"

    conf_reduction = ((best_original_conf - best_patched_conf) /
                      best_original_conf * 100) if best_original_conf > 0 else 0
    detection_count_change = len(patched_detections) - len(original_detections)

    print(f"\nPatch Impact Results:")
    print(f"Target: {target_name}")
    print(f"Original - Best IoU: {best_original_iou:.4f}, Best Conf: {best_original_conf:.4f}")
    print(f"Patched  - Best IoU: {best_patched_iou:.4f}, Best Conf: {best_patched_conf:.4f}")
    print(f"IoU effectiveness: {iou_effectiveness}")
    print(f"Confidence reduction: {conf_reduction:.1f}%")
    print(f"Detection count change: {detection_count_change:+d}")

    # Create comprehensive visualization - same structure as before but update analysis text
    fig = plt.figure(figsize=(28, 20))

    # Row 1: Image comparisons
    # Original image with detections
    ax1 = plt.subplot(4, 4, 1)
    orig_img = prep_image.permute(1, 2, 0).detach().cpu().numpy()
    ax1.imshow(orig_img)
    ax1.set_title(f'Original Image\n({len(original_detections)} detections)')

    # Draw target box
    target_box_np = target_box.detach().cpu().numpy()
    target_rect = patches.Rectangle((target_box_np[0], target_box_np[1]),
                                    target_box_np[2] - target_box_np[0],
                                    target_box_np[3] - target_box_np[1],
                                    linewidth=3, edgecolor='green', facecolor='none',
                                    label=target_name)
    ax1.add_patch(target_rect)

    # Draw original detections (top 5)
    colors = ['red', 'orange', 'purple', 'brown', 'pink']
    for i, det in enumerate(original_detections[:5]):
        box = det['box']
        color = colors[i % len(colors)]
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                 linewidth=2, edgecolor=color, facecolor='none', alpha=0.8)
        ax1.add_patch(rect)
        ax1.text(box[0], box[1] - 5, f'{det["confidence"]:.3f}',
                 color=color, fontsize=8, weight='bold',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

    if best_original_detection:
        status_color = 'lightgreen' if not trainer.match_detection else 'lightblue'
        ax1.text(0.02, 0.98, f'Best: IoU={best_original_detection["iou"]:.3f}',
                 transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor=status_color, alpha=0.8))
    ax1.legend(loc='upper right')

    # Patched image with detections
    ax2 = plt.subplot(4, 4, 2)
    patched_img = patched_image.permute(1, 2, 0).detach().cpu().numpy()
    ax2.imshow(patched_img)
    ax2.set_title(f'Patched Image with Detections\n({len(patched_detections)} detections)')

    # Draw target box
    target_rect2 = patches.Rectangle((target_box_np[0], target_box_np[1]),
                                     target_box_np[2] - target_box_np[0],
                                     target_box_np[3] - target_box_np[1],
                                     linewidth=3, edgecolor='green', facecolor='none',
                                     label=target_name)
    ax2.add_patch(target_rect2)

    # Draw patched detections (top 5)
    for i, det in enumerate(patched_detections[:5]):
        box = det['box']
        color = colors[i % len(colors)]
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                 linewidth=2, edgecolor=color, facecolor='none', alpha=0.8)
        ax2.add_patch(rect)
        ax2.text(box[0], box[1] - 5, f'{det["confidence"]:.3f}',
                 color=color, fontsize=8, weight='bold',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

    if best_patched_detection:
        status_color = 'lightcoral' if not trainer.match_detection else 'lightgreen'
        ax2.text(0.02, 0.98, f'Best: IoU={best_patched_detection["iou"]:.3f}',
                 transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor=status_color, alpha=0.8))
    ax2.legend(loc='upper right')

    # Clean patched image without detection overlays
    ax3 = plt.subplot(4, 4, 3)
    ax3.imshow(patched_img)
    ax3.set_title('Clean Patched Image\n(no detection overlays)')
    ax3.axis('off')

    # Patch visualization
    ax4 = plt.subplot(4, 4, 4)
    patch_display = torch.tanh(patch) * 0.5 + 0.5  # Convert back to [0,1] for display
    patch_img = patch_display.detach().cpu().permute(1, 2, 0).numpy()
    ax4.imshow(patch_img)
    ax4.set_title(f'Applied Patch\n({PATCH_WIDTH}×{PATCH_HEIGHT})')
    ax4.axis('off')

    # Row 2: Analysis visualizations
    # Patch mask visualization
    ax5 = plt.subplot(4, 4, 5)
    if patch_mask is not None:
        mask_display = patch_mask[0, 0].detach().cpu().numpy()  # First channel of first batch
        ax5.imshow(mask_display, cmap='hot', alpha=0.8)
        ax5.set_title('Patch Application Mask')
    else:
        ax5.text(0.5, 0.5, 'No mask available\n(patch application failed)',
                 ha='center', va='center', transform=ax5.transAxes,
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        ax5.set_title('Patch Application Mask')
    ax5.axis('off')

    # Side-by-side difference
    ax6 = plt.subplot(4, 4, 6)
    diff_img = np.abs(patched_img - orig_img)
    ax6.imshow(diff_img)
    ax6.set_title('Absolute Difference\n(Patched - Original)')
    ax6.axis('off')

    # IoU comparison chart
    ax7 = plt.subplot(4, 4, 7)
    if original_detections and patched_detections:
        orig_ious = [det['iou'] for det in original_detections]
        patch_ious = [det['iou'] for det in patched_detections]

        ax7.hist(orig_ious, bins=10, alpha=0.7, label='Original', color='red', density=True)
        ax7.hist(patch_ious, bins=10, alpha=0.7, label='Patched', color='blue', density=True)
        ax7.set_xlabel(f'IoU with {target_name}')
        ax7.set_ylabel('Density')
        ax7.set_title('IoU Distribution')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
    else:
        ax7.text(0.5, 0.5, 'No detections\nfor comparison', ha='center', va='center')
        ax7.set_title('IoU Distribution')

    # Confidence comparison chart
    ax8 = plt.subplot(4, 4, 8)
    if original_detections and patched_detections:
        orig_confs = [det['confidence'] for det in original_detections]
        patch_confs = [det['confidence'] for det in patched_detections]

        ax8.hist(orig_confs, bins=10, alpha=0.7, label='Original', color='red', density=True)
        ax8.hist(patch_confs, bins=10, alpha=0.7, label='Patched', color='blue', density=True)
        ax8.set_xlabel('Confidence')
        ax8.set_ylabel('Density')
        ax8.set_title('Confidence Distribution')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
    else:
        ax8.text(0.5, 0.5, 'No detections\nfor comparison', ha='center', va='center')
        ax8.set_title('Confidence Distribution')

    # Row 3: Metrics
    # Metrics comparison bar chart
    ax9 = plt.subplot(4, 4, 9)
    metrics = ['Best IoU', 'Best Conf', '# Detections']
    original_values = [best_original_iou, best_original_conf, len(original_detections)]
    patched_values = [best_patched_iou, best_patched_conf, len(patched_detections)]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax9.bar(x - width / 2, original_values, width, label='Original', color='red', alpha=0.7)
    bars2 = ax9.bar(x + width / 2, patched_values, width, label='Patched', color='blue', alpha=0.7)

    ax9.set_ylabel('Score')
    ax9.set_title('Metrics Comparison')
    ax9.set_xticks(x)
    ax9.set_xticklabels(metrics, rotation=45)
    ax9.legend()
    ax9.grid(True, alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax9.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{height:.3f}' if height < 10 else f'{int(height)}',
                     ha='center', va='bottom', fontsize=8)

    # Row 4: Detailed analysis text panel (spanning bottom row)
    ax10 = plt.subplot(4, 4, (13, 16))  # Span entire bottom row
    ax10.axis('off')
    ax10.set_title('Detailed Impact Analysis', pad=20, fontsize=14, weight='bold')

    analysis_text = f"PATCH EFFECTIVENESS ANALYSIS\n{'='*50}\n\n"

    # Basic stats
    analysis_text += f"Configuration:\n"
    analysis_text += f"  • Match detection: {trainer.match_detection}\n"
    analysis_text += f"  • Impersonation target: {trainer.impersonation_target or 'None'}\n"
    analysis_text += f"  • Target: {target_name}\n"
    analysis_text += f"  • Total batches available: {len(trainer.train_loader)}\n"
    analysis_text += f"  • Image resolution: 384×384\n"
    analysis_text += f"  • Patch size: {PATCH_WIDTH}×{PATCH_HEIGHT}\n\n"

    # Detection counts
    analysis_text += f"Detection Counts:\n"
    analysis_text += f"  • Original: {len(original_detections)} detections\n"
    analysis_text += f"  • Patched:  {len(patched_detections)} detections\n"
    analysis_text += f"  • Change:   {detection_count_change:+d} detections\n\n"

    # Best detection metrics
    analysis_text += f"Best Detection Metrics:\n"
    analysis_text += f"  • Original IoU:  {best_original_iou:.4f}\n"
    analysis_text += f"  • Patched IoU:   {best_patched_iou:.4f}\n"
    analysis_text += f"  • IoU Change:    {iou_effectiveness}\n\n"

    analysis_text += f"  • Original Conf: {best_original_conf:.4f}\n"
    analysis_text += f"  • Patched Conf:  {best_patched_conf:.4f}\n"
    analysis_text += f"  • Conf Reduction: {conf_reduction:.1f}%\n\n"

    # Effectiveness assessment
    analysis_text += f"Effectiveness Assessment:\n"

    if trainer.match_detection:
        # For match_detection, higher IoU with patch area is better
        if best_patched_iou > best_original_iou + 0.3:
            status = "🟢 HIGHLY EFFECTIVE (Match Detection)"
            analysis_text += f"  Status: {status}\n"
            analysis_text += f"  • Strong IoU increase with patch area achieved\n"
            analysis_text += f"  • Patch successfully attracts detections\n"
        elif best_patched_iou > best_original_iou + 0.1:
            status = "🟡 MODERATELY EFFECTIVE (Match Detection)"
            analysis_text += f"  Status: {status}\n"
            analysis_text += f"  • Moderate IoU increase with patch area observed\n"
            analysis_text += f"  • Patch shows some attraction effect\n"
        elif best_patched_iou > best_original_iou:
            status = "🟠 MINIMALLY EFFECTIVE (Match Detection)"
            analysis_text += f"  Status: {status}\n"
            analysis_text += f"  • Minor IoU increase detected\n"
            analysis_text += f"  • Patch has limited attraction impact\n"
        else:
            status = "🔴 INEFFECTIVE (Match Detection)"
            analysis_text += f"  Status: {status}\n"
            analysis_text += f"  • No meaningful IoU increase\n"
            analysis_text += f"  • Patch fails to attract detections\n"
    else:
        # Original behavior: lower IoU with ground truth is better
        if iou_reduction > 30 or conf_reduction > 30:
            status = "🔴 HIGHLY EFFECTIVE"
            analysis_text += f"  Status: {status}\n"
            analysis_text += f"  • Significant detection degradation achieved\n"
            analysis_text += f"  • Patch successfully disrupts model performance\n"
        elif iou_reduction > 10 or conf_reduction > 10:
            status = "🟡 MODERATELY EFFECTIVE"
            analysis_text += f"  Status: {status}\n"
            analysis_text += f"  • Moderate detection impact observed\n"
            analysis_text += f"  • Patch shows some adversarial effect\n"
        elif iou_reduction > 0 or conf_reduction > 0:
            status = "🟢 MINIMALLY EFFECTIVE"
            analysis_text += f"  Status: {status}\n"
            analysis_text += f"  • Minor detection changes detected\n"
            analysis_text += f"  • Patch has limited adversarial impact\n"
        else:
            status = "⚫ INEFFECTIVE"
            analysis_text += f"  Status: {status}\n"
            analysis_text += f"  • No meaningful detection degradation\n"
            analysis_text += f"  • Patch fails to achieve adversarial effect\n"

    # Additional insights
    if len(original_detections) == 0:
        analysis_text += f"\n⚠️  WARNING: No detections on original image!\n"
        analysis_text += f"   This may indicate issues with the model or ground truth.\n"
    elif len(patched_detections) == 0 and len(original_detections) > 0:
        analysis_text += f"\n✅ COMPLETE SUPPRESSION ACHIEVED!\n"
        analysis_text += f"   Patch completely eliminates all detections.\n"

    if patch_mask is None:
        analysis_text += f"\n⚠️  WARNING: Patch application failed!\n"
        analysis_text += f"   No valid mask generated. Check patch placement logic.\n"

    ax10.text(0.02, 0.98, analysis_text, transform=ax10.transAxes, fontsize=10,
              verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.9))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nDebug visualization saved to: {output_path}")
    print(f"Patch effectiveness: {status}")

    # Return metrics for programmatic use
    return {
        'original_detections': len(original_detections),
        'patched_detections': len(patched_detections),
        'best_original_iou': best_original_iou,
        'best_patched_iou': best_patched_iou,
        'best_original_conf': best_original_conf,
        'best_patched_conf': best_patched_conf,
        'iou_effectiveness': iou_effectiveness,
        'conf_reduction_percent': conf_reduction,
        'detection_count_change': detection_count_change,
        'effectiveness_status': status,
        'match_detection': trainer.match_detection,
        'target_type': target_name
    }


def debug_ocr_accuracy(csv_path: str, patch_file: str = None,
                       output_path: str = "debug_ocr.png", **kwargs):
    """
    Enhanced OCR test: compare OCR results with and without patch showing crops and losses.
    """
    print("Testing OCR accuracy with/without patch...")

    # Initialize trainer with new options
    trainer = AdversarialPatchTrainer(csv_path, **kwargs)
    alphabet = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_'

    print(f"OCR target: {trainer.impersonation_target or 'VRJ7774 (penalized)'}")

    # Load patch if provided, otherwise use trainer's current patch
    if patch_file:
        print(f"Loading patch from: {patch_file}")
        patch_tensor = load_patch_from_file(patch_file, PATCH_HEIGHT, PATCH_WIDTH, trainer.device)
        patch = torch.arctanh(torch.clamp(patch_tensor * 2 - 1, -0.99, 0.99))
        trainer.patch.data = patch

    # Get first batch (already shuffled)
    batch = next(iter(trainer.train_loader))
    batch = {k: v[0] for k, v in batch.items()}  # Remove batch dim

    print("Processing sample image...")

    def logits_to_text(logits, alphabet_str):
        """Convert OCR logits to text"""
        probs = torch.softmax(logits, dim=-1)
        pred_chars = torch.argmax(probs, dim=-1).squeeze(0)
        text = ""
        for char_idx in pred_chars:
            char_idx = char_idx.item()
            if char_idx < len(alphabet_str) and alphabet_str[char_idx] != '_':
                text += alphabet_str[char_idx]
        return text.strip()

    def run_ocr_pipeline(use_patch=False):
        """Run detection + OCR pipeline, return all intermediate results"""
        batch_copy = {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}

        if use_patch:
            # Apply patch to prep image
            patched_image, _ = trainer.apply_patch_to_image(
                batch_copy['prep_image'].to(trainer.device).unsqueeze(0),
                batch_copy['new_corners'].to(trainer.device).unsqueeze(0)
            )
            if patched_image is not None:
                batch_copy['prep_image'] = patched_image.squeeze(0)

            # Apply patch to orig image
            patched_image, _ = trainer.apply_patch_to_image(
                batch_copy['orig_image'].to(trainer.device).unsqueeze(0),
                batch_copy['orig_corners'].to(trainer.device).unsqueeze(0)
            )
            if patched_image is not None:
                batch_copy['orig_image'] = patched_image.squeeze(0)

        # YOLO detection
        prep_image = batch_copy['prep_image'].to(trainer.device)
        model_output = trainer.model(prep_image.unsqueeze(0))

        # Find best detection
        corners = batch_copy['new_corners'].to(trainer.device)

        if trainer.match_detection:
            target_box = trainer.get_patch_bounding_box(corners)
        else:
            target_box = trainer.corners_to_bbox(corners)

        best_detection = None
        best_iou = 0.0

        for detection in model_output:
            pred_box = detection[1:5]
            iou = trainer.boxes_IoU(pred_box.unsqueeze(0), target_box.unsqueeze(0)).squeeze()
            if iou > best_iou:
                best_iou = iou.item()
                best_detection = detection

        if best_detection is None:
            return {
                'text': None,
                'conf': 0.0,
                'iou': 0.0,
                'ocr_loss': float('inf'),
                'image': batch_copy['prep_image'],
                'cropped_plate': None,
                'ocr_logits': None,
                'detection_box': None
            }

        # Transform detection back to original coordinates
        pred_box = best_detection[1:5]
        conf = best_detection[6].item()
        orig_projection = trainer.invert_bbox(pred_box.to('cpu'), batch_copy['transform'])
        corners_box = trainer.bbox_to_corners(orig_projection, device='cpu')

        # Crop and resize for OCR
        cropped_plate = kornia.geometry.crop_and_resize(
            batch_copy['orig_image'].unsqueeze(0),
            corners_box,
            trainer.ocr_input_shape[:2],
            mode='bilinear',
            align_corners=True
        ).to(trainer.device)

        # Run OCR
        ocr_input = cropped_plate.permute(0, 2, 3, 1) * 255  # NHWC format
        with torch.no_grad():
            ocr_logits = trainer.ocr(ocr_input)

        # Calculate OCR loss (use base loss function, not the modified version from partial_loss)
        ocr_loss = trainer.ocr_loss(trainer.ocr_target, ocr_logits).item()

        # Convert logits to text
        ocr_text = logits_to_text(ocr_logits, alphabet)

        return {
            'text': ocr_text,
            'conf': conf,
            'iou': best_iou,
            'ocr_loss': ocr_loss,
            'image': batch_copy['prep_image'],
            'cropped_plate': cropped_plate.squeeze(0),  # Remove batch dim
            'ocr_logits': ocr_logits,
            'detection_box': pred_box
        }

    # Test without patch
    try:
        results_no_patch = run_ocr_pipeline(use_patch=False)
        print(
            f"Without patch - Text: '{results_no_patch['text']}', Loss: {results_no_patch['ocr_loss']:.4f}")
    except Exception as e:
        print(f"Error without patch: {e}")
        raise

    # Test with patch
    try:
        results_with_patch = run_ocr_pipeline(use_patch=True)
        print(
            f"With patch - Text: '{results_with_patch['text']}', Loss: {results_with_patch['ocr_loss']:.4f}")
    except Exception as e:
        print(f"Error with patch: {e}")
        raise

    # Create enhanced visualization with 3x3 grid
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))

    # Row 1: Full images
    # Original image
    orig_img = results_no_patch['image'].permute(1, 2, 0).detach().cpu().numpy()
    axes[0, 0].imshow(orig_img)
    axes[0, 0].set_title(f'Without Patch\nOCR: "{results_no_patch["text"] or "NONE"}"')
    axes[0, 0].axis('off')

    # Patched image
    patched_img = results_with_patch['image'].permute(1, 2, 0).detach().cpu().numpy()
    axes[0, 1].imshow(patched_img)
    axes[0, 1].set_title(f'With Patch\nOCR: "{results_with_patch["text"] or "NONE"}"')
    axes[0, 1].axis('off')

    # Patch visualization
    if hasattr(trainer, 'patch'):
        patch_display = torch.tanh(trainer.patch) * 0.5 + 0.5
        patch_img = patch_display.detach().cpu().permute(1, 2, 0).numpy()
        axes[0, 2].imshow(patch_img)
        axes[0, 2].set_title(f'Applied Patch\n({PATCH_WIDTH}×{PATCH_HEIGHT})')
    else:
        axes[0, 2].text(0.5, 0.5, 'No patch', ha='center', va='center')
        axes[0, 2].set_title('Patch')
    axes[0, 2].axis('off')

    # Row 2: Cropped plates fed to OCR
    if results_no_patch['cropped_plate'] is not None:
        crop_no_patch = results_no_patch['cropped_plate'].permute(1, 2, 0).detach().cpu().numpy()
        axes[1, 0].imshow(crop_no_patch)
        axes[1, 0].set_title(
            f'OCR Input (No Patch)\nSize: {crop_no_patch.shape[:2]}\nLoss: {results_no_patch["ocr_loss"]:.4f}')
    else:
        axes[1, 0].text(0.5, 0.5, 'No detection\nfound', ha='center', va='center')
        axes[1, 0].set_title('OCR Input (No Patch)\nLoss: ∞')
    axes[1, 0].axis('off')

    if results_with_patch['cropped_plate'] is not None:
        crop_with_patch = results_with_patch['cropped_plate'].permute(
            1, 2, 0).detach().cpu().numpy()
        axes[1, 1].imshow(crop_with_patch)
        axes[1, 1].set_title(
            f'OCR Input (With Patch)\nSize: {crop_with_patch.shape[:2]}\nLoss: {results_with_patch["ocr_loss"]:.4f}')
    else:
        axes[1, 1].text(0.5, 0.5, 'No detection\nfound', ha='center', va='center')
        axes[1, 1].set_title('OCR Input (With Patch)\nLoss: ∞')
    axes[1, 1].axis('off')

    # Side-by-side crop comparison (if both available)
    if (results_no_patch['cropped_plate'] is not None and
            results_with_patch['cropped_plate'] is not None):

        # Compute difference
        crop_diff = np.abs(crop_with_patch - crop_no_patch)
        axes[1, 2].imshow(crop_diff)
        axes[1, 2].set_title('OCR Input Difference\n|With Patch - No Patch|')
    else:
        axes[1, 2].text(0.5, 0.5, 'Cannot compare\ncrops', ha='center', va='center')
        axes[1, 2].set_title('OCR Input Difference')
    axes[1, 2].axis('off')

    # Row 3: Analysis and metrics
    # Loss comparison chart
    axes[2, 0].axis('off')

    if (results_no_patch['ocr_loss'] < float('inf') and
            results_with_patch['ocr_loss'] < float('inf')):

        losses = [results_no_patch['ocr_loss'], results_with_patch['ocr_loss']]
        labels = ['No Patch', 'With Patch']
        colors = ['blue', 'red']

        bars = axes[2, 0].bar(labels, losses, color=colors, alpha=0.7)
        axes[2, 0].set_ylabel('OCR Loss')
        axes[2, 0].set_title('OCR Loss Comparison')
        axes[2, 0].grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, loss in zip(bars, losses):
            height = bar.get_height()
            axes[2, 0].text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                            f'{loss:.4f}', ha='center', va='bottom', fontsize=10)
    else:
        axes[2, 0].text(0.5, 0.5, 'No valid losses\nto compare', ha='center', va='center')
        axes[2, 0].set_title('OCR Loss Comparison')

    # Detection metrics comparison
    axes[2, 1].axis('off')

    metrics = ['IoU', 'Confidence']
    no_patch_vals = [results_no_patch['iou'], results_no_patch['conf']]
    with_patch_vals = [results_with_patch['iou'], results_with_patch['conf']]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = axes[2, 1].bar(x - width / 2, no_patch_vals, width,
                           label='No Patch', color='blue', alpha=0.7)
    bars2 = axes[2, 1].bar(x + width / 2, with_patch_vals, width,
                           label='With Patch', color='red', alpha=0.7)

    axes[2, 1].set_ylabel('Score')
    axes[2, 1].set_title('Detection Metrics')
    axes[2, 1].set_xticks(x)
    axes[2, 1].set_xticklabels(metrics)
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[2, 1].text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                            f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    # Detailed results summary
    axes[2, 2].axis('off')
    axes[2, 2].set_title('Detailed Analysis', pad=20, fontsize=12, weight='bold')

    # Calculate changes
    loss_change = results_with_patch['ocr_loss'] - results_no_patch['ocr_loss']
    conf_change = results_with_patch['conf'] - results_no_patch['conf']
    iou_change = results_with_patch['iou'] - results_no_patch['iou']

    # Determine status based on configuration
    target_text = trainer.impersonation_target or 'VRJ7774'

    if results_no_patch['text'] and results_with_patch['text']:
        if trainer.impersonation_target:
            # For impersonation, we want the OCR to read the target
            if results_with_patch['text'] == trainer.impersonation_target:
                status = "IMPERSONATION SUCCESS"
                color = 'lightgreen'
            elif results_no_patch['text'] == results_with_patch['text']:
                status = "NO CHANGE"
                color = 'orange'
            else:
                status = "TEXT CHANGED (NOT TARGET)"
                color = 'yellow'
        else:
            # Original behavior - we want to prevent correct reading
            if results_no_patch['text'] == results_with_patch['text']:
                if loss_change > 0:
                    status = "TEXT SAME, LOSS INCREASED"
                    color = 'orange'
                else:
                    status = "NO CHANGE"
                    color = 'green'
            else:
                status = "TEXT CHANGED"
                color = 'red'
    elif results_no_patch['text'] and not results_with_patch['text']:
        status = "DETECTION SUPPRESSED"
        color = 'darkred'
    elif not results_no_patch['text'] and results_with_patch['text']:
        if trainer.impersonation_target and results_with_patch['text'] == trainer.impersonation_target:
            status = "IMPERSONATION SUCCESS (FROM NOTHING)"
            color = 'lightgreen'
        else:
            status = "DETECTION ENABLED"
            color = 'blue'
    else:
        status = "NO DETECTIONS"
        color = 'gray'

    results_text = f"OCR ANALYSIS RESULTS\n{'='*20}\n\n"
    results_text += f"Configuration:\n"
    results_text += f"  Target: {target_text}\n"
    results_text += f"  Mode: {'Impersonation' if trainer.impersonation_target else 'Disruption'}\n\n"

    results_text += f"Without Patch:\n"
    results_text += f"  Text: '{results_no_patch['text'] or 'NONE'}'\n"
    results_text += f"  Loss: {results_no_patch['ocr_loss']:.4f}\n"
    results_text += f"  IoU:  {results_no_patch['iou']:.3f}\n"
    results_text += f"  Conf: {results_no_patch['conf']:.3f}\n\n"

    results_text += f"With Patch:\n"
    results_text += f"  Text: '{results_with_patch['text'] or 'NONE'}'\n"
    results_text += f"  Loss: {results_with_patch['ocr_loss']:.4f}\n"
    results_text += f"  IoU:  {results_with_patch['iou']:.3f}\n"
    results_text += f"  Conf: {results_with_patch['conf']:.3f}\n\n"

    results_text += f"Changes:\n"
    if results_with_patch['ocr_loss'] < float(
            'inf') and results_no_patch['ocr_loss'] < float('inf'):
        results_text += f"  Loss:  {loss_change:+.4f}\n"
    else:
        results_text += f"  Loss:  N/A\n"
    results_text += f"  IoU:   {iou_change:+.3f}\n"
    results_text += f"  Conf:  {conf_change:+.3f}\n\n"

    results_text += f"Status: {status}\n"

    # Loss interpretation
    if trainer.impersonation_target:
        if results_with_patch['text'] == trainer.impersonation_target:
            results_text += f"Impact: SUCCESS\n"
        else:
            results_text += f"Impact: FAILED\n"
    else:
        if loss_change > 0.5:
            results_text += f"OCR Impact: HIGH\n"
        elif loss_change > 0.1:
            results_text += f"OCR Impact: MODERATE\n"
        elif loss_change > 0.0:
            results_text += f"OCR Impact: LOW\n"
        else:
            results_text += f"OCR Impact: MINIMAL\n"

    axes[2, 2].text(0.05, 0.95, results_text, transform=axes[2, 2].transAxes, fontsize=9,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.2))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Print results
    print(f"\nEnhanced OCR Results:")
    print(
        f"  Target: {target_text} ({'Impersonation' if trainer.impersonation_target else 'Disruption'} mode)")
    print(
        f"  Without patch: '{results_no_patch['text'] or 'NO DETECTION'}' (loss: {results_no_patch['ocr_loss']:.4f})")
    print(
        f"  With patch:    '{results_with_patch['text'] or 'NO DETECTION'}' (loss: {results_with_patch['ocr_loss']:.4f})")
    print(f"  Status: {status}")
    print(f"  Loss change: {loss_change:+.4f}")
    print(f"  Visualization saved: {output_path}")

    return {
        'without_patch': {
            'text': results_no_patch['text'],
            'loss': results_no_patch['ocr_loss'],
            'conf': results_no_patch['conf'],
            'iou': results_no_patch['iou']
        },
        'with_patch': {
            'text': results_with_patch['text'],
            'loss': results_with_patch['ocr_loss'],
            'conf': results_with_patch['conf'],
            'iou': results_with_patch['iou']
        },
        'changes': {
            'loss': loss_change,
            'conf': conf_change,
            'iou': iou_change
        },
        'status': status,
        'target': target_text,
        'mode': 'impersonation' if trainer.impersonation_target else 'disruption'
    }


def main():
    parser = argparse.ArgumentParser(description='Adversarial Patch Training')
    parser.add_argument('--test', action='store_true',
                        help='Test mode: visualize detections on single image without patch')
    parser.add_argument('--debug-patch', nargs='?', const=True, default=False,
                        help='Debug mode: apply patch and visualize impact. '
                        'Optionally specify patch file path (default: white patch)')
    parser.add_argument('--output', default='test_detection.png',
                        help='Output path for test/debug visualization')
    parser.add_argument('--debug-ocr', nargs='?', const=True, default=False,
                        help='Optionally specify a patch file')
    parser.add_argument('--match-detection', action='store_true',
                        help='Maximize IoU with patch bounding box instead of minimizing with ground truth')
    parser.add_argument('--impersonation-target', type=str, default=None,
                        help='Target plate text for impersonation (e.g., "ABC123"). If not provided, '
                        'uses disruption mode to prevent correct reading of VRJ7774')
    args = parser.parse_args()

    # Configuration
    CSV_PATH = "preproc_labels.csv"
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.1

    # Common trainer kwargs
    trainer_kwargs = {
        'device': 'cpu',
        'grad_accumulate': 1,
        'match_detection': args.match_detection,
        'impersonation_target': args.impersonation_target
    }

    if args.test:
        try:
            test_detection_visualization(CSV_PATH, args.output, **trainer_kwargs)
        except Exception as e:
            print(f"Test failed: {e}")
            raise
        return

    if args.debug_patch:
        try:
            patch_file = args.debug_patch if isinstance(args.debug_patch, str) else None
            debug_patch_application(CSV_PATH, patch_file, args.output, **trainer_kwargs)
        except Exception as e:
            print(f"Debug patch failed: {e}")
            raise
        return

    if args.debug_ocr:
        try:
            patch_file = args.debug_ocr if isinstance(args.debug_ocr, str) else None
            results = debug_ocr_accuracy(CSV_PATH, patch_file, args.output, **trainer_kwargs)
        except Exception as e:
            print(f"OCR debug failed: {e}")
            raise
        return

    # Normal training mode
    try:
        trainer = AdversarialPatchTrainer(CSV_PATH, **trainer_kwargs)

        history = trainer.train(
            num_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            save_interval=1,
            early_stop_patience=20
        )

        # Plot training results
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 10))

        # Loss curve
        ax1.plot(history['loss'], 'b-', label='Training Loss')
        ax1.plot(history['val_score'], 'r-', label='Validation Loss')
        ax1.set_title('Loss Over Time')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Learning rate
        ax2.semilogy(history['learning_rate'], 'purple', label='Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate (log scale)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Final adversarial patch
        final_patch = torch.tanh(trainer.patch) * 0.5 + 0.5
        final_patch_np = final_patch.detach().cpu().permute(1, 2, 0).numpy()
        ax3.imshow(final_patch_np)
        ax3.set_title(f'Final Adversarial Patch ({PATCH_WIDTH}×{PATCH_HEIGHT})')
        ax3.axis('off')

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
