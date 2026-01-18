#!/usr/bin/env python3
"""
Black-box adversarial patch optimization via model extraction.

This module provides tools to optimize adversarial patches against black-box
ALPR systems by iteratively:
1. Extracting a surrogate model that mimics the black-box behavior
2. Optimizing patches against the surrogate
3. Re-extracting to match black-box behavior in the adversarial regime

Usage:
    from model_extraction import BlackBoxModel, ALPRResult, optimize_patch_bb

    class MyALPR(BlackBoxModel):
        def evaluate(self, images):
            # Call your black-box API
            return [ALPRResult(text="ABC123", confidence=0.95) for _ in images]

    history = optimize_patch_bb(
        black_box=MyALPR(),
        csv_path="preproc_labels.csv",
        num_epochs=100
    )
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import random
import math
from collections import deque

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
import numpy as np
from tqdm import tqdm
import kornia

from optimize_patch import (
    ALPRModels,
    AdversarialPatchTrainer,
    TrainerConfig,
    PatchAdapter,
    create_focal_cce_loss,
    text_to_target_tensor,
    logits_to_text,
    corners_to_bbox,
    compute_iou,
    invert_bbox,
    bbox_to_corners,
    OCR_ALPHABET,
    OCR_MAX_SLOTS,
    OCR_INPUT_SHAPE,
    PATCH_HEIGHT,
    PATCH_WIDTH,
)


# =============================================================================
# Black-Box Surrogate Model
# =============================================================================

class BlackBoxSurrogate(nn.Module):
    """
    Lightweight regression model that predicts black-box ALPR behavior.

    Given an adversarial patch and homography transformation, predicts:
    - OCR loss (how different the prediction will be from ground truth)
    - Detection confidence

    This replaces the adapter-based approach with a simple query-based model.
    """

    def __init__(self, patch_height: int = 256, patch_width: int = 512):
        super().__init__()

        # CNN encoder for patch
        # Input: 3x256x512 (patch)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  # 32x128x256
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 64x64x128
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 128x32x64
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # 256x16x32
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=2, padding=1),  # 256x8x16
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)  # 256x1x1

        # Homography encoder (processes 3x3 matrix → 8 params)
        # Homography has 8 DOF (9 params with scale normalization)
        self.homography_encoder = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

        # Fusion layer (concatenate CNN features + homography features)
        # 256 (CNN) + 128 (homography) = 384
        self.fc_fusion = nn.Sequential(
            nn.Linear(256 + 128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        # OCR loss head (predicts focal CCE loss, typically 0-10 range)
        self.ocr_loss_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Softplus()  # Ensure positive output
        )

        # Confidence head (predicts confidence in [0, 1])
        self.confidence_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )

    def forward(
        self,
        patch: torch.Tensor,
        homography: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict black-box behavior from patch and homography.

        Args:
            patch: Adversarial patch [B, 3, 256, 512]
            homography: Homography matrix [B, 3, 3] or [B, 8] (flattened without last scale)

        Returns:
            Tuple of (ocr_loss, confidence) predictions, each [B, 1]
        """
        # Encode patch through CNN
        x = self.conv1(patch)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        # Global pooling
        x = self.gap(x)  # [B, 256, 1, 1]
        patch_features = x.view(x.size(0), -1)  # [B, 256]

        # Flatten homography if needed and normalize
        if homography.dim() == 3:  # [B, 3, 3]
            # Take first 8 parameters (exclude h33 which is typically 1)
            h_flat = homography.view(homography.size(0), 9)[:, :8]  # [B, 8]
        else:  # Already [B, 8]
            h_flat = homography

        # Encode homography
        homography_features = self.homography_encoder(h_flat)  # [B, 128]

        # Concatenate features
        combined = torch.cat([patch_features, homography_features], dim=1)  # [B, 384]

        # Fusion
        fused = self.fc_fusion(combined)  # [B, 256]

        # Predictions
        ocr_loss = self.ocr_loss_head(fused)  # [B, 1]
        confidence = self.confidence_head(fused)  # [B, 1]

        return ocr_loss, confidence


# =============================================================================
# Learning Rate Schedulers
# =============================================================================

class LinearWarmupCosineAnnealingLR(optim.lr_scheduler._LRScheduler):
    """
    Linear warmup followed by cosine annealing learning rate scheduler.

    Args:
        optimizer: Wrapped optimizer
        warmup_epochs: Number of epochs for linear warmup
        max_epochs: Total number of epochs (warmup + annealing)
        min_lr: Minimum learning rate (default: 0)
        last_epoch: The index of last epoch (default: -1)
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        min_lr: float = 0.0,
        last_epoch: int = -1
    ):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / max(1, self.warmup_epochs)
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / max(1, self.max_epochs - self.warmup_epochs)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_factor
                for base_lr in self.base_lrs
            ]


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ALPRResult:
    """Result from a black-box ALPR system."""
    text: Optional[str]  # Detected plate text, None if no detection
    confidence: float  # Detection confidence (0-1)

    def __post_init__(self):
        if self.text is not None:
            self.text = self.text.upper().strip()


@dataclass
class FinetuneMetrics:
    """Metrics from surrogate fine-tuning."""
    ocr_loss: float
    confidence_mse: float
    num_samples: int
    converged: bool


@dataclass
class EpochResult:
    """Results from one optimization epoch."""
    epoch: int
    patch_loss: float
    finetune_metrics: FinetuneMetrics
    black_box_success_rate: float  # Rate at which black-box correctly reads plates


# =============================================================================
# Abstract Black-Box Model
# =============================================================================

class BlackBoxModel(ABC):
    """
    Abstract base class for black-box ALPR systems.

    Users should extend this class and implement the evaluate() method
    to interface with their specific ALPR system.
    """

    @abstractmethod
    def evaluate(
        self,
        images: List[torch.Tensor],
        corners: Optional[List[torch.Tensor]] = None
    ) -> List[ALPRResult]:
        """
        Evaluate the black-box ALPR on a batch of images.

        Args:
            images: List of image tensors [C, H, W] in [0, 1] range, RGB format
            corners: Optional list of ground truth corner tensors [4, 2] for IoU-based selection

        Returns:
            List of ALPRResult, one per image. If no plate detected,
            return ALPRResult(text=None, confidence=0.0)
        """
        pass


# =============================================================================
# Replay Buffer
# =============================================================================

class ReplayBuffer:
    """
    Manages patches and their evaluation samples with exponential decay weighting.

    Uses exponential decay to determine sampling frequency: recent patches appear
    more often, older patches fade out. Randomly samples homographies/results for
    each patch to maintain diversity.
    """

    def __init__(
        self,
        decay_half_life: int = 4
    ):
        """
        Args:
            decay_half_life: Half-life for exponential decay (in epochs)
        """
        self.decay_half_life = decay_half_life
        self.decay_rate = math.log(2) / decay_half_life

        # Storage: list of (epoch, patch_tensor, samples_list)
        # samples_list contains metadata dicts (corners, transform, bb_result, gt_text)
        self.patches: List[Tuple[int, torch.Tensor, List[Dict]]] = []
        self.current_epoch = 0

    def add_patch_epoch(
        self,
        epoch: int,
        patch: torch.Tensor,
        samples: List[Dict]
    ):
        """
        Add patch and samples from an optimization epoch.

        Args:
            epoch: Current epoch number
            patch: The patch tensor used
            samples: List of metadata dicts (corners, transform, bb_result, gt_text)
        """
        self.current_epoch = epoch
        self.patches.append((epoch, patch.clone().detach().cpu(), samples))

    def get_training_samples(self) -> List[Dict]:
        """
        Sample from all patches according to exponential decay weights.

        Each patch contributes: count = total_samples * decay_weight / sum(weights)
        For each contribution, randomly pick one sample from that patch's samples.

        Returns:
            List of metadata dicts for surrogate training
        """
        if not self.patches:
            return []

        # Calculate decay weights for each patch
        weights = []
        for epoch, _, samples in self.patches:
            age = self.current_epoch - epoch
            weight = math.exp(-self.decay_rate * age)
            weights.append(weight)

        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            weights = [1.0] * len(weights)
            total_weight = len(weights)
        normalized_weights = [w / total_weight for w in weights]

        # Target total samples (use average dataset size * 3)
        # This gives reasonable coverage across all patches
        target_total = 300  # Reasonable default

        result = []

        # For each patch, sample appropriate number based on weight
        for i, (epoch, patch, samples) in enumerate(self.patches):
            # How many times should this patch appear?
            count = int(round(target_total * normalized_weights[i]))

            if count > 0 and samples:
                # Randomly sample from this patch's samples (with replacement)
                selected_indices = np.random.choice(
                    len(samples),
                    size=count,
                    replace=True
                )
                for idx in selected_indices:
                    result.append(samples[idx])

        return result

    def __len__(self) -> int:
        """Total number of samples across all patches."""
        total = 0
        for _, _, samples in self.patches:
            total += len(samples)
        return total


# =============================================================================
# Surrogate Trainer
# =============================================================================

class SurrogateTrainer:
    """
    Trains a lightweight query-based surrogate to predict black-box behavior.

    Instead of fine-tuning ALPR models or using adapters, this trains a simple
    regression model that predicts:
    - OCR loss (how different black-box prediction is from ground truth)
    - Detection confidence

    Given a patched image, the surrogate directly predicts these metrics without
    running the full ALPR pipeline.
    """

    def __init__(
        self,
        trainer: 'AdversarialPatchTrainer',
        device: str,
        ocr_loss_threshold: float = 0.3,
        confidence_mse_threshold: float = 0.1,
        learning_rate: float = 5e-3,
        max_epochs: int = 100
    ):
        """
        Args:
            trainer: AdversarialPatchTrainer instance (for patch application)
            device: Target device
            ocr_loss_threshold: Maximum MSE of OCR loss predictions for convergence
            confidence_mse_threshold: Maximum MSE of confidence predictions for convergence
            learning_rate: Learning rate for surrogate training
            max_epochs: Maximum training epochs before giving up
        """
        self.trainer = trainer
        self.device = device
        self.ocr_loss_threshold = ocr_loss_threshold
        self.confidence_mse_threshold = confidence_mse_threshold
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

        # Create surrogate model
        self.surrogate = BlackBoxSurrogate().to(device)

        # For computing ground truth OCR loss
        self.ocr_loss_fn = create_focal_cce_loss(len(OCR_ALPHABET))

    def fine_tune(
        self,
        samples: List[Dict],
        batch_size: int = 32,
        verbose: bool = True,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
    ) -> FinetuneMetrics:
        """
        Train surrogate model until convergence or max epochs.

        Each epoch processes the entire dataset with batched gradient updates.
        Convergence is checked when MSE of both predictions is below threshold.

        Args:
            samples: List of metadata dicts (dataset_idx, corners, bb_result, gt_text)
            batch_size: Training batch size for gradient updates
            verbose: Whether to show progress bar
            optimizer: Optional pre-created optimizer (reuses across epochs)
            scheduler: Optional pre-created scheduler

        Returns:
            FinetuneMetrics with final metrics
        """
        if not samples:
            return FinetuneMetrics(
                ocr_loss=float('inf'),
                confidence_mse=float('inf'),
                num_samples=0,
                converged=False
            )

        # Set up optimizer (create if not provided)
        if optimizer is None:
            optimizer = optim.Adam(self.surrogate.parameters(), lr=self.learning_rate)

        # Learning rate scheduler (create if not provided)
        if scheduler is None:
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=10,
                max_epochs=self.max_epochs,
                min_lr=1e-6
            )

        converged = False

        for epoch in range(self.max_epochs):
            # Shuffle samples at start of each epoch
            random.shuffle(samples)

            # Training pass over entire dataset
            epoch_ocr_mse = 0.0
            epoch_conf_mse = 0.0
            samples_processed = 0

            desc = f"Surrogate epoch {epoch + 1}/{self.max_epochs}"
            pbar = tqdm(range(0, len(samples), batch_size), desc=desc,
                        disable=not verbose, leave=False)

            for i in pbar:
                batch = samples[i:i + batch_size]
                ocr_mse, conf_mse, valid_samples = self._train_step(batch, optimizer)

                epoch_ocr_mse += ocr_mse
                epoch_conf_mse += conf_mse
                samples_processed += valid_samples

                pbar.set_postfix({
                    'ocr_mse': f'{epoch_ocr_mse / max(1, samples_processed):.4f}',
                    'conf_mse': f'{epoch_conf_mse / max(1, samples_processed):.4f}'
                })

            pbar.close()

            # Compute average MSEs
            avg_ocr_mse = epoch_ocr_mse / max(1, samples_processed)
            avg_conf_mse = epoch_conf_mse / max(1, samples_processed)

            if verbose:
                lr = optimizer.param_groups[0]['lr']
                print(f"  Epoch {epoch + 1}: ocr_mse={avg_ocr_mse:.4f}, "
                      f"conf_mse={avg_conf_mse:.4f} | LR={lr:.2e}")

            # Step scheduler
            scheduler.step()

            # Check convergence
            if (avg_ocr_mse <= self.ocr_loss_threshold and
                    avg_conf_mse <= self.confidence_mse_threshold):
                converged = True
                if verbose:
                    print(f"  Converged at epoch {epoch + 1}")
                break

        # Return final metrics
        return FinetuneMetrics(
            ocr_loss=avg_ocr_mse,
            confidence_mse=avg_conf_mse,
            num_samples=samples_processed,
            converged=converged
        )

    def _train_step(
        self,
        batch: List[Dict],
        optimizer: optim.Optimizer
    ) -> Tuple[float, float, int]:
        """
        Single training step on a batch.

        Returns:
            Tuple of (ocr_mse_sum, conf_mse_sum, valid_samples)
        """
        self.surrogate.train()
        optimizer.zero_grad()

        total_ocr_mse = 0.0
        total_conf_mse = 0.0
        valid_samples = 0

        # Collect batch data
        homographies = []
        gt_ocr_losses = []
        gt_confidences = []

        for sample in batch:
            bb_result = sample['bb_result']
            gt_text = sample.get('gt_text', bb_result.text)
            transform = sample['transform']

            if bb_result.text is None or gt_text is None:
                continue

            # Compute ground truth OCR loss from black-box result
            # (How different is bb_result.text from gt_text)
            target_tensor = text_to_target_tensor(
                gt_text, OCR_MAX_SLOTS, OCR_ALPHABET, self.device
            )
            bb_text_tensor = text_to_target_tensor(
                bb_result.text, OCR_MAX_SLOTS, OCR_ALPHABET, self.device
            )

            # Convert to one-hot for loss computation
            with torch.no_grad():
                # Compute focal CCE between bb prediction and ground truth
                # This gives us the "ground truth" OCR loss to predict
                gt_ocr_loss = self.ocr_loss_fn(target_tensor, bb_text_tensor.unsqueeze(0))

            homographies.append(transform)
            gt_ocr_losses.append(gt_ocr_loss.item())
            gt_confidences.append(bb_result.confidence)
            valid_samples += 1

        if valid_samples == 0:
            return 0.0, 0.0, 0

        # Get current patch (normalized to [0, 1])
        patch_normalized = torch.sigmoid(self.trainer.patch)  # [3, 256, 512]

        # Repeat patch for batch and move to device
        patch_batch = patch_normalized.unsqueeze(0).repeat(valid_samples, 1, 1, 1).to(self.device)  # [B, 3, 256, 512]

        # Stack homographies and convert to float32
        homography_batch = torch.stack(homographies).to(self.device).float()  # [B, 3, 3]

        # Stack ground truth
        gt_ocr_losses_tensor = torch.tensor(gt_ocr_losses, device=self.device, dtype=torch.float32).unsqueeze(1)  # [B, 1]
        gt_confidences_tensor = torch.tensor(gt_confidences, device=self.device, dtype=torch.float32).unsqueeze(1)  # [B, 1]

        # Forward through surrogate
        pred_ocr_loss, pred_confidence = self.surrogate(patch_batch, homography_batch)

        # Compute MSE losses
        ocr_mse = F.mse_loss(pred_ocr_loss, gt_ocr_losses_tensor)
        conf_mse = F.mse_loss(pred_confidence, gt_confidences_tensor)

        # Combined loss
        total_loss = ocr_mse + conf_mse

        # Backward and optimize
        total_loss.backward()
        optimizer.step()

        return (
            ocr_mse.item() * valid_samples,
            conf_mse.item() * valid_samples,
            valid_samples
        )



# =============================================================================
# Blur Calibration
# =============================================================================

def apply_plate_blur(
    image: torch.Tensor,
    corners: torch.Tensor,
    sigma: float
) -> torch.Tensor:
    """
    Apply Gaussian blur to the license plate region.

    Args:
        image: Image tensor [C, H, W]
        corners: Plate corner coordinates [4, 2]
        sigma: Blur sigma (0 = no blur)

    Returns:
        Blurred image tensor
    """
    if sigma <= 0:
        return image

    # Create a copy
    result = image.clone()
    C, H, W = image.shape

    # Get bounding box of plate
    min_x = int(max(0, corners[:, 0].min().item()))
    max_x = int(min(W, corners[:, 0].max().item()))
    min_y = int(max(0, corners[:, 1].min().item()))
    max_y = int(min(H, corners[:, 1].max().item()))

    if max_x <= min_x or max_y <= min_y:
        return result

    # Extract plate region
    plate_region = image[:, min_y:max_y, min_x:max_x].unsqueeze(0)

    # Apply Gaussian blur
    kernel_size = int(sigma * 6) | 1  # Ensure odd
    kernel_size = max(3, kernel_size)

    blurred_region = kornia.filters.gaussian_blur2d(
        plate_region,
        kernel_size=(kernel_size, kernel_size),
        sigma=(sigma, sigma)
    ).squeeze(0)

    # Replace plate region
    result[:, min_y:max_y, min_x:max_x] = blurred_region

    return result


def calibrate_blur_level(
    black_box: BlackBoxModel,
    dataset_loader,
    ground_truth_texts: Dict[int, str],
    target_correct_rate: float = 0.5,
    sigma_min: float = 0.0,
    sigma_max: float = 20.0,
    sigma_init: Optional[float] = None,
    tolerance: float = 0.10,  # ±10% is acceptable
    max_iterations: int = 20,
    device: str = 'cpu'
) -> Tuple[float, List[Tuple]]:
    """
    Find blur sigma that gives target correct read rate on black-box.

    Uses binary search to find the blur level. Tolerance is ±10% by default.

    Args:
        black_box: Black-box ALPR model
        dataset_loader: DataLoader with images
        ground_truth_texts: Dict mapping image index to ground truth text
        target_correct_rate: Target correct read rate (default 0.5)
        sigma_min: Minimum blur sigma
        sigma_max: Maximum blur sigma
        sigma_init: Optional initial sigma to evaluate first (speeds up binary search)
        tolerance: Acceptable deviation from target rate (default ±10%)
        max_iterations: Maximum binary search iterations
        device: Device for processing

    Returns:
        Tuple of (calibrated_blur_sigma, cached_samples)
        where cached_samples is from the final evaluation to avoid re-querying
    """
    print(
        f"Calibrating blur level for {target_correct_rate:.0%} (±{tolerance:.0%}) correct rate...")

    def evaluate_sigma(sigma: float) -> Tuple[float, List[Tuple]]:
        """Evaluate sigma and return (rate, cached_samples)."""
        correct = 0
        total = 0
        samples = []

        with tqdm(enumerate(dataset_loader), total=len(dataset_loader),
                  desc=f"  Evaluating sigma={sigma:.2f}", leave=False) as pbar:
            for idx, batch in pbar:
                batch = {k: v[0] for k, v in batch.items()}

                orig_image = batch['orig_image']
                prep_image = batch['prep_image']
                corners = batch['new_corners']
                orig_corners = batch['orig_corners']
                transform = batch['transform']

                # Apply blur
                blurred_orig = apply_plate_blur(orig_image, orig_corners, sigma)
                prep_blur_sigma = sigma * (384 / max(orig_image.shape[1], orig_image.shape[2]))
                blurred_prep = apply_plate_blur(prep_image, corners, prep_blur_sigma)

                # Query black-box
                results = black_box.evaluate([blurred_orig])
                bb_result = results[0] if results else ALPRResult(text=None, confidence=0.0)

                # Cache sample
                samples.append((
                    blurred_prep,
                    blurred_orig,
                    corners,
                    orig_corners,
                    transform,
                    bb_result
                ))

                # Track success rate only for indices in ground truth
                if idx in ground_truth_texts:
                    if bb_result.text == ground_truth_texts[idx]:
                        correct += 1
                    total += 1

                    rate = correct / total if total > 0 else 0.0
                    pbar.set_postfix({'rate': f'{rate:.1%}'})

        return (correct / total if total > 0 else 0.0), samples

    # Binary search
    low, high = sigma_min, sigma_max
    final_samples = []
    iteration = 0

    # If sigma_init provided, evaluate it first to narrow search bounds
    if sigma_init is not None:
        print(f"  Evaluating initial suggestion sigma={sigma_init:.2f}...")
        rate, samples = evaluate_sigma(sigma_init)
        final_samples = samples
        print(f"  Initial: sigma={sigma_init:.2f}, correct_rate={rate:.2%}")

        # Accept if within tolerance
        if abs(rate - target_correct_rate) <= tolerance:
            print(f"  ✓ Converged at sigma={sigma_init:.2f} (rate={rate:.1%})")
            return sigma_init, final_samples

        # Use the initial evaluation to narrow search bounds
        if rate > target_correct_rate:
            # Need more blur, so search above sigma_init
            low = sigma_init
        else:
            # Need less blur, so search below sigma_init
            high = sigma_init

    for iteration in range(max_iterations):
        mid = (low + high) / 2
        rate, samples = evaluate_sigma(mid)
        final_samples = samples  # Cache the latest evaluation

        iter_num = (iteration + 2) if sigma_init is not None else (iteration + 1)
        print(
            f"  Iteration {iter_num}/{max_iterations}: sigma={mid:.2f}, correct_rate={rate:.2%}")

        # Accept if within tolerance (±10%)
        if abs(rate - target_correct_rate) <= tolerance:
            print(f"  ✓ Converged at sigma={mid:.2f} (rate={rate:.1%})")
            return mid, final_samples

        if rate > target_correct_rate:
            # Need more blur
            low = mid
        else:
            # Need less blur
            high = mid

    final_sigma = (low + high) / 2
    print(f"  ⚠ Max iterations reached, using sigma={final_sigma:.2f}")
    return final_sigma, final_samples


# =============================================================================
# Main Optimization Loop
# =============================================================================

def collect_dataset_with_blur(
    black_box: BlackBoxModel,
    trainer: AdversarialPatchTrainer,
    blur_sigma: float,
    ground_truth_texts: Dict[int, str]
) -> List[Tuple]:
    """
    Collect samples with blur applied and black-box results.

    Returns:
        List of (prep_image, orig_image, corners, orig_corners, transform, bb_result)
    """
    samples = []

    with tqdm(enumerate(trainer.train_loader), total=len(trainer.train_loader),
              desc="Collecting clean samples", unit="image") as pbar:
        for idx, batch in pbar:
            batch = {k: v[0] for k, v in batch.items()}

            orig_image = batch['orig_image']
            prep_image = batch['prep_image']
            corners = batch['new_corners']
            orig_corners = batch['orig_corners']
            transform = batch['transform']

            # Apply blur to original image
            if blur_sigma > 0:
                blurred_orig = apply_plate_blur(orig_image, orig_corners, blur_sigma)
                # Also blur prep image proportionally
                prep_blur_sigma = blur_sigma * (384 / max(orig_image.shape[1], orig_image.shape[2]))
                blurred_prep = apply_plate_blur(prep_image, corners, prep_blur_sigma)
            else:
                blurred_orig = orig_image
                blurred_prep = prep_image

            # Query black-box
            results = black_box.evaluate([blurred_orig])
            bb_result = results[0] if results else ALPRResult(text=None, confidence=0.0)

            samples.append((
                blurred_prep,
                blurred_orig,
                corners,
                orig_corners,
                transform,
                bb_result
            ))

    return samples


def collect_patched_samples(
    black_box: BlackBoxModel,
    trainer: AdversarialPatchTrainer,
    ground_truth_texts: Dict[int, str],
    blur_sigma: float = 0.0
) -> Tuple[List[Tuple], float]:
    """
    Apply current patch to all images, apply blur, and collect black-box results.

    Args:
        black_box: Black-box ALPR model
        trainer: Patch trainer instance
        ground_truth_texts: Dict mapping dataset index to ground truth plate text
        blur_sigma: Blur sigma to apply after patching (simulates real-world degradation)

    Returns:
        Tuple of (samples, success_rate) where success_rate is the rate
        at which black-box correctly reads plates (compared to ground truth).
        Samples are now lightweight metadata dicts, not full images.
    """
    samples = []
    correct = 0
    total = 0

    with torch.no_grad():
        with tqdm(enumerate(trainer.train_loader), total=len(trainer.train_loader),
                  desc="Collecting patched samples", unit="image") as pbar:
            for idx, batch in pbar:
                batch = {k: v[0] for k, v in batch.items()}

                corners = batch['new_corners']
                orig_corners = batch['orig_corners']
                transform = batch['transform']

                # Apply ORIGINAL patch (without adapter) for black-box query
                patched_prep, _ = trainer.apply_patch_to_image(
                    batch['prep_image'].to(trainer.device).unsqueeze(0),
                    batch['new_corners'].to(trainer.device).unsqueeze(0)
                )
                patched_orig, _ = trainer.apply_patch_to_image(
                    batch['orig_image'].to(trainer.device).unsqueeze(0),
                    batch['orig_corners'].to(trainer.device).unsqueeze(0)
                )

                patched_prep = patched_prep.squeeze(0).cpu()
                patched_orig = patched_orig.squeeze(0).cpu()

                # Apply blur after patching (simulates real-world degradation)
                if blur_sigma > 0:
                    patched_orig = apply_plate_blur(patched_orig, orig_corners, blur_sigma)
                    # Scale blur for preprocessed image
                    prep_blur_sigma = blur_sigma * (384 / max(batch['orig_image'].shape[1], batch['orig_image'].shape[2]))
                    patched_prep = apply_plate_blur(patched_prep, corners, prep_blur_sigma)

                # Query black-box with ORIGINAL patch (pass corners for IoU-based selection)
                results = black_box.evaluate([patched_orig], corners=[orig_corners])
                bb_result = results[0] if results else ALPRResult(text=None, confidence=0.0)

                # Discard images, store only lightweight metadata
                del patched_prep, patched_orig

                # Get ground truth text for this sample
                gt_text = ground_truth_texts.get(idx, None)

                samples.append({
                    'dataset_idx': idx,
                    'corners': corners.cpu(),
                    'orig_corners': orig_corners.cpu(),
                    'transform': transform.cpu(),
                    'bb_result': bb_result,
                    'gt_text': gt_text
                })

                # Track success rate against ground truth
                if idx in ground_truth_texts:
                    total += 1
                    if bb_result.text == ground_truth_texts[idx]:
                        correct += 1

                    success_rate = correct / total if total > 0 else 0.0
                    pbar.set_postfix({'success_rate': f'{success_rate:.1%}'})

    success_rate = correct / total if total > 0 else 0.0
    return samples, success_rate


def optimize_patch_bb(
    black_box: BlackBoxModel,
    csv_path: str,
    ground_truth_texts: Dict[int, str],
    num_epochs: int = 400,
    device: str = None,
    learning_rate: float = 0.1,
    blur_target_rate: float = 0.5,
    blur_sigma_init: Optional[float] = None,
    ocr_loss_threshold: float = 0.2,
    confidence_mse_threshold: float = 0.1,
    patch_epochs_per_cycle: int = 1,
    save_interval: int = 10,
    verbose: bool = True,
    **trainer_kwargs
) -> Dict[str, Any]:
    """
    Main entry point for black-box adversarial patch optimization.

    Args:
        black_box: BlackBoxModel instance
        csv_path: Path to dataset CSV
        ground_truth_texts: Dict mapping dataset index to ground truth plate text
        num_epochs: Number of optimization epochs (default: 400)
        device: Target device
        learning_rate: Learning rate for patch optimization
        blur_target_rate: Target correct rate for blur calibration
        blur_sigma_init: Optional initial blur sigma suggestion (speeds up calibration)
        ocr_loss_threshold: Maximum average OCR loss for surrogate convergence
        confidence_mse_threshold: Confidence MSE threshold for surrogate convergence
        patch_epochs_per_cycle: Patch optimization epochs before surrogate training (default: 1)
        save_interval: Save patch every N epochs
        verbose: Whether to print progress
        **trainer_kwargs: Additional kwargs for AdversarialPatchTrainer

    Returns:
        Dict with training history and results
    """
    # Device setup
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

    print(f"Using device: {device}")

    # Create shared models
    models = ALPRModels(device=device).load()

    # Create trainer with gradient accumulation for more steps per epoch
    config = TrainerConfig(
        grad_accumulate=64,  # Accumulate gradients over 64 steps per epoch
        use_tv_loss=True,
        use_homography=True,
        **{k: v for k, v in trainer_kwargs.items() if hasattr(TrainerConfig, k)}
    )

    trainer = AdversarialPatchTrainer(
        csv_path=csv_path,
        models=models,
        device=device,
        config=config
    )

    # Create surrogate trainer (query-based, no ALPR models needed)
    surrogate_trainer = SurrogateTrainer(
        trainer=trainer,
        device=device,
        ocr_loss_threshold=ocr_loss_threshold,
        confidence_mse_threshold=confidence_mse_threshold,
        max_epochs=100
    )

    # Create replay buffer (uses exponential decay to weight patch frequency)
    replay_buffer = ReplayBuffer(decay_half_life=4)

    # =========================================================================
    # Phase 1: Initial setup and blur calibration
    # =========================================================================
    print("\n" + "=" * 60)
    print("Phase 1: Blur Calibration")
    print("=" * 60)

    # Calibrate blur level (no initial fine-tuning - adapter trains on patch samples)
    blur_sigma, _ = calibrate_blur_level(
        black_box=black_box,
        dataset_loader=trainer.train_loader,
        ground_truth_texts=ground_truth_texts,
        target_correct_rate=blur_target_rate,
        sigma_init=blur_sigma_init,
        device=device
    )

    print(f"\nCalibrated blur sigma: {blur_sigma:.2f}")

    # Create surrogate optimizer (reuse across cycles)
    surrogate_optimizer = optim.Adam(
        surrogate_trainer.surrogate.parameters(),
        lr=surrogate_trainer.learning_rate
    )

    print("\n" + "=" * 60)
    print("Phase 2: Adversarial Patch Optimization")
    print("=" * 60)

    history = {
        'epoch': [],
        # Patch optimization losses
        'patch_loss_total': [],
        'patch_loss_det': [],
        'patch_loss_ocr': [],
        'patch_loss_tv': [],
        # Surrogate fine-tuning metrics
        'surrogate_ocr_loss': [],
        'surrogate_conf_mse': [],
        'surrogate_converged': [],
        # Black-box evaluation
        'bb_success_rate': [],
        # Training state
        'blur_sigma': [],
    }

    # Create optimizer for patch
    patch_optimizer = optim.Adam([trainer.patch], lr=learning_rate)

    # Threshold for reducing blur (when attack is too effective)
    blur_reduction_threshold = 0.25  # 25% success rate
    current_blur_sigma = blur_sigma

    # Calculate number of cycles based on patch_epochs_per_cycle ratio
    num_cycles = num_epochs // patch_epochs_per_cycle

    total_patch_epoch = 0
    initial_ocr_loss = None  # Saved from first fine-tune for adaptive LR scaling

    for cycle in range(num_cycles):
        print(f"\n{'=' * 60}")
        print(f"Cycle {cycle + 1}/{num_cycles} (Epochs {total_patch_epoch + 1}-{total_patch_epoch + patch_epochs_per_cycle})")
        print('=' * 60)

        # Train patch for 4 epochs before adapter fine-tuning
        cycle_samples = []
        for patch_epoch_in_cycle in range(patch_epochs_per_cycle):
            epoch = total_patch_epoch + patch_epoch_in_cycle
            print(f"\n--- Patch Epoch {epoch + 1}/{num_epochs} ---")

            # Step 1: Train patch for one epoch
            patch_losses = trainer.train_single_epoch(epoch, optimizer=patch_optimizer)
            print(f"Patch loss: {patch_losses['total']:.4f} "
                  f"(det={patch_losses['det']:.4f}, ocr={patch_losses['ocr']:.4f}, tv={patch_losses['tv']:.4f})")

            # Step 2: Apply patch to all images (with blur) and query black-box
            print("Collecting patched samples...")
            patched_samples, bb_success_rate = collect_patched_samples(
                black_box=black_box,
                trainer=trainer,
                ground_truth_texts=ground_truth_texts,
                blur_sigma=current_blur_sigma
            )
            print(f"Black-box success rate: {bb_success_rate:.1%}")

            # Step 2b: Check if we need to reduce blur (attack too effective)
            if bb_success_rate < blur_reduction_threshold and current_blur_sigma > 0:
                print(
                    f"\n*** Attack very effective (success rate {bb_success_rate:.1%} < {blur_reduction_threshold:.0%})")
                print("*** Reducing blur to maintain training diversity...")

                # Re-calibrate with a higher target (to reduce blur)
                new_blur_sigma, cached_clean_samples = calibrate_blur_level(
                    black_box=black_box,
                    dataset_loader=trainer.train_loader,
                    ground_truth_texts=ground_truth_texts,
                    target_correct_rate=blur_target_rate,
                    sigma_min=0.0,
                    sigma_max=current_blur_sigma,  # Can only decrease blur
                    tolerance=0.10,
                    max_iterations=5,  # Quick recalibration
                    device=device
                )

                if new_blur_sigma < current_blur_sigma:
                    current_blur_sigma = new_blur_sigma
                    trainer.set_blur_sigma(current_blur_sigma)
                    print(f"*** New blur sigma: {current_blur_sigma:.2f}")

            # Step 3: Update replay buffer
            replay_buffer.add_patch_epoch(
                epoch=epoch,
                patch=trainer.patch.detach().clone(),
                samples=patched_samples
            )
            cycle_samples.extend(patched_samples)

            # Record history
            history['epoch'].append(epoch + 1)
            history['patch_loss_total'].append(patch_losses['total'])
            history['patch_loss_det'].append(patch_losses['det'])
            history['patch_loss_ocr'].append(patch_losses['ocr'])
            history['patch_loss_tv'].append(patch_losses['tv'])
            history['bb_success_rate'].append(bb_success_rate)
            history['blur_sigma'].append(current_blur_sigma)

            # Save checkpoint every epoch
            trainer.save_patch(epoch, save_dir="bb_patches")

            # Save models periodically (less frequently to save disk space)
            if (epoch + 1) % save_interval == 0:
                models.save_state(f"bb_patches/models_epoch_{epoch + 1:04d}.pt")

        # Step 4: Train surrogate after patch optimization cycle
        print(f"\n{'*' * 60}")
        print(f"Training surrogate (after {patch_epochs_per_cycle} patch epoch{'s' if patch_epochs_per_cycle != 1 else ''})...")
        print('*' * 60)

        # Scale learning rate for subsequent cycles based on OCR MSE improvement
        if cycle > 0 and initial_ocr_loss is not None:
            # Ratio = current_mse / initial_mse; as MSE improves (< 1), lr scales down
            lr_scale = metrics.ocr_loss / initial_ocr_loss
            scaled_lr = surrogate_trainer.learning_rate * lr_scale
            for param_group in surrogate_optimizer.param_groups:
                param_group['lr'] = scaled_lr
            if verbose:
                print(f"Scaled learning rate: {scaled_lr:.2e} (ratio: {lr_scale:.3f})")

        # Create fresh scheduler for this cycle (prevents LR oscillation from scheduler state persistence)
        surrogate_scheduler = LinearWarmupCosineAnnealingLR(
            surrogate_optimizer,
            warmup_epochs=10,
            max_epochs=surrogate_trainer.max_epochs,
            min_lr=1e-6
        )

        training_samples = replay_buffer.get_training_samples()
        metrics = surrogate_trainer.fine_tune(
            training_samples,
            batch_size=100,
            verbose=verbose,
            optimizer=surrogate_optimizer,
            scheduler=surrogate_scheduler
        )

        print(f"Surrogate metrics:")
        print(f"  OCR MSE: {metrics.ocr_loss:.4f}")
        print(f"  Confidence MSE: {metrics.confidence_mse:.4f}")
        print(f"  Converged: {metrics.converged}")
        print(f"  Surrogate LR: {surrogate_optimizer.param_groups[0]['lr']:.2e}")

        # Save initial OCR MSE from first cycle for adaptive LR scaling
        if cycle == 0:
            initial_ocr_loss = metrics.ocr_loss
            if verbose:
                print(f"Saved initial OCR MSE: {initial_ocr_loss:.4f}")

        # Save surrogate model after fine-tuning
        final_patch_epoch = total_patch_epoch + patch_epochs_per_cycle - 1
        torch.save(surrogate_trainer.surrogate.state_dict(),
                   f"bb_patches/surrogate_epoch_{final_patch_epoch + 1:04d}.pt")

        # Record adapter metrics for all patch epochs in this cycle
        for _ in range(patch_epochs_per_cycle):
            history['surrogate_ocr_loss'].append(metrics.ocr_loss)
            history['surrogate_conf_mse'].append(metrics.confidence_mse)
            history['surrogate_converged'].append(metrics.converged)

        total_patch_epoch += patch_epochs_per_cycle

    # Final save
    trainer.save_patch(num_epochs - 1, save_dir="bb_patches_final")
    models.save_state("bb_patches_final/models_final.pt")
    torch.save(surrogate_trainer.surrogate.state_dict(), "bb_patches_final/surrogate_final.pt")

    print("\n" + "=" * 60)
    print("Optimization Complete")
    print("=" * 60)
    print(f"Final blur sigma: {current_blur_sigma:.2f}")
    print(f"Final black-box success rate: {history['bb_success_rate'][-1]:.1%}")

    return {
        'history': history,
        'final_patch': trainer.patch.detach().cpu(),
        'blur_sigma': current_blur_sigma,
        'initial_blur_sigma': blur_sigma,
    }


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Example: Dummy black-box for testing
    class DummyBlackBox(BlackBoxModel):
        """Dummy black-box that always returns the same result."""

        def evaluate(self, images: List[torch.Tensor]) -> List[ALPRResult]:
            return [ALPRResult(text="ABC1234", confidence=0.95) for _ in images]

    # This would be replaced with actual ground truth
    ground_truth = {i: "ABC1234" for i in range(1000)}

    print("This is a demonstration. To use:")
    print("1. Extend BlackBoxModel with your ALPR API")
    print("2. Call optimize_patch_bb() with your model and dataset")
    print()
    print("Example:")
    print("  from model_extraction import BlackBoxModel, ALPRResult, optimize_patch_bb")
    print()
    print("  class MyALPR(BlackBoxModel):")
    print("      def evaluate(self, images):")
    print("          # Call your API here")
    print("          return [ALPRResult(text='ABC123', confidence=0.9)]")
    print()
    print("  history = optimize_patch_bb(")
    print("      black_box=MyALPR(),")
    print("      csv_path='preproc_labels.csv',")
    print("      ground_truth_texts={0: 'ABC123', 1: 'XYZ789', ...}")
    print("  )")
