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
    def evaluate(self, images: List[torch.Tensor]) -> List[ALPRResult]:
        """
        Evaluate the black-box ALPR on a batch of images.

        Args:
            images: List of image tensors [C, H, W] in [0, 1] range, RGB format

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
    Manages fine-tuning dataset with structured sampling.

    After initial epochs, maintains:
    - 25% clean (unpatched) images
    - 25% images with the most recent patch
    - 50% historical patches with exponential decay favoring recency
    """

    def __init__(
        self,
        original_dataset_size: int,
        decay_half_life: int = 4,
        initial_epochs: int = 3
    ):
        """
        Args:
            original_dataset_size: Size of the original training set
            decay_half_life: Half-life for exponential decay (in epochs)
            initial_epochs: Number of epochs before applying sampling strategy
        """
        self.original_size = original_dataset_size
        self.max_size = 4 * original_dataset_size
        self.decay_half_life = decay_half_life
        self.initial_epochs = initial_epochs
        self.decay_rate = math.log(2) / decay_half_life

        # Storage: each entry is (image_tensor, corners, transform, bb_result)
        self.clean_samples: List[Tuple] = []
        self.last_patch_samples: List[Tuple] = []
        # Historical: list of (epoch, patch_tensor, samples)
        self.historical: List[Tuple[int, torch.Tensor, List[Tuple]]] = []

        self.current_epoch = 0

    def add_clean_samples(self, samples: List[Tuple]):
        """Add clean (unpatched) samples with their black-box results."""
        self.clean_samples = samples

    def add_patch_epoch(
        self,
        epoch: int,
        patch: torch.Tensor,
        samples: List[Tuple]
    ):
        """
        Add samples from a patch optimization epoch.

        Args:
            epoch: Current epoch number
            patch: The patch tensor used
            samples: List of (image, corners, transform, bb_result)
        """
        self.current_epoch = epoch

        # Move previous "last patch" to historical
        if self.last_patch_samples:
            prev_epoch = epoch - 1
            # Find the patch from previous epoch if it exists
            prev_patch = None
            if self.historical:
                prev_patch = self.historical[-1][1]
            if prev_patch is not None:
                self.historical.append((prev_epoch, prev_patch, self.last_patch_samples))

        # Update last patch samples
        self.last_patch_samples = samples

        # Store current patch for when it becomes historical
        if not self.historical or epoch > 0:
            # We'll add the patch reference when it moves to historical
            pass

    def get_training_samples(self) -> List[Tuple]:
        """
        Get samples for fine-tuning according to the sampling strategy.

        Returns:
            List of (image, corners, transform, bb_result) tuples
        """
        if self.current_epoch < self.initial_epochs:
            # Initial epochs: use all available samples
            all_samples = list(self.clean_samples)
            all_samples.extend(self.last_patch_samples)
            for _, _, samples in self.historical:
                all_samples.extend(samples)
            return all_samples

        # After initial epochs: apply 25/25/50 split
        target_size = min(self.max_size, len(self.clean_samples) * 4)

        clean_count = target_size // 4
        last_patch_count = target_size // 4
        historical_count = target_size - clean_count - last_patch_count

        result = []

        # 25% clean samples
        if self.clean_samples:
            clean_indices = self._sample_indices(len(self.clean_samples), clean_count)
            result.extend([self.clean_samples[i] for i in clean_indices])

        # 25% last patch samples
        if self.last_patch_samples:
            last_indices = self._sample_indices(len(self.last_patch_samples), last_patch_count)
            result.extend([self.last_patch_samples[i] for i in last_indices])

        # 50% historical with exponential decay
        if self.historical and historical_count > 0:
            historical_samples = self._sample_historical(historical_count)
            result.extend(historical_samples)

        return result

    def _sample_indices(self, pool_size: int, sample_count: int) -> List[int]:
        """Sample indices with replacement if needed."""
        if pool_size == 0:
            return []
        if sample_count <= pool_size:
            return random.sample(range(pool_size), sample_count)
        else:
            # Sample with replacement
            return [random.randint(0, pool_size - 1) for _ in range(sample_count)]

    def _sample_historical(self, count: int) -> List[Tuple]:
        """Sample from historical patches with exponential decay weights."""
        if not self.historical:
            return []

        # Calculate weights based on recency
        weights = []
        all_samples_with_epoch = []

        for epoch, patch, samples in self.historical:
            age = self.current_epoch - epoch
            weight = math.exp(-self.decay_rate * age)
            for sample in samples:
                weights.append(weight)
                all_samples_with_epoch.append(sample)

        if not all_samples_with_epoch:
            return []

        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            weights = [1.0] * len(weights)
            total_weight = len(weights)
        probs = [w / total_weight for w in weights]

        # Sample according to weights
        indices = np.random.choice(
            len(all_samples_with_epoch),
            size=min(count, len(all_samples_with_epoch)),
            replace=True,
            p=probs
        )

        return [all_samples_with_epoch[i] for i in indices]

    def __len__(self) -> int:
        total = len(self.clean_samples) + len(self.last_patch_samples)
        for _, _, samples in self.historical:
            total += len(samples)
        return total


# =============================================================================
# Surrogate Trainer
# =============================================================================

class SurrogateTrainer:
    """
    Fine-tunes the fast-alpr surrogate models to match black-box behavior.

    Only trains:
    - Detection confidence (not bounding boxes)
    - OCR text output
    """

    def __init__(
        self,
        models: ALPRModels,
        device: str,
        ocr_loss_threshold: float = 0.1,
        confidence_mse_threshold: float = 0.1,
        learning_rate: float = 1e-4,
        max_epochs: int = 100
    ):
        """
        Args:
            models: ALPRModels instance to fine-tune
            device: Target device
            ocr_loss_threshold: Maximum average OCR loss for convergence
            confidence_mse_threshold: Maximum confidence MSE for convergence
            learning_rate: Learning rate for fine-tuning
            max_epochs: Maximum training epochs before giving up
        """
        self.models = models
        self.device = device
        self.ocr_loss_threshold = ocr_loss_threshold
        self.confidence_mse_threshold = confidence_mse_threshold
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

        self.ocr_loss_fn = create_focal_cce_loss(len(OCR_ALPHABET))

    def fine_tune(
        self,
        samples: List[Tuple],
        batch_size: int = 64,
        verbose: bool = True
    ) -> FinetuneMetrics:
        """
        Fine-tune surrogate models until convergence or max epochs.

        Each epoch processes the entire dataset with batched gradient updates.
        Convergence is checked based on metrics over the whole dataset after each epoch.

        Args:
            samples: List of (prep_image, orig_image, corners, orig_corners, transform, bb_result)
            batch_size: Training batch size for gradient updates
            verbose: Whether to show progress bar

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

        # Unfreeze models for training
        self.models.unfreeze_all()

        # Set up optimizers
        ocr_optimizer = optim.Adam(self.models.get_ocr_parameters(), lr=self.learning_rate)
        detector_optimizer = optim.Adam(
            self.models.get_detector_parameters(),
            lr=self.learning_rate * 0.1  # Lower LR for detector
        )

        # Learning rate schedulers for faster convergence
        ocr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            ocr_optimizer, mode='min', factor=0.5, patience=3, verbose=False
        )
        detector_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            detector_optimizer, mode='min', factor=0.5, patience=3, verbose=False
        )

        converged = False
        best_ocr_loss = float('inf')

        for epoch in range(self.max_epochs):
            # Shuffle samples at start of each epoch
            random.shuffle(samples)

            # Training pass over entire dataset
            epoch_ocr_loss = 0.0
            epoch_conf_loss = 0.0
            samples_processed = 0

            desc = f"Fine-tune epoch {epoch + 1}/{self.max_epochs}"
            pbar = tqdm(range(0, len(samples), batch_size), desc=desc,
                        disable=not verbose, leave=False)

            for i in pbar:
                batch = samples[i:i + batch_size]
                ocr_loss, conf_loss, _, _, valid_samples = self._train_step(
                    batch, ocr_optimizer, detector_optimizer
                )
                epoch_ocr_loss += ocr_loss
                epoch_conf_loss += conf_loss
                samples_processed += valid_samples

                pbar.set_postfix({
                    'ocr_loss': f'{epoch_ocr_loss / max(1, samples_processed):.4f}',
                    'conf_loss': f'{epoch_conf_loss / max(1, samples_processed):.4f}'
                })

            pbar.close()

            # Evaluate on entire dataset after epoch
            metrics = self._evaluate(samples)

            if verbose:
                ocr_lr = ocr_optimizer.param_groups[0]['lr']
                det_lr = detector_optimizer.param_groups[0]['lr']
                print(f"  Epoch {epoch + 1}: ocr_loss={metrics.ocr_loss:.4f}, "
                      f"conf_mse={metrics.confidence_mse:.4f} | "
                      f"LR: ocr={ocr_lr:.2e}, det={det_lr:.2e}")

            # Step schedulers to reduce LR if loss plateaus
            ocr_scheduler.step(metrics.ocr_loss)
            detector_scheduler.step(metrics.confidence_mse)

            # Check convergence based on whole-dataset metrics
            if (metrics.ocr_loss <= self.ocr_loss_threshold and
                    metrics.confidence_mse <= self.confidence_mse_threshold):
                converged = True
                if verbose:
                    print(f"  Converged at epoch {epoch + 1}")
                break

        # Freeze models again
        self.models.freeze_all()

        # Final evaluation
        final_metrics = self._evaluate(samples)
        final_metrics.converged = converged

        return final_metrics

    def _train_step(
        self,
        batch: List[Tuple],
        ocr_optimizer: optim.Optimizer,
        detector_optimizer: optim.Optimizer
    ) -> Tuple[float, float, float, float, int]:
        """Single training step on a batch.

        Returns:
            Tuple of (ocr_loss_sum, conf_loss_sum, match_rate, conf_mse, valid_samples)
            where valid_samples is the count of samples that produced valid losses
        """
        self.models.detector.train()
        self.models.ocr.train()

        total_ocr_loss = 0.0
        total_conf_loss = 0.0
        correct_matches = 0
        valid_samples = 0
        conf_squared_errors = []

        ocr_optimizer.zero_grad()
        detector_optimizer.zero_grad()

        for sample in batch:
            prep_image, orig_image, corners, orig_corners, transform, bb_result = sample

            if bb_result.text is None:
                continue

            # Run detector
            prep_image = prep_image.to(self.device)
            corners = corners.to(self.device)
            detector_output = self.models.detector(prep_image.unsqueeze(0))

            if len(detector_output) == 0:
                continue

            # Find best detection (highest IoU with target plate region)
            target_box = corners_to_bbox(corners)
            best_det = None
            best_iou = -1.0

            for detection in detector_output:
                pred_box = detection[1:5]
                iou = compute_iou(pred_box.unsqueeze(0), target_box.unsqueeze(0)).item()
                if iou > best_iou:
                    best_iou = iou
                    best_det = detection

            if best_det is None:
                continue

            valid_samples += 1
            pred_conf = best_det[6]

            # Confidence loss
            target_conf = torch.tensor(bb_result.confidence, device=self.device)
            conf_loss = F.mse_loss(pred_conf, target_conf)
            total_conf_loss += conf_loss

            conf_squared_errors.append((pred_conf.item() - bb_result.confidence) ** 2)

            # Get plate crop for OCR
            pred_box = best_det[1:5]
            orig_projection = invert_bbox(pred_box.to('cpu'), transform)
            corners_box = bbox_to_corners(orig_projection, device='cpu')

            cropped_plate = kornia.geometry.crop_and_resize(
                orig_image.unsqueeze(0),
                corners_box,
                OCR_INPUT_SHAPE[:2],
                mode='bilinear',
                align_corners=True
            ).to(self.device)

            # OCR forward
            ocr_input = cropped_plate.permute(0, 2, 3, 1) * 255
            ocr_output = self.models.ocr(ocr_input)

            # OCR loss against black-box text
            target_tensor = text_to_target_tensor(
                bb_result.text, OCR_MAX_SLOTS, OCR_ALPHABET, self.device
            )
            ocr_loss = self.ocr_loss_fn(target_tensor, ocr_output)
            total_ocr_loss += ocr_loss

            # Check exact match
            pred_text = logits_to_text(ocr_output)
            if pred_text == bb_result.text:
                correct_matches += 1

        # Backward passes
        if total_ocr_loss > 0:
            total_ocr_loss.backward(retain_graph=True)
            ocr_optimizer.step()

        if total_conf_loss > 0:
            total_conf_loss.backward()
            detector_optimizer.step()

        self.models.detector.eval()
        self.models.ocr.eval()

        n = len(batch)
        match_rate = correct_matches / n if n > 0 else 0.0
        conf_mse = np.mean(conf_squared_errors) if conf_squared_errors else float('inf')

        return (
            total_ocr_loss.item() if isinstance(total_ocr_loss, torch.Tensor) else 0.0,
            total_conf_loss.item() if isinstance(total_conf_loss, torch.Tensor) else 0.0,
            match_rate,
            conf_mse,
            valid_samples
        )

    def _evaluate(self, samples: List[Tuple]) -> FinetuneMetrics:
        """Evaluate current model on all samples."""
        ocr_losses = []
        conf_squared_errors = []
        valid_samples = 0

        with torch.no_grad():
            for sample in samples:
                prep_image, orig_image, corners, orig_corners, transform, bb_result = sample

                if bb_result.text is None:
                    continue

                valid_samples += 1

                # Run detector
                prep_image = prep_image.to(self.device)
                corners = corners.to(self.device)
                detector_output = self.models.detector(prep_image.unsqueeze(0))

                if len(detector_output) == 0:
                    conf_squared_errors.append(bb_result.confidence ** 2)
                    continue

                # Find best detection (highest IoU with target plate region)
                target_box = corners_to_bbox(corners)
                best_det = None
                best_iou = -1.0

                for detection in detector_output:
                    pred_box = detection[1:5]
                    iou = compute_iou(pred_box.unsqueeze(0), target_box.unsqueeze(0)).item()
                    if iou > best_iou:
                        best_iou = iou
                        best_det = detection

                if best_det is None:
                    conf_squared_errors.append(bb_result.confidence ** 2)
                    continue

                pred_conf = best_det[6].item()

                conf_squared_errors.append((pred_conf - bb_result.confidence) ** 2)

                # Get OCR prediction
                pred_box = best_det[1:5]
                orig_projection = invert_bbox(pred_box.to('cpu'), transform)
                corners_box = bbox_to_corners(orig_projection, device='cpu')

                cropped_plate = kornia.geometry.crop_and_resize(
                    orig_image.unsqueeze(0),
                    corners_box,
                    OCR_INPUT_SHAPE[:2],
                    mode='bilinear',
                    align_corners=True
                ).to(self.device)

                ocr_input = cropped_plate.permute(0, 2, 3, 1) * 255
                ocr_output = self.models.ocr(ocr_input)

                # Compute OCR loss against black-box text
                target_tensor = text_to_target_tensor(
                    bb_result.text, OCR_MAX_SLOTS, OCR_ALPHABET, self.device
                )
                ocr_loss = self.ocr_loss_fn(target_tensor, ocr_output)
                ocr_losses.append(ocr_loss.item())

        avg_ocr_loss = np.mean(ocr_losses) if ocr_losses else float('inf')
        conf_mse = np.mean(conf_squared_errors) if conf_squared_errors else float('inf')

        return FinetuneMetrics(
            ocr_loss=avg_ocr_loss,
            confidence_mse=conf_mse,
            num_samples=valid_samples,
            converged=False
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
    tolerance: float = 0.10,  # ±10% is acceptable
    max_iterations: int = 20,
    device: str = 'cpu'
) -> float:
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
        tolerance: Acceptable deviation from target rate (default ±10%)
        max_iterations: Maximum binary search iterations
        device: Device for processing

    Returns:
        Calibrated blur sigma
    """
    print(
        f"Calibrating blur level for {target_correct_rate:.0%} (±{tolerance:.0%}) correct rate...")

    def evaluate_sigma(sigma: float) -> float:
        """Get correct rate at given sigma."""
        correct = 0
        total = 0

        with tqdm(enumerate(dataset_loader), total=len(dataset_loader),
                  desc=f"  Evaluating sigma={sigma:.2f}", leave=False) as pbar:
            for idx, batch in pbar:
                batch = {k: v[0] for k, v in batch.items()}

                if idx not in ground_truth_texts:
                    continue

                gt_text = ground_truth_texts[idx]
                orig_image = batch['orig_image']
                corners = batch['orig_corners']

                # Apply blur
                blurred = apply_plate_blur(orig_image, corners, sigma)

                # Query black-box
                results = black_box.evaluate([blurred])

                if results and results[0].text == gt_text:
                    correct += 1
                total += 1

                rate = correct / total if total > 0 else 0.0
                pbar.set_postfix({'rate': f'{rate:.1%}'})

        return correct / total if total > 0 else 0.0

    # Binary search
    low, high = sigma_min, sigma_max

    for iteration in range(max_iterations):
        mid = (low + high) / 2
        rate = evaluate_sigma(mid)

        print(
            f"  Iteration {iteration + 1}/{max_iterations}: sigma={mid:.2f}, correct_rate={rate:.2%}")

        # Accept if within tolerance (±10%)
        if abs(rate - target_correct_rate) <= tolerance:
            print(f"  ✓ Converged at sigma={mid:.2f} (rate={rate:.1%})")
            return mid

        if rate > target_correct_rate:
            # Need more blur
            low = mid
        else:
            # Need less blur
            high = mid

    final_sigma = (low + high) / 2
    print(f"  ⚠ Max iterations reached, using sigma={final_sigma:.2f}")
    return final_sigma


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
        at which black-box correctly reads plates (compared to ground truth)
    """
    samples = []
    correct = 0
    total = 0

    with torch.no_grad():
        with tqdm(enumerate(trainer.train_loader), total=len(trainer.train_loader),
                  desc="Collecting patched samples", unit="image") as pbar:
            for idx, batch in pbar:
                batch = {k: v[0] for k, v in batch.items()}

                orig_image = batch['orig_image'].to(trainer.device)
                prep_image = batch['prep_image'].to(trainer.device)
                corners = batch['new_corners'].to(trainer.device)
                orig_corners = batch['orig_corners'].to(trainer.device)
                transform = batch['transform']

                # Apply patch
                patched_prep, _ = trainer.apply_patch_to_image(
                    prep_image.unsqueeze(0),
                    corners.unsqueeze(0)
                )
                patched_orig, _ = trainer.apply_patch_to_image(
                    orig_image.unsqueeze(0),
                    orig_corners.unsqueeze(0)
                )

                patched_prep = patched_prep.squeeze(0).cpu()
                patched_orig = patched_orig.squeeze(0).cpu()

                # Apply blur after patching (simulates real-world degradation)
                if blur_sigma > 0:
                    patched_orig = apply_plate_blur(patched_orig, orig_corners.cpu(), blur_sigma)
                    # Scale blur for preprocessed image
                    prep_blur_sigma = blur_sigma * (384 / max(orig_image.shape[1], orig_image.shape[2]))
                    patched_prep = apply_plate_blur(patched_prep, corners.cpu(), prep_blur_sigma)

                # Query black-box
                results = black_box.evaluate([patched_orig])
                bb_result = results[0] if results else ALPRResult(text=None, confidence=0.0)

                samples.append((
                    patched_prep,
                    patched_orig,
                    corners.cpu(),
                    orig_corners.cpu(),
                    transform,
                    bb_result
                ))

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
    ocr_loss_threshold: float = 0.1,
    confidence_mse_threshold: float = 0.1,
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
        ocr_loss_threshold: Maximum average OCR loss for surrogate convergence
        confidence_mse_threshold: Confidence MSE threshold for surrogate convergence
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

    # Create trainer with single gradient update per epoch
    config = TrainerConfig(
        grad_accumulate=None,  # Single update per epoch
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

    # Create surrogate trainer
    surrogate_trainer = SurrogateTrainer(
        models=models,
        device=device,
        ocr_loss_threshold=ocr_loss_threshold,
        confidence_mse_threshold=confidence_mse_threshold
    )

    # Create replay buffer
    dataset_size = len(trainer.train_loader)
    replay_buffer = ReplayBuffer(
        original_dataset_size=dataset_size,
        decay_half_life=4,
        initial_epochs=3
    )

    # =========================================================================
    # Phase 1: Calibrate blur and initial surrogate extraction
    # =========================================================================
    print("\n" + "=" * 60)
    print("Phase 1: Initial Model Extraction")
    print("=" * 60)

    # Calibrate blur level
    blur_sigma = calibrate_blur_level(
        black_box=black_box,
        dataset_loader=trainer.train_loader,
        ground_truth_texts=ground_truth_texts,
        target_correct_rate=blur_target_rate,
        device=device
    )

    print(f"\nCalibrated blur sigma: {blur_sigma:.2f}")

    # Collect blurred samples with black-box labels
    print("\nCollecting initial blurred samples...")
    clean_samples = collect_dataset_with_blur(
        black_box=black_box,
        trainer=trainer,
        blur_sigma=blur_sigma,
        ground_truth_texts=ground_truth_texts
    )

    replay_buffer.add_clean_samples(clean_samples)

    # Initial fine-tuning
    print("\nInitial surrogate fine-tuning...")
    initial_metrics = surrogate_trainer.fine_tune(clean_samples, verbose=verbose)

    print(f"\nInitial extraction results:")
    print(f"  OCR loss: {initial_metrics.ocr_loss:.4f}")
    print(f"  Confidence MSE: {initial_metrics.confidence_mse:.4f}")
    print(f"  Converged: {initial_metrics.converged}")

    if not initial_metrics.converged:
        print("WARNING: Initial extraction did not converge. Continuing anyway...")

    # =========================================================================
    # Phase 2: Iterative patch optimization
    # =========================================================================
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
    blur_reduction_threshold = 0.20  # 20% success rate
    current_blur_sigma = blur_sigma

    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")

        # Step 1: Train patch for one epoch (single gradient update)
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
            # Target: get back to ~50% success rate zone
            new_blur_sigma = calibrate_blur_level(
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

                # Re-collect clean samples with new blur level
                print("*** Re-collecting clean samples with reduced blur...")
                clean_samples = collect_dataset_with_blur(
                    black_box=black_box,
                    trainer=trainer,
                    blur_sigma=current_blur_sigma,
                    ground_truth_texts=ground_truth_texts
                )
                replay_buffer.add_clean_samples(clean_samples)

                # Fine-tune surrogate on new clean samples
                print("*** Fine-tuning surrogate on less blurred data...")
                surrogate_trainer.fine_tune(clean_samples, verbose=verbose)

        # Step 3: Update replay buffer
        replay_buffer.add_patch_epoch(
            epoch=epoch,
            patch=trainer.patch.detach().clone(),
            samples=patched_samples
        )

        # Step 4: Fine-tune surrogate on replay buffer
        print("Fine-tuning surrogate...")
        training_samples = replay_buffer.get_training_samples()
        metrics = surrogate_trainer.fine_tune(training_samples, verbose=verbose)

        print(f"Surrogate metrics:")
        print(f"  OCR loss: {metrics.ocr_loss:.4f}")
        print(f"  Confidence MSE: {metrics.confidence_mse:.4f}")
        print(f"  Converged: {metrics.converged}")

        # Record history
        history['epoch'].append(epoch + 1)
        history['patch_loss_total'].append(patch_losses['total'])
        history['patch_loss_det'].append(patch_losses['det'])
        history['patch_loss_ocr'].append(patch_losses['ocr'])
        history['patch_loss_tv'].append(patch_losses['tv'])
        history['surrogate_ocr_loss'].append(metrics.ocr_loss)
        history['surrogate_conf_mse'].append(metrics.confidence_mse)
        history['surrogate_converged'].append(metrics.converged)
        history['bb_success_rate'].append(bb_success_rate)
        history['blur_sigma'].append(current_blur_sigma)

        # Save checkpoint every epoch
        trainer.save_patch(epoch, save_dir="bb_patches")

        # Save models periodically (less frequently to save disk space)
        if (epoch + 1) % save_interval == 0:
            models.save_state(f"bb_patches/models_epoch_{epoch + 1:04d}.pt")

    # Final save
    trainer.save_patch(num_epochs - 1, save_dir="bb_patches_final")
    models.save_state("bb_patches_final/models_final.pt")

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
        'initial_metrics': initial_metrics,
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
