#!/usr/bin/env python3
"""
Quick test script for batched _train_step fine-tuning.
Tests the batched detector/OCR inference without requiring full black-box setup.
"""

import torch
import time
from typing import Dict
from model_extraction import SurrogateTrainer, ALPRResult
from optimize_patch import ALPRModels, AdversarialPatchTrainer, TrainerConfig
from torch import optim


def create_dummy_samples(num_samples: int, device: str) -> list:
    """Create dummy training samples for testing."""
    samples = []

    for _ in range(num_samples):
        # Create dummy tensors matching expected shapes
        prep_image = torch.rand(3, 384, 384)  # Preprocessed image
        orig_image = torch.rand(3, 1080, 1920)  # Original image
        corners = torch.tensor([[100, 100], [300, 100], [300, 200], [100, 200]], dtype=torch.float32)
        orig_corners = torch.tensor([[300, 300], [900, 300], [900, 600], [300, 600]], dtype=torch.float32)
        transform = torch.eye(3)

        # Dummy black-box result
        bb_result = ALPRResult(text="ABC1234", confidence=0.95)

        samples.append((prep_image, orig_image, corners, orig_corners, transform, bb_result))

    return samples


def test_batched_train_step(batch_size: int = 32, num_samples: int = 32, device: str = 'cuda'):
    """Test the batched train step."""

    print(f"Testing batched _train_step")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of test samples: {num_samples}")
    print(f"  Device: {device}")
    print()

    # Load models
    print("Loading ALPR models...")
    models = ALPRModels(device=device).load()

    # Create dummy trainer (only needs patch and apply_patch_to_image)
    print("Creating trainer...")
    config = TrainerConfig(use_tv_loss=False, use_homography=False)
    trainer = AdversarialPatchTrainer(
        csv_path="preproc_labels.csv",
        models=models,
        device=device,
        config=config
    )

    # Create surrogate trainer
    print("Creating surrogate trainer...")
    surrogate_trainer = SurrogateTrainer(
        models=models,
        trainer=trainer,
        device=device,
        ocr_loss_threshold=0.2,
        confidence_mse_threshold=0.1,
        max_epochs=5  # Just 5 epochs for testing
    )

    # Create optimizer
    adapter_optimizer = optim.Adam(
        models.get_adapter_parameters(),
        lr=surrogate_trainer.learning_rate
    )

    # Create dummy samples
    print(f"Creating {num_samples} dummy training samples...")
    samples = create_dummy_samples(num_samples, device)

    # Prepare batch
    batch = samples[:batch_size]

    print()
    print("=" * 60)
    print(f"Running batched _train_step with batch_size={len(batch)}")
    print("=" * 60)

    # Warm up
    print("Warming up GPU...")
    with torch.no_grad():
        _ = models.detector(torch.rand(2, 3, 384, 384).to(device))

    # Run train step
    print("Running train step...")
    start_time = time.time()

    ocr_loss, conf_loss, match_rate, conf_mse, valid_samples = surrogate_trainer._train_step(
        batch, adapter_optimizer
    )

    elapsed = time.time() - start_time

    print()
    print("=" * 60)
    print("Results:")
    print("=" * 60)
    print(f"Time elapsed: {elapsed:.2f}s")
    print(f"Valid samples: {valid_samples}/{len(batch)}")
    print(f"OCR loss: {ocr_loss:.4f}")
    print(f"Confidence loss: {conf_loss:.4f}")
    print(f"Match rate: {match_rate:.1%}")
    print(f"Confidence MSE: {conf_mse:.4f}")
    print(f"Time per sample: {elapsed/max(1, valid_samples)*1000:.1f}ms")
    print()


def test_fine_tune_convergence(batch_size: int = 32, num_samples: int = 64, device: str = 'cuda'):
    """Test fine-tune with early stopping."""

    print(f"Testing fine-tune convergence")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of test samples: {num_samples}")
    print(f"  Device: {device}")
    print()

    # Load models
    print("Loading ALPR models...")
    models = ALPRModels(device=device).load()

    # Create dummy trainer
    print("Creating trainer...")
    config = TrainerConfig(use_tv_loss=False, use_homography=False)
    trainer = AdversarialPatchTrainer(
        csv_path="preproc_labels.csv",
        models=models,
        device=device,
        config=config
    )

    # Create surrogate trainer with high loss threshold so it runs full epochs
    print("Creating surrogate trainer...")
    surrogate_trainer = SurrogateTrainer(
        models=models,
        trainer=trainer,
        device=device,
        ocr_loss_threshold=10.0,  # High threshold so it runs all epochs
        confidence_mse_threshold=10.0,
        max_epochs=3  # Just 3 epochs for testing
    )

    # Create optimizer
    adapter_optimizer = optim.Adam(
        models.get_adapter_parameters(),
        lr=surrogate_trainer.learning_rate
    )

    # Create dummy samples
    print(f"Creating {num_samples} dummy training samples...")
    samples = create_dummy_samples(num_samples, device)

    print()
    print("=" * 60)
    print(f"Running fine_tune with {len(samples)} samples, batch_size={batch_size}")
    print("=" * 60)
    print()

    start_time = time.time()

    metrics = surrogate_trainer.fine_tune(
        samples,
        batch_size=batch_size,
        verbose=True,
        adapter_optimizer=adapter_optimizer
    )

    elapsed = time.time() - start_time

    print()
    print("=" * 60)
    print("Fine-tune Results:")
    print("=" * 60)
    print(f"Total time: {elapsed:.2f}s")
    print(f"Final OCR loss: {metrics.ocr_loss:.4f}")
    print(f"Final confidence MSE: {metrics.confidence_mse:.4f}")
    print(f"Converged: {metrics.converged}")
    print(f"Time per epoch: {elapsed/3:.2f}s")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test batched fine-tuning")
    parser.add_argument("--mode", choices=["step", "finetune"], default="step",
                        help="Test mode: 'step' for single train step, 'finetune' for full fine-tune")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--num-samples", type=int, default=32, help="Number of test samples")
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")

    args = parser.parse_args()

    if args.mode == "step":
        test_batched_train_step(args.batch_size, args.num_samples, args.device)
    else:
        test_fine_tune_convergence(args.batch_size, args.num_samples, args.device)
