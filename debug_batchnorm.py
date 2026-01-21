#!/usr/bin/env python3
"""
Debug script to check if BatchNorm train vs eval mode is causing output collapse.
"""
import torch
from progressive_patch import FoundationPatchGenerator

def debug_batchnorm(
    checkpoint_path: str = "generator_export/final_layer_checkpoint_epoch_0100/generator_epoch_0104.pt",
    num_samples: int = 5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    print(f"Loading checkpoint from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    basis_dim = checkpoint['basis_dim']
    patch_size = checkpoint['patch_size']
    patch_height, patch_width = patch_size

    generator = FoundationPatchGenerator(
        latent_dim=basis_dim,
        patch_height=patch_height,
        patch_width=patch_width
    ).to(device)

    generator.load_state_dict(checkpoint['generator_state_dict'])

    # Check BatchNorm running statistics
    print("\n=== BatchNorm Running Statistics ===")
    for name, module in generator.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            print(f"\n{name}:")
            print(f"  running_mean: min={module.running_mean.min():.6f}, max={module.running_mean.max():.6f}, mean={module.running_mean.mean():.6f}")
            print(f"  running_var:  min={module.running_var.min():.6f}, max={module.running_var.max():.6f}, mean={module.running_var.mean():.6f}")
            print(f"  num_batches_tracked: {module.num_batches_tracked}")

    # Generate same z samples
    torch.manual_seed(42)
    z_samples = torch.rand(num_samples, basis_dim, device=device)

    # Test in EVAL mode
    generator.eval()
    with torch.no_grad():
        patches_eval = generator(z_samples)

    print("\n=== EVAL Mode Output ===")
    print(f"Shape: {patches_eval.shape}")
    print(f"Min: {patches_eval.min():.6f}, Max: {patches_eval.max():.6f}, Mean: {patches_eval.mean():.6f}")
    for i in range(num_samples):
        p = patches_eval[i]
        print(f"  Sample {i}: min={p.min():.6f}, max={p.max():.6f}, std={p.std():.6f}")

    # Test in TRAIN mode
    generator.train()
    with torch.no_grad():
        patches_train = generator(z_samples)

    print("\n=== TRAIN Mode Output ===")
    print(f"Shape: {patches_train.shape}")
    print(f"Min: {patches_train.min():.6f}, Max: {patches_train.max():.6f}, Mean: {patches_train.mean():.6f}")
    for i in range(num_samples):
        p = patches_train[i]
        print(f"  Sample {i}: min={p.min():.6f}, max={p.max():.6f}, std={p.std():.6f}")

    # Compare pairwise differences
    print("\n=== Pairwise L2 Distances ===")
    print(f"{'Pair':<8} {'EVAL Mode':<15} {'TRAIN Mode':<15}")
    print("-" * 38)
    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            l2_eval = torch.norm(patches_eval[i] - patches_eval[j]).item()
            l2_train = torch.norm(patches_train[i] - patches_train[j]).item()
            print(f"{i}-{j}      {l2_eval:<15.6f} {l2_train:<15.6f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="generator_export/final_layer_checkpoint_epoch_0100/generator_epoch_0104.pt")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    debug_batchnorm(args.checkpoint, args.num_samples, args.device)
