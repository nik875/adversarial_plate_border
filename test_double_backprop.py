#!/usr/bin/env python3
"""
Test script to isolate where double backprop fails with grid_sampler.
"""
import torch
import torch.nn.functional as F
import kornia
import kornia.geometry as K

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

def test_case(name, test_fn):
    """Run a test case and report if it succeeds or fails"""
    print(f"\nTesting: {name}")
    try:
        test_fn()
        print(f"  ✓ PASSED")
        return True
    except RuntimeError as e:
        if "cudnn_grid_sampler_backward" in str(e):
            print(f"  ✗ FAILED: grid_sampler double backprop not supported")
        else:
            print(f"  ✗ FAILED: {e}")
        return False

def test_simple_ops():
    """Test that simple ops support double backprop"""
    # Simulate: parameter -> generator -> patch -> operation -> loss -> grad -> diversity_loss
    param = torch.randn(3 * 64 * 64, device=device, requires_grad=True)
    patch = param.view(1, 3, 64, 64)

    # Operation on patch
    y = patch * 2 + 1
    loss = y.sum()

    # First-order gradient w.r.t. patch (not param)
    grad = torch.autograd.grad(loss, patch, create_graph=True)[0]

    # Use gradient in diversity loss
    diversity_loss = (grad ** 2).sum()

    # Backprop diversity loss to param (second-order)
    diversity_loss.backward()

def test_gaussian_blur():
    """Test if Gaussian blur supports double backprop"""
    param = torch.randn(3 * 64 * 64, device=device, requires_grad=True)
    patch = param.view(1, 3, 64, 64)

    blurred = kornia.filters.gaussian_blur2d(patch, (17, 17), (4.0, 4.0))
    loss = blurred.sum()

    # First-order gradient w.r.t. patch
    grad = torch.autograd.grad(loss, patch, create_graph=True)[0]

    # Use gradient in diversity loss
    diversity_loss = (grad ** 2).sum()
    diversity_loss.backward()

def test_interpolate():
    """Test if F.interpolate supports double backprop"""
    param = torch.randn(3 * 256 * 512, device=device, requires_grad=True)
    patch = param.view(1, 3, 256, 512)

    downsampled = F.interpolate(patch, size=(32, 64), mode='bilinear', align_corners=True)
    loss = downsampled.sum()

    # First-order gradient w.r.t. patch
    grad = torch.autograd.grad(loss, patch, create_graph=True)[0]

    # Use gradient in diversity loss
    diversity_loss = (grad ** 2).sum()
    diversity_loss.backward()

def test_warp_perspective():
    """Test if kornia.warp_perspective supports double backprop"""
    param = torch.randn(3 * 256 * 512, device=device, requires_grad=True)
    patch = param.view(1, 3, 256, 512)

    # Create a simple perspective transform
    src = torch.tensor([[
        [0, 0], [512, 0], [512, 256], [0, 256]
    ]], dtype=torch.float32, device=device)
    dst = torch.tensor([[
        [10, 10], [502, 5], [510, 250], [5, 245]
    ]], dtype=torch.float32, device=device)

    M = K.get_perspective_transform(src, dst)
    warped = K.warp_perspective(patch, M, dsize=(256, 512))
    loss = warped.sum()

    # First-order gradient w.r.t. patch
    grad = torch.autograd.grad(loss, patch, create_graph=True)[0]

    # Use gradient in diversity loss
    diversity_loss = (grad ** 2).sum()
    diversity_loss.backward()

def test_crop_and_resize():
    """Test if kornia.crop_and_resize supports double backprop"""
    param = torch.randn(3 * 384 * 384, device=device, requires_grad=True)
    patch = param.view(1, 3, 384, 384)

    # Create crop box - shape should be [B, 4, 2]
    boxes = torch.tensor([[
        [100, 100],
        [300, 100],
        [300, 200],
        [100, 200]
    ]], dtype=torch.float32, device=device)

    cropped = kornia.geometry.crop_and_resize(patch, boxes, (64, 128))
    loss = cropped.sum()

    # First-order gradient w.r.t. patch
    grad = torch.autograd.grad(loss, patch, create_graph=True)[0]

    # Use gradient in diversity loss
    diversity_loss = (grad ** 2).sum()
    diversity_loss.backward()

def test_pad_and_slice():
    """Test if pad + slicing supports double backprop"""
    param = torch.randn(3 * 64 * 64, device=device, requires_grad=True)
    patch = param.view(1, 3, 64, 64)

    # Shift using pad + slice (our jitter implementation)
    shifted = F.pad(patch[:, :, :, :-2], (2, 0, 0, 0))
    loss = shifted.sum()

    # First-order gradient w.r.t. patch
    grad = torch.autograd.grad(loss, patch, create_graph=True)[0]

    # Use gradient in diversity loss
    diversity_loss = (grad ** 2).sum()
    diversity_loss.backward()

if __name__ == "__main__":
    print("="*60)
    print("Testing which operations support double backprop")
    print("="*60)

    results = {}
    results['Simple ops (mul, add)'] = test_case('Simple ops (mul, add)', test_simple_ops)
    results['Pad + Slice (our jitter)'] = test_case('Pad + Slice (our jitter)', test_pad_and_slice)
    results['F.interpolate'] = test_case('F.interpolate', test_interpolate)
    results['Gaussian blur'] = test_case('Gaussian blur', test_gaussian_blur)
    results['Warp perspective'] = test_case('Warp perspective', test_warp_perspective)
    results['Crop and resize'] = test_case('Crop and resize', test_crop_and_resize)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, passed in results.items():
        status = "✓" if passed else "✗"
        print(f"{status} {name}")

    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)

    failing = [name for name, passed in results.items() if not passed]
    if failing:
        print("The following operations are used in your pipeline and don't")
        print("support double backprop:")
        for name in failing:
            print(f"  - {name}")
        print("\nThese are likely called in:")
        if 'Warp perspective' in failing:
            print("  - apply_patch_to_image() → K.warp_perspective()")
        if 'Crop and resize' in failing:
            print("  - partial_loss() → OCR cropping → kornia.geometry.crop_and_resize()")
        if 'Gaussian blur' in failing:
            print("  - compute_diversity_loss() → kornia.filters.gaussian_blur2d()")
    else:
        print("All operations passed! The issue must be elsewhere.")
