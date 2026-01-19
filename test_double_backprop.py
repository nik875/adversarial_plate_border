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

def bilinear_grid_sample(input_tensor, grid, padding_mode='zeros'):
    """
    Pure PyTorch implementation of bilinear grid sampling.
    Supports double backprop since it only uses basic operations.

    Args:
        input_tensor: [B, C, H, W]
        grid: [B, H_out, W_out, 2] - normalized coordinates in [-1, 1]
        padding_mode: 'zeros' or 'border'

    Returns:
        sampled: [B, C, H_out, W_out]
    """
    B, C, H, W = input_tensor.shape
    _, H_out, W_out, _ = grid.shape

    # Denormalize grid from [-1, 1] to pixel coordinates
    # grid is (x, y) where x in [-1, 1] maps to [0, W-1], y maps to [0, H-1]
    grid_x = (grid[..., 0] + 1) * (W - 1) / 2  # [B, H_out, W_out]
    grid_y = (grid[..., 1] + 1) * (H - 1) / 2

    # Get integer coordinates and weights
    x0 = torch.floor(grid_x).long()
    x1 = x0 + 1
    y0 = torch.floor(grid_y).long()
    y1 = y0 + 1

    # Bilinear weights
    wx = grid_x - x0.float()
    wy = grid_y - y0.float()

    # Clamp coordinates to valid range
    x0 = torch.clamp(x0, 0, W - 1)
    x1 = torch.clamp(x1, 0, W - 1)
    y0 = torch.clamp(y0, 0, H - 1)
    y1 = torch.clamp(y1, 0, H - 1)

    # Create mask for out-of-bounds pixels
    valid_mask = ((grid_x >= 0) & (grid_x <= W - 1) &
                  (grid_y >= 0) & (grid_y <= H - 1)).float()  # [B, H_out, W_out]

    # Gather pixel values
    # Need to index as [batch, channel, y, x]
    output = torch.zeros(B, C, H_out, W_out, device=input_tensor.device, dtype=input_tensor.dtype)

    for b in range(B):
        for c in range(C):
            # Get 4 corner pixels
            p00 = input_tensor[b, c, y0[b], x0[b]]  # [H_out, W_out]
            p01 = input_tensor[b, c, y0[b], x1[b]]
            p10 = input_tensor[b, c, y1[b], x0[b]]
            p11 = input_tensor[b, c, y1[b], x1[b]]

            # Bilinear interpolation
            interp = (1 - wx[b]) * (1 - wy[b]) * p00 + \
                     wx[b] * (1 - wy[b]) * p01 + \
                     (1 - wx[b]) * wy[b] * p10 + \
                     wx[b] * wy[b] * p11

            # Apply mask if using zero padding
            if padding_mode == 'zeros':
                interp = interp * valid_mask[b]

            output[b, c] = interp

    return output

def manual_warp_perspective(input_tensor, M, dsize):
    """
    Pure PyTorch warp perspective using manual grid generation and bilinear sampling.

    Args:
        input_tensor: [B, C, H, W]
        M: [B, 3, 3] - perspective transformation matrices
        dsize: (H_out, W_out) - output size

    Returns:
        warped: [B, C, H_out, W_out]
    """
    B, C, H, W = input_tensor.shape
    H_out, W_out = dsize

    # Create output pixel grid
    # y_coords: [H_out, W_out], x_coords: [H_out, W_out]
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H_out, device=input_tensor.device, dtype=torch.float32),
        torch.arange(W_out, device=input_tensor.device, dtype=torch.float32),
        indexing='ij'
    )

    # Homogeneous coordinates [H_out, W_out, 3]
    ones = torch.ones_like(x_coords)
    grid_homo = torch.stack([x_coords, y_coords, ones], dim=-1)  # [H_out, W_out, 3]

    # Apply inverse perspective transform for each batch
    grid_list = []
    for b in range(B):
        # Flatten grid to [H_out*W_out, 3]
        grid_flat = grid_homo.reshape(-1, 3).t()  # [3, H_out*W_out]

        # Apply inverse transform: src = M^-1 @ dst
        M_inv = torch.inverse(M[b])
        src_homo = M_inv @ grid_flat  # [3, H_out*W_out]

        # Normalize by homogeneous coordinate
        src_x = src_homo[0] / src_homo[2]
        src_y = src_homo[1] / src_homo[2]

        # Normalize to [-1, 1] for grid_sample convention
        src_x_norm = 2 * src_x / (W - 1) - 1
        src_y_norm = 2 * src_y / (H - 1) - 1

        # Stack and reshape to [H_out, W_out, 2]
        grid_b = torch.stack([src_x_norm, src_y_norm], dim=-1).reshape(H_out, W_out, 2)
        grid_list.append(grid_b)

    grid = torch.stack(grid_list, dim=0)  # [B, H_out, W_out, 2]

    # Use our manual bilinear sampling
    return bilinear_grid_sample(input_tensor, grid, padding_mode='zeros')

def test_manual_warp_perspective():
    """Test if our manual warp_perspective supports double backprop"""
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

    # Use our manual implementation
    warped = manual_warp_perspective(patch, M, dsize=(256, 512))
    loss = warped.sum()

    # First-order gradient w.r.t. patch
    grad = torch.autograd.grad(loss, patch, create_graph=True)[0]

    # Use gradient in diversity loss
    diversity_loss = (grad ** 2).sum()
    diversity_loss.backward()

def manual_crop_and_resize(input_tensor, boxes, crop_size):
    """
    Pure PyTorch implementation of crop_and_resize.
    Supports double backprop by using manual_warp_perspective.

    Args:
        input_tensor: [B, C, H, W]
        boxes: [B, 4, 2] - 4 corner points (x, y) defining quadrilateral
        crop_size: (H_out, W_out) - output size

    Returns:
        cropped: [B, C, H_out, W_out]
    """
    B, C, H, W = input_tensor.shape
    H_out, W_out = crop_size

    # Define destination rectangle (axis-aligned)
    # Top-left: (0, 0), Top-right: (W_out-1, 0), Bottom-right: (W_out-1, H_out-1), Bottom-left: (0, H_out-1)
    dst_corners = torch.tensor([
        [0, 0],
        [W_out - 1, 0],
        [W_out - 1, H_out - 1],
        [0, H_out - 1]
    ], dtype=torch.float32, device=input_tensor.device)

    # Compute perspective transform for each batch element
    M_list = []
    for b in range(B):
        # Source corners from input boxes [4, 2]
        src_corners = boxes[b]  # [4, 2]

        # Compute transform from src (input quad) to dst (output rectangle)
        M = K.get_perspective_transform(src_corners.unsqueeze(0), dst_corners.unsqueeze(0))
        M_list.append(M[0])

    M_batch = torch.stack(M_list, dim=0)  # [B, 3, 3]

    # Use manual warp perspective
    return manual_warp_perspective(input_tensor, M_batch, dsize=crop_size)

def test_manual_crop_and_resize():
    """Test if our manual crop_and_resize supports double backprop"""
    param = torch.randn(3 * 384 * 384, device=device, requires_grad=True)
    patch = param.view(1, 3, 384, 384)

    # Create crop box - shape should be [B, 4, 2]
    boxes = torch.tensor([[
        [100, 100],
        [300, 100],
        [300, 200],
        [100, 200]
    ]], dtype=torch.float32, device=device)

    # Use our manual implementation
    cropped = manual_crop_and_resize(patch, boxes, (64, 128))
    loss = cropped.sum()

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
    results['Warp perspective (Kornia)'] = test_case('Warp perspective (Kornia)', test_warp_perspective)
    results['Warp perspective (Manual)'] = test_case('Warp perspective (Manual)', test_manual_warp_perspective)
    results['Crop and resize (Kornia)'] = test_case('Crop and resize (Kornia)', test_crop_and_resize)
    results['Crop and resize (Manual)'] = test_case('Crop and resize (Manual)', test_manual_crop_and_resize)

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
        print("The following operations don't support double backprop:")
        for name in failing:
            print(f"  - {name}")

        # Check if manual versions passed
        manual_passed = [name for name, passed in results.items() if passed and 'Manual' in name]
        if manual_passed:
            print("\n✓ Manual implementations passed! These can replace Kornia ops:")
            for name in manual_passed:
                print(f"  - {name}")

        print("\nKornia operations used in pipeline:")
        if any('Warp perspective' in name and 'Kornia' in name for name in failing):
            print("  - apply_patch_to_image() → K.warp_perspective()")
            print("    → Replace with manual_warp_perspective()")
        if any('Crop and resize' in name and 'Kornia' in name for name in failing):
            print("  - partial_loss() → OCR cropping → kornia.geometry.crop_and_resize()")
            print("    → Replace with manual_crop_and_resize()")
        if any('Gaussian blur' in name for name in failing):
            print("  - compute_diversity_loss() → kornia.filters.gaussian_blur2d()")
    else:
        print("All operations passed! Ready for gradient-based diversity training.")
