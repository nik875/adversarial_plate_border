"""
Verify that the new all-tensor corner transform in diff_prep produces
identical results to the old numpy-based implementation.
"""
import numpy as np
import torch
import torch.nn.functional as F

# ── Old implementations (copied verbatim from trainer.py) ─────────────────────

def _diff_letterbox(img_chw, target_size):
    C, H, W = img_chw.shape
    r      = min(target_size / H, target_size / W)
    new_h  = int(round(H * r))
    new_w  = int(round(W * r))
    img    = F.interpolate(img_chw.unsqueeze(0), size=(new_h, new_w),
                           mode="bilinear", align_corners=False).squeeze(0)
    dw     = (target_size - new_w) / 2
    dh     = (target_size - new_h) / 2
    top    = int(round(dh - 0.1)); bottom = int(round(dh + 0.1))
    left   = int(round(dw - 0.1)); right  = int(round(dw + 0.1))
    img    = F.pad(img, (left, right, top, bottom), value=114.0 / 255.0)
    return img, r, dw, dh

def _diff_resize(img_chw, target_w, target_h):
    C, H, W = img_chw.shape
    img     = F.interpolate(img_chw.unsqueeze(0), size=(target_h, target_w),
                            mode="bilinear", align_corners=False).squeeze(0)
    return img, target_w / W, target_h / H

def _corners_letterbox_np(corners: np.ndarray, r: float, dw: float, dh: float) -> np.ndarray:
    """Old numpy version."""
    c = corners.astype(np.float32).copy()
    c[:, 0] = c[:, 0] * r + dw
    c[:, 1] = c[:, 1] * r + dh
    return c

def _corners_resize_np(corners: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """Old numpy version."""
    c = corners.astype(np.float32).copy()
    c[:, 0] *= sx
    c[:, 1] *= sy
    return c

# ── New implementations ────────────────────────────────────────────────────────

def _corners_letterbox_t(corners: torch.Tensor, r: float, dw: float, dh: float) -> torch.Tensor:
    """New tensor version."""
    c = corners.clone()
    c[:, 0] = c[:, 0] * r + dw
    c[:, 1] = c[:, 1] * r + dh
    return c

def _corners_resize_t(corners: torch.Tensor, sx: float, sy: float) -> torch.Tensor:
    """New tensor version."""
    c = corners.clone()
    c[:, 0] = c[:, 0] * sx
    c[:, 1] = c[:, 1] * sy
    return c

# ── Test harness ───────────────────────────────────────────────────────────────

def run_tests(device: str):
    print(f"\n=== Testing on device: {device} ===")
    rng = np.random.default_rng(42)
    all_passed = True

    # Test cases: (H, W, target_size_or_dims, mode)
    test_cases = [
        # Various aspect ratios with letterbox (yolo-style)
        (720,  1280, 640,  "letterbox"),
        (1080, 1920, 640,  "letterbox"),
        (480,  640,  384,  "letterbox"),
        (1000, 800,  640,  "letterbox"),  # portrait
        (640,  640,  640,  "letterbox"),  # already square
        (3024, 4032, 640,  "letterbox"),  # large HEIC-style
        # Resize
        (720,  1280, (640, 640), "resize"),
        (480,  640,  (384, 384), "resize"),
    ]

    for H, W, target, mode in test_cases:
        corners_np = rng.uniform(50, min(H, W) - 50, size=(4, 2)).astype(np.float32)
        corners_t  = torch.from_numpy(corners_np).to(device)
        img        = torch.rand(3, H, W, device=device)

        if mode == "letterbox":
            target_size = target
            _, r, dw, dh = _diff_letterbox(img, target_size)
            out_np = _corners_letterbox_np(corners_np, r, dw, dh)
            out_t  = _corners_letterbox_t(corners_t,  r, dw, dh)
            label  = f"letterbox H={H} W={W} → {target_size}"
        else:
            tw, th = target
            _, sx, sy = _diff_resize(img, tw, th)
            out_np = _corners_resize_np(corners_np, sx, sy)
            out_t  = _corners_resize_t(corners_t,   sx, sy)
            label  = f"resize H={H} W={W} → {tw}×{th}"

        # Compare: convert tensor result back to numpy for comparison
        out_t_np = out_t.cpu().numpy()
        max_err  = np.abs(out_t_np - out_np).max()
        passed   = max_err < 1e-5

        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {label}  max_err={max_err:.2e}")
        if not passed:
            print(f"         numpy:  {out_np}")
            print(f"         tensor: {out_t_np}")
            all_passed = False

    return all_passed


if __name__ == "__main__":
    passed = run_tests("cpu")
    if torch.cuda.is_available():
        passed &= run_tests("cuda")
    else:
        print("\n(CUDA not available — skipping GPU test)")

    print()
    if passed:
        print("All tests passed.")
    else:
        print("SOME TESTS FAILED.")
        raise SystemExit(1)
