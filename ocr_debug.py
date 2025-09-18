#!/usr/bin/env python3
"""
Debug OCR input formats to find the difference between ALPR and pipeline
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as T
import onnx
import onnx2torch
from pathlib import Path


def load_ocr_model(device):
    """Load the same OCR model used by both ALPR and pipeline"""
    ocr_path = Path.home() / ".cache/fast-plate-ocr/cct-xs-v1-global-model/cct_xs_v1_global.onnx"

    if not ocr_path.exists():
        raise FileNotFoundError(f"OCR model not found: {ocr_path}")

    print(f"Loading OCR model from: {ocr_path}")
    ocr_onnx = onnx.load(str(ocr_path))
    ocr_model = onnx2torch.convert(ocr_onnx)
    ocr_model.to(device)
    ocr_model.eval()

    for param in ocr_model.parameters():
        param.requires_grad = False

    return ocr_model


def logits_to_text(logits, alphabet='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_'):
    """Convert OCR logits to readable text"""
    probs = F.softmax(logits, dim=-1)
    pred_chars = torch.argmax(probs, dim=-1).squeeze(0)

    text = ""
    for char_idx in pred_chars:
        char_idx = char_idx.item()
        if char_idx < len(alphabet):
            char = alphabet[char_idx]
            if char != '_':
                text += char
    return text.strip()


def test_ocr_formats(image_path, device='cpu'):
    """Test different OCR input formats on the same cropped image"""

    print(f"Testing OCR formats on: {image_path}")
    print(f"Device: {device}")

    # Load OCR model
    ocr_model = load_ocr_model(device)

    # Load test image (should be one of the debug crops)
    pil_image = Image.open(image_path).convert('RGB')
    print(f"Loaded image: {pil_image.size}")

    # Test different input formats
    test_cases = []

    # Case 1: Pipeline format (current approach)
    tensor_01 = T.ToTensor()(pil_image)  # [0,1] range
    resized_01 = T.Resize((64, 128))(tensor_01)
    batch_01 = resized_01.unsqueeze(0).to(device)
    nhwc_01 = batch_01.permute(0, 2, 3, 1)  # NCHW -> NHWC

    test_cases.append({
        'name': 'Pipeline format [0,1] NHWC',
        'input': nhwc_01,
        'description': f'Shape: {nhwc_01.shape}, Range: [{nhwc_01.min():.3f}, {nhwc_01.max():.3f}]'
    })

    # Case 2: Integer [0,255] format NHWC
    np_array = np.array(pil_image.resize((128, 64)))  # PIL resize format
    tensor_255 = torch.from_numpy(np_array).float().unsqueeze(0).to(device)  # NHWC directly

    test_cases.append({
        'name': 'Integer [0,255] NHWC',
        'input': tensor_255,
        'description': f'Shape: {tensor_255.shape}, Range: [{tensor_255.min():.1f}, {tensor_255.max():.1f}]'
    })

    # Case 3: Normalized [0,255] -> [0,1] NHWC
    tensor_255_norm = tensor_255 / 255.0

    test_cases.append({
        'name': 'Normalized [0,255]->[0,1] NHWC',
        'input': tensor_255_norm,
        'description': f'Shape: {tensor_255_norm.shape}, Range: [{tensor_255_norm.min():.3f}, {tensor_255_norm.max():.3f}]'
    })

    # Case 4: Different resize method (CV2-style)
    import cv2
    cv_image = cv2.imread(image_path)
    cv_resized = cv2.resize(cv_image, (128, 64))  # Note: CV2 uses (width, height)
    cv_rgb = cv2.cvtColor(cv_resized, cv2.COLOR_BGR2RGB)
    cv_tensor = torch.from_numpy(cv_rgb).float().unsqueeze(0).to(device)

    test_cases.append({
        'name': 'OpenCV resize BGR->RGB',
        'input': cv_tensor,
        'description': f'Shape: {cv_tensor.shape}, Range: [{cv_tensor.min():.1f}, {cv_tensor.max():.1f}]'
    })

    # Case 5: OpenCV normalized
    cv_tensor_norm = cv_tensor / 255.0

    test_cases.append({
        'name': 'OpenCV normalized [0,1]',
        'input': cv_tensor_norm,
        'description': f'Shape: {cv_tensor_norm.shape}, Range: [{cv_tensor_norm.min():.3f}, {cv_tensor_norm.max():.3f}]'
    })

    # Case 6: Try different normalization (ImageNet style)
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 1, 3).to(device)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 1, 3).to(device)
    imagenet_norm = (tensor_255_norm - imagenet_mean) / imagenet_std

    test_cases.append({
        'name': 'ImageNet normalized',
        'input': imagenet_norm,
        'description': f'Shape: {imagenet_norm.shape}, Range: [{imagenet_norm.min():.3f}, {imagenet_norm.max():.3f}]'
    })

    print(f"\nTesting {len(test_cases)} different input formats:\n")

    # Test each format
    results = []
    for i, test_case in enumerate(test_cases):
        print(f"{i+1}. {test_case['name']}")
        print(f"   {test_case['description']}")

        try:
            with torch.no_grad():
                logits = ocr_model(test_case['input'])
            text = logits_to_text(logits)
            success = True
            error = None
        except Exception as e:
            text = None
            success = False
            error = str(e)

        result = {
            'name': test_case['name'],
            'text': text,
            'success': success,
            'error': error
        }
        results.append(result)

        if success:
            print(f"   Result: '{text}'")
            if text == 'VRJ7774' or 'VRJ' in text or '7774' in text:
                print(f"   *** POTENTIAL MATCH! ***")
        else:
            print(f"   ERROR: {error}")

        print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    successful_results = [r for r in results if r['success']]

    print(f"Successful formats: {len(successful_results)}/{len(test_cases)}")

    target_text = 'VRJ7774'
    matches = [r for r in successful_results if r['text'] == target_text]
    partial_matches = [r for r in successful_results if r['text'] != target_text and (
        'VRJ' in r['text'] or '7774' in r['text'] or len(set(r['text']) & set(target_text)) >= 3
    )]

    if matches:
        print(f"\nEXACT MATCHES ('{target_text}'):")
        for match in matches:
            print(f"  ✓ {match['name']}")

    if partial_matches:
        print(f"\nPARTIAL MATCHES:")
        for match in partial_matches:
            print(f"  ~ {match['name']}: '{match['text']}'")

    print(f"\nALL RESULTS:")
    for result in successful_results:
        status = "✓" if result['text'] == target_text else "✗"
        print(f"  {status} {result['name']}: '{result['text']}'")

    if not matches:
        print(f"\n⚠️  No exact matches found. The issue may be:")
        print(f"   1. Different OCR model version/configuration")
        print(f"   2. Missing preprocessing steps")
        print(f"   3. Different backend (ONNX runtime vs PyTorch)")
        print(f"   4. Model conversion differences")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Debug OCR input formats')
    parser.add_argument('crop_image', help='Path to cropped license plate image')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu',
                        help='Device to use for inference')

    args = parser.parse_args()

    # Auto-detect device if not specified
    if args.device == 'cpu':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device

    try:
        results = test_ocr_formats(args.crop_image, device)
    except Exception as e:
        print(f"Testing failed: {e}")
        raise


if __name__ == "__main__":
    main()
