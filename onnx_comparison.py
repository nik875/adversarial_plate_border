#!/usr/bin/env python3
"""
Compare ONNX Runtime vs onnx2torch execution on the same input
This should help identify if model conversion is causing the OCR differences
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as T
import onnx
import onnx2torch
import onnxruntime as ort
from pathlib import Path
import argparse


def load_ocr_models(device='cpu'):
    """Load both ONNX Runtime and onnx2torch versions of the same model"""
    ocr_path = Path.home() / ".cache/fast-plate-ocr/cct-xs-v1-global-model/cct_xs_v1_global.onnx"

    if not ocr_path.exists():
        raise FileNotFoundError(f"OCR model not found: {ocr_path}")

    print(f"Loading models from: {ocr_path}")

    # Load ONNX Runtime session (what ALPR uses)
    print("Loading ONNX Runtime session...")
    ort_providers = ['CPUExecutionProvider']
    if device == 'cuda':
        ort_providers.insert(0, 'CUDAExecutionProvider')
    elif device == 'mps':
        # ONNX Runtime doesn't support MPS, fallback to CPU
        print("Note: ONNX Runtime doesn't support MPS, using CPU")

    ort_session = ort.InferenceSession(str(ocr_path), providers=ort_providers)
    print(f"ONNX Runtime providers: {ort_session.get_providers()}")

    # Load onnx2torch model (what your pipeline uses)
    print("Loading onnx2torch model...")
    onnx_model = onnx.load(str(ocr_path))
    pytorch_model = onnx2torch.convert(onnx_model)
    pytorch_model.to(device)
    pytorch_model.eval()

    for param in pytorch_model.parameters():
        param.requires_grad = False

    return ort_session, pytorch_model


def logits_to_text(logits, alphabet='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_'):
    """Convert OCR logits to readable text"""
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits)

    probs = F.softmax(logits, dim=-1)
    pred_chars = torch.argmax(probs, dim=-1).squeeze()

    if pred_chars.dim() == 0:  # Single character
        pred_chars = pred_chars.unsqueeze(0)

    text = ""
    for char_idx in pred_chars:
        char_idx = char_idx.item()
        if char_idx < len(alphabet):
            char = alphabet[char_idx]
            if char != '_':
                text += char
    return text.strip()


def prepare_test_input(image_path, target_size=(64, 128)):
    """Prepare test input exactly as the ALPR package would"""
    # Load and resize image
    pil_image = Image.open(image_path).convert('RGB')
    resized_image = pil_image.resize((target_size[1], target_size[0]))  # PIL uses (width, height)

    print(f"Original size: {pil_image.size}")
    print(f"Resized to: {resized_image.size}")

    # Convert to numpy array in NHWC format [0, 255] range
    np_image = np.array(resized_image)
    np_batch = np.expand_dims(np_image, axis=0).astype(np.float32)

    print(f"NumPy input shape: {np_batch.shape}")
    print(f"NumPy input range: [{np_batch.min():.1f}, {np_batch.max():.1f}]")

    return np_batch, resized_image


def test_onnx_vs_pytorch(image_path, device='cpu'):
    """Compare ONNX Runtime vs onnx2torch on the same input"""

    print(f"{'='*60}")
    print(f"COMPARING ONNX RUNTIME vs ONNX2TORCH")
    print(f"{'='*60}")
    print(f"Image: {image_path}")
    print(f"Device: {device}")
    print()

    # Load models
    ort_session, pytorch_model = load_ocr_models(device)

    # Prepare test input
    np_input, pil_image = prepare_test_input(image_path)

    # Test various input formats
    test_cases = [
        {
            'name': 'Raw [0,255] NHWC',
            'onnx_input': np_input,
            'pytorch_input': torch.from_numpy(np_input).to(device)
        },
        {
            'name': 'Normalized [0,1] NHWC',
            'onnx_input': np_input / 255.0,
            'pytorch_input': torch.from_numpy(np_input / 255.0).to(device)
        },
    ]

    results = []

    for i, test_case in enumerate(test_cases):
        print(f"{i+1}. Testing: {test_case['name']}")
        print(f"   ONNX input shape: {test_case['onnx_input'].shape}")
        print(
            f"   ONNX input range: [{test_case['onnx_input'].min():.3f}, {test_case['onnx_input'].max():.3f}]")

        # Run ONNX Runtime
        try:
            input_name = ort_session.get_inputs()[0].name
            print(f"   ONNX input name: {input_name}")

            ort_outputs = ort_session.run(None, {input_name: test_case['onnx_input']})
            ort_logits = ort_outputs[0]
            ort_text = logits_to_text(ort_logits)
            ort_success = True
            ort_error = None

            print(f"   ONNX Runtime result: '{ort_text}'")

        except Exception as e:
            ort_text = None
            ort_success = False
            ort_error = str(e)
            print(f"   ONNX Runtime ERROR: {ort_error}")

        # Run onnx2torch
        try:
            with torch.no_grad():
                pytorch_logits = pytorch_model(test_case['pytorch_input'])
            pytorch_text = logits_to_text(pytorch_logits)
            pytorch_success = True
            pytorch_error = None

            print(f"   onnx2torch result: '{pytorch_text}'")

        except Exception as e:
            pytorch_text = None
            pytorch_success = False
            pytorch_error = str(e)
            print(f"   onnx2torch ERROR: {pytorch_error}")

        # Compare results
        if ort_success and pytorch_success:
            if ort_text == pytorch_text:
                print(f"   ✓ RESULTS MATCH!")
            else:
                print(f"   ✗ RESULTS DIFFER!")
                print(f"     ONNX Runtime: '{ort_text}'")
                print(f"     onnx2torch:   '{pytorch_text}'")

        results.append({
            'name': test_case['name'],
            'onnx_success': ort_success,
            'onnx_text': ort_text,
            'onnx_error': ort_error,
            'pytorch_success': pytorch_success,
            'pytorch_text': pytorch_text,
            'pytorch_error': pytorch_error,
            'match': ort_success and pytorch_success and (ort_text == pytorch_text)
        })

        print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    successful_results = [r for r in results if r['onnx_success'] and r['pytorch_success']]
    matching_results = [r for r in results if r['match']]

    print(f"Successful comparisons: {len(successful_results)}/{len(results)}")
    print(f"Matching results: {len(matching_results)}/{len(successful_results)}")

    if successful_results:
        print("\nDETAILED RESULTS:")
        for result in successful_results:
            match_symbol = "✓" if result['match'] else "✗"
            print(f"  {match_symbol} {result['name']}:")
            print(f"    ONNX Runtime: '{result['onnx_text']}'")
            print(f"    onnx2torch:   '{result['pytorch_text']}'")

    # Check if any ONNX Runtime result is correct
    target_text = 'VRJ7774'
    onnx_correct = any(r['onnx_text'] == target_text for r in successful_results)
    pytorch_correct = any(r['pytorch_text'] == target_text for r in successful_results)

    print(f"\nCORRECTNESS CHECK (expected: '{target_text}'):")
    print(f"  ONNX Runtime correct: {onnx_correct}")
    print(f"  onnx2torch correct:   {pytorch_correct}")

    if onnx_correct and not pytorch_correct:
        print("\n🔍 DIAGNOSIS: onnx2torch conversion is the issue!")
        print("   The ONNX Runtime produces correct results but onnx2torch doesn't.")
        print("   This confirms that model conversion is causing the OCR failures.")

        # Find the correct input format
        for result in successful_results:
            if result['onnx_text'] == target_text:
                print(f"   Correct format: {result['name']}")
                break

    elif pytorch_correct and not onnx_correct:
        print("\n🤔 UNEXPECTED: onnx2torch is correct but ONNX Runtime isn't!")
        print("   This is unusual - please double-check the test setup.")

    elif not onnx_correct and not pytorch_correct:
        print("\n⚠️  NEITHER APPROACH WORKS")
        print("   The issue may be in preprocessing, not model conversion.")
        print("   Try different input formats or check the model file integrity.")

    else:
        print("\n✅ BOTH APPROACHES WORK")
        print("   The issue must be elsewhere in your pipeline.")

    return results


def main():
    parser = argparse.ArgumentParser(description='Compare ONNX Runtime vs onnx2torch OCR execution')
    parser.add_argument('image_path', help='Path to cropped license plate image')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu',
                        help='Device for PyTorch model (ONNX Runtime uses CPU/CUDA automatically)')

    args = parser.parse_args()

    # Auto-detect device if not specified
    if args.device == 'cpu' and torch.cuda.is_available():
        device = 'cuda'
        print("Auto-detected CUDA, using GPU for PyTorch model")
    else:
        device = args.device

    try:
        results = test_onnx_vs_pytorch(args.image_path, device)
    except Exception as e:
        print(f"Comparison failed: {e}")
        raise


if __name__ == "__main__":
    main()
