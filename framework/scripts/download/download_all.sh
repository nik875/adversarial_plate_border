#!/usr/bin/env bash
# =============================================================================
# download_all.sh — Download 100k balanced dataset subset for ensemble training
#
# Datasets: ImageNet-train (34k), COCO (33k sampled), TextVQA (33k sampled)
#
# Usage:
#   export HF_TOKEN=hf_...    # required for ImageNet
#   ./download_all.sh
#   ./download_all.sh --data-root ~/.cache
#
# Requires: HF_TOKEN set for ImageNet, pip install datasets
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="$HOME/.cache"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-root) DATA_ROOT="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo "  Ensemble training dataset downloader (100k subset)"
echo "  Data root: $DATA_ROOT"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# Helper: count images in a directory
# ---------------------------------------------------------------------------
count_images() {
    find "$1" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" -o -name "*.JPEG" \) 2>/dev/null | wc -l
}

# ---------------------------------------------------------------------------
# Connectivity checks
# ---------------------------------------------------------------------------
echo ">>> Checking connectivity..."
echo -n "  ImageNet (HuggingFace): "
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "SKIP (HF_TOKEN not set)"
    SKIP_IMAGENET=1
else
    if python - <<'PYEOF'
import os, sys
try:
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ['HF_TOKEN'])
    info = api.dataset_info('ILSVRC/imagenet-1k')
    assert info.id is not None
    print("OK")
except Exception as e:
    print(f"FAIL ({e})", file=sys.stderr)
    sys.exit(1)
PYEOF
    then
        SKIP_IMAGENET=0
    else
        echo "Token check failed"
        exit 1
    fi
fi

echo -n "  COCO (cocodataset.org): "
if curl -sf --head "http://images.cocodataset.org/zips/val2017.zip" > /dev/null; then
    echo "OK"
else
    echo "FAIL"
    exit 1
fi

echo -n "  TextVQA (fbaipublicfiles.com): "
if curl -sf --head "https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip" > /dev/null; then
    echo "OK"
else
    echo "FAIL"
    exit 1
fi

echo ""

# ---------------------------------------------------------------------------
# Download datasets
# ---------------------------------------------------------------------------
echo "============================================================"
echo "  Downloading datasets..."
echo "============================================================"
echo ""

# ---- 1. ImageNet-train (streaming, stops at 34k) ----
if [[ $SKIP_IMAGENET -eq 0 ]]; then
    echo ">>> [1/3] ImageNet-train (34,000 images, streaming download)"
    python "$SCRIPT_DIR/download_imagenet.py" \
        --output-dir "$DATA_ROOT/imagenet" \
        --train-samples 34000 \
        --skip-test \
        --max-size 640
    echo ""
else
    echo ">>> [1/3] ImageNet-train — SKIPPED (HF_TOKEN not set)"
    echo ""
fi

# ---- 2. COCO (33k sampled from train+val) ----
echo ">>> [2/3] COCO (33,000 images randomly sampled)"
python "$SCRIPT_DIR/download_coco.py" \
    --output-dir "$DATA_ROOT/coco" \
    --max-samples 33000
echo ""

# ---- 3. TextVQA (33k sampled) ----
echo ">>> [3/3] TextVQA (33,000 images randomly sampled)"
python "$SCRIPT_DIR/download_textvqa.py" \
    --output-dir "$DATA_ROOT/textvqa" \
    --max-samples 33000
echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "============================================================"
echo "  Download complete. Final image counts:"
echo "============================================================"

total=0
for name in imagenet coco textvqa; do
    case "$name" in
        imagenet) pattern="$DATA_ROOT/imagenet/train" ;;
        coco) pattern="$DATA_ROOT/coco" ;;
        textvqa) pattern="$DATA_ROOT/textvqa" ;;
    esac

    if [[ -d "$pattern" ]]; then
        n=$(count_images "$pattern")
        printf "  %-20s %8d images\n" "$name" "$n"
        total=$((total + n))
    else
        printf "  %-20s (not found)\n" "$name"
    fi
done

echo "  ────────────────────────────────"
printf "  %-20s %8d images\n" "TOTAL" "$total"
echo "============================================================"
echo ""
echo "Ready to train with:"
echo "  python framework/scripts/train_ensemble.py framework/configs/ensemble_broad.yaml"
