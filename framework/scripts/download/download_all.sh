#!/usr/bin/env bash
# =============================================================================
# download_all.sh — Download 100k balanced dataset subset for ensemble training
#
# Datasets: COCO (66k sampled), TextVQA (34k sampled)
# No ImageNet to avoid HuggingFace infrastructure bottlenecks
#
# Usage:
#   ./download_all.sh
#   ./download_all.sh --data-root ~/.cache
#
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

# ---- 1. COCO (66k sampled from train+val) ----
echo ">>> [1/2] COCO (66,000 images randomly sampled)"
python "$SCRIPT_DIR/download_coco.py" \
    --output-dir "$DATA_ROOT/coco" \
    --max-samples 66000
echo ""

# ---- 2. TextVQA (34k sampled) ----
echo ">>> [2/2] TextVQA (34,000 images randomly sampled)"
python "$SCRIPT_DIR/download_textvqa.py" \
    --output-dir "$DATA_ROOT/textvqa" \
    --max-samples 34000
echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "============================================================"
echo "  Download complete. Final image counts:"
echo "============================================================"

total=0
for name in coco textvqa; do
    case "$name" in
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
