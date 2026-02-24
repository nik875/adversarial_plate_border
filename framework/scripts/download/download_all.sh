#!/usr/bin/env bash
# =============================================================================
# download_all.sh — Download all training datasets for broad ensemble training
#
# Usage:
#   export HF_TOKEN=hf_...    # required for ImageNet only
#   bash download_all.sh
#   bash download_all.sh --data-root /data --jobs 16
#
# Individual datasets can be run independently; each script is idempotent
# (skips already-downloaded data via a .done flag).
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="$HOME/.cache"
JOBS=16

# Parse optional args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-root) DATA_ROOT="$2"; shift 2 ;;
        --jobs)      JOBS="$2";      shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo "  Ensemble training dataset downloader"
echo "  Data root : $DATA_ROOT"
echo "  Processes : $JOBS"
echo "============================================================"
echo ""

# ---- 1. ImageNet-1K (requires HF_TOKEN) ------------------------------------
echo ">>> [1/5] ImageNet-1K val (50k) + train subset (150k)"
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "    SKIP: HF_TOKEN not set."
    echo "    To download ImageNet:"
    echo "      1. Accept license: https://huggingface.co/datasets/ILSVRC/imagenet-1k"
    echo "      2. export HF_TOKEN=hf_..."
    echo "      3. python $SCRIPT_DIR/download_imagenet.py --output-dir $DATA_ROOT/imagenet"
else
    python "$SCRIPT_DIR/download_imagenet.py" \
        --output-dir "$DATA_ROOT/imagenet" \
        --train-samples 1000000 \
        --max-size 640
fi
echo ""

# ---- 2. COCO 2017 -----------------------------------------------------------
echo ">>> [2/5] COCO 2017 (train: 118k + val: 5k images)"
python "$SCRIPT_DIR/download_coco.py" \
    --output-dir "$DATA_ROOT/coco"
echo ""

# ---- 3. TextVQA -------------------------------------------------------------
echo ">>> [3/5] TextVQA train+val images (~34k text-in-scene images)"
python "$SCRIPT_DIR/download_textvqa.py" \
    --output-dir "$DATA_ROOT/textvqa"
echo ""

# ---- 4. CC3M subset ---------------------------------------------------------
echo ">>> [4/5] CC3M subset (~200-300k web images via img2dataset)"
python "$SCRIPT_DIR/download_cc3m.py" \
    --output-dir "$DATA_ROOT/cc3m" \
    --url-count 2500000 \
    --processes "$JOBS"
echo ""

# ---- 5. Open Images V7 subset -----------------------------------------------
echo ">>> [5/5] Open Images V7 subset (150k detection-style images)"
python "$SCRIPT_DIR/download_openimages.py" \
    --output-dir "$DATA_ROOT/openimages" \
    --num-samples 1000000 \
    --processes "$JOBS"
echo ""

# ---- Summary ----------------------------------------------------------------
echo "============================================================"
echo "  Download complete. Verify counts:"
for dir in imagenet/val imagenet/train coco/train2017 coco/val2017 \
           textvqa/train_val_images cc3m/images openimages; do
    full="$DATA_ROOT/$dir"
    if [[ -d "$full" ]]; then
        n=$(find "$full" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" -o -name "*.JPEG" \) 2>/dev/null | wc -l)
        printf "  %-40s %s images\n" "$dir" "$n"
    else
        printf "  %-40s (not found)\n" "$dir"
    fi
done
echo "============================================================"
