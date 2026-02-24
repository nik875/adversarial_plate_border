#!/usr/bin/env bash
# =============================================================================
# download_all.sh — Download all training datasets for broad ensemble training
#
# Usage:
#   export HF_TOKEN=hf_...    # required for ImageNet only
#   ./download_all.sh
#   ./download_all.sh --data-root ~/.cache --jobs 16
#   ./download_all.sh --skip-smoke   # skip smoke test, go straight to full download
#
# Runs a smoke-test phase first (small downloads to verify connectivity and
# tooling), then proceeds with the full download only if all tests pass.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="$HOME/.cache"
JOBS=16
SKIP_SMOKE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-root)   DATA_ROOT="$2"; shift 2 ;;
        --jobs)        JOBS="$2";      shift 2 ;;
        --skip-smoke)  SKIP_SMOKE=true; shift  ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo "  Ensemble training dataset downloader"
echo "  Data root : $DATA_ROOT"
echo "  Processes : $JOBS"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# Helper: count images in a directory
# ---------------------------------------------------------------------------
count_images() {
    find "$1" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" -o -name "*.JPEG" \) 2>/dev/null | wc -l
}

# ---------------------------------------------------------------------------
# PHASE 1: Smoke tests
# ---------------------------------------------------------------------------
if [[ "$SKIP_SMOKE" == false ]]; then
    echo "============================================================"
    echo "  PHASE 1: Smoke tests"
    echo "  (small downloads to verify connectivity + tooling)"
    echo "============================================================"
    echo ""

    SMOKE_DIR="$(mktemp -d)"
    trap 'echo ""; echo "Cleaning up smoke test dir..."; rm -rf "$SMOKE_DIR"' EXIT
    SMOKE_FAILED=0

    # -- COCO: HEAD request only (ZIP download, can't do partial) -------------
    echo ">>> Smoke [1/5]: COCO — checking URL reachability"
    if curl -sf --head "http://images.cocodataset.org/zips/val2017.zip" > /dev/null; then
        echo "    PASS"
    else
        echo "    FAIL: COCO server unreachable"
        SMOKE_FAILED=1
    fi
    echo ""

    # -- TextVQA: HEAD request only -------------------------------------------
    echo ">>> Smoke [2/5]: TextVQA — checking URL reachability"
    if curl -sf --head "https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip" > /dev/null; then
        echo "    PASS"
    else
        echo "    FAIL: TextVQA server unreachable"
        SMOKE_FAILED=1
    fi
    echo ""

    # -- CC3M: stream 200 metadata rows, fetch ~100 images --------------------
    echo ">>> Smoke [3/5]: CC3M — streaming metadata + downloading ~100 images"
    if python "$SCRIPT_DIR/download_cc3m.py" \
        --output-dir "$SMOKE_DIR/cc3m" \
        --url-count 200 \
        --processes 2; then
        n=$(count_images "$SMOKE_DIR/cc3m")
        if [[ $n -gt 0 ]]; then
            echo "    PASS ($n images downloaded)"
        else
            echo "    FAIL: 0 images downloaded — check img2dataset install or network"
            SMOKE_FAILED=1
        fi
    else
        echo "    FAIL: script exited with error"
        SMOKE_FAILED=1
    fi
    echo ""

    # -- OpenImages: download 50 images from S3 -------------------------------
    echo ">>> Smoke [4/5]: OpenImages — downloading 50 images from S3"
    if python "$SCRIPT_DIR/download_openimages.py" \
        --output-dir "$SMOKE_DIR/openimages" \
        --num-samples 50 \
        --processes 2; then
        n=$(count_images "$SMOKE_DIR/openimages")
        if [[ $n -ge 40 ]]; then
            echo "    PASS ($n/50 images downloaded)"
        else
            echo "    FAIL: only $n/50 images — S3 URLs should be stable, check network"
            SMOKE_FAILED=1
        fi
    else
        echo "    FAIL: script exited with error"
        SMOKE_FAILED=1
    fi
    echo ""

    # -- ImageNet: verify HF token + stream 1 sample --------------------------
    echo ">>> Smoke [5/5]: ImageNet"
    if [[ -z "${HF_TOKEN:-}" ]]; then
        echo "    SKIP (HF_TOKEN not set — ImageNet will be skipped in full download too)"
    else
        if python - <<'PYEOF'
import os, sys
try:
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ['HF_TOKEN'])
    info = api.dataset_info('ILSVRC/imagenet-1k')
    assert info.id is not None
    print("    PASS (token valid, dataset accessible)")
except Exception as e:
    print(f"    FAIL: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF
        then
            : # pass already printed inside python
        else
            echo "    FAIL: ImageNet token check failed"
            SMOKE_FAILED=1
        fi
    fi
    echo ""

    # -- Result ----------------------------------------------------------------
    # Disable the EXIT trap before the failure check so we control cleanup
    trap - EXIT
    rm -rf "$SMOKE_DIR"

    if [[ $SMOKE_FAILED -ne 0 ]]; then
        echo "============================================================"
        echo "  SMOKE TEST FAILED — fix the issues above before re-running"
        echo "  To skip smoke tests: ./download_all.sh --skip-smoke"
        echo "============================================================"
        exit 1
    fi

    echo "============================================================"
    echo "  All smoke tests passed — proceeding with full download"
    echo "============================================================"
    echo ""
fi

# ---------------------------------------------------------------------------
# PHASE 2: Full download
# ---------------------------------------------------------------------------
echo "============================================================"
echo "  PHASE 2: Full download"
echo "============================================================"
echo ""

# ---- 1. ImageNet-1K (requires HF_TOKEN) ------------------------------------
echo ">>> [1/5] ImageNet-1K val (50k) + train (1M images)"
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
echo ">>> [4/5] CC3M subset (~1.2M images via img2dataset)"
python "$SCRIPT_DIR/download_cc3m.py" \
    --output-dir "$DATA_ROOT/cc3m" \
    --url-count 2500000 \
    --processes "$JOBS"
echo ""

# ---- 5. Open Images V7 subset -----------------------------------------------
echo ">>> [5/5] Open Images V7 (1M images from S3)"
python "$SCRIPT_DIR/download_openimages.py" \
    --output-dir "$DATA_ROOT/openimages" \
    --num-samples 1000000 \
    --processes "$JOBS"
echo ""

# ---- Summary ----------------------------------------------------------------
echo "============================================================"
echo "  Download complete. Final image counts:"
for dir in imagenet/val imagenet/train coco/train2017 coco/val2017 \
           textvqa/train_val_images cc3m/images openimages/images; do
    full="$DATA_ROOT/$dir"
    if [[ -d "$full" ]]; then
        n=$(count_images "$full")
        printf "  %-40s %s images\n" "$dir" "$n"
    else
        printf "  %-40s (not found)\n" "$dir"
    fi
done
echo "============================================================"
