#!/usr/bin/env bash
# =============================================================================
# pack_and_upload.sh — tar all downloaded datasets and upload to B2
#
# Usage:
#   ./pack_and_upload.sh
#   ./pack_and_upload.sh --cache-dir ~/.cache --tar-path ~/general_image_data.tar
#   ./pack_and_upload.sh --skip-tar   # upload an already-created tar
#
# Requires: b2 CLI  (pip install b2)
# =============================================================================
set -euo pipefail

CACHE_DIR="$HOME/.cache"
TAR_PATH="$(pwd)/general_image_data.tar"
B2_BUCKET="licenseplate-dataset"
B2_KEY="general_image_data.tar"
SKIP_TAR=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cache-dir)  CACHE_DIR="$2";  shift 2 ;;
        --tar-path)   TAR_PATH="$2";   shift 2 ;;
        --skip-tar)   SKIP_TAR=true;   shift   ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

DATASETS=(imagenet coco textvqa cc3m openimages)

# ---------------------------------------------------------------------------
# 1. Build the tar archive
# ---------------------------------------------------------------------------
if [[ "$SKIP_TAR" == false ]]; then
    # Collect only directories that were fully downloaded (.done flag present)
    DIRS_TO_TAR=()
    for ds in "${DATASETS[@]}"; do
        dir="$CACHE_DIR/$ds"
        if [[ -d "$dir" ]]; then
            DIRS_TO_TAR+=("$ds")
            echo "  [+] $ds"
        else
            echo "  [-] $ds  (not found, skipping)"
        fi
    done

    if [[ ${#DIRS_TO_TAR[@]} -eq 0 ]]; then
        echo "ERROR: no dataset directories found under $CACHE_DIR"
        exit 1
    fi

    echo ""
    echo "==> Creating tar archive: $TAR_PATH"
    echo "    (no compression — images are already JPEG)"
    echo "    Datasets: ${DIRS_TO_TAR[*]}"
    echo ""

    # Run from CACHE_DIR so paths inside the tar are relative (e.g. imagenet/val/...)
    # Use pv for progress if available, otherwise plain tar
    if command -v pv &>/dev/null; then
        # Estimate total size for pv
        TOTAL_SIZE=$(du -sb "${DIRS_TO_TAR[@]/#/$CACHE_DIR/}" 2>/dev/null | awk '{sum+=$1} END{print sum}')
        tar -cf - -C "$CACHE_DIR" "${DIRS_TO_TAR[@]}" \
            | pv -s "$TOTAL_SIZE" -petrab \
            > "$TAR_PATH"
    else
        echo "    (install pv for a progress bar: apt install pv)"
        tar -cf "$TAR_PATH" -C "$CACHE_DIR" "${DIRS_TO_TAR[@]}"
    fi

    TAR_SIZE=$(du -sh "$TAR_PATH" | cut -f1)
    echo "    Created: $TAR_PATH  ($TAR_SIZE)"
else
    if [[ ! -f "$TAR_PATH" ]]; then
        echo "ERROR: $TAR_PATH not found (--skip-tar was set but file is missing)"
        exit 1
    fi
    echo "==> Skipping tar creation, using existing: $TAR_PATH"
fi

# ---------------------------------------------------------------------------
# 2. Upload to B2
# ---------------------------------------------------------------------------
echo ""
echo "==> Uploading to B2: b2://$B2_BUCKET/$B2_KEY"
b2 file upload "$B2_BUCKET" "$TAR_PATH" "$B2_KEY"
echo ""
echo "Done. File available at:"
echo "  b2://$B2_BUCKET/$B2_KEY"
