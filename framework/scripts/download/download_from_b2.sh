#!/usr/bin/env bash
# =============================================================================
# download_from_b2.sh — download general_image_data.tar from B2 and extract
#
# Usage:
#   ./download_from_b2.sh
#   ./download_from_b2.sh --cache-dir ~/.cache --tar-path ~/general_image_data.tar
#   ./download_from_b2.sh --skip-download  # extract an already-downloaded tar
#
# Requires: b2 CLI  (pip install b2)
# =============================================================================
set -euo pipefail

CACHE_DIR="$HOME/.cache"
TAR_PATH="$(pwd)/general_image_data.tar"
B2_URI="b2://licenseplate-dataset/general_image_data.tar"
SKIP_DOWNLOAD=false
KEEP_TAR=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cache-dir)      CACHE_DIR="$2";  shift 2 ;;
        --tar-path)       TAR_PATH="$2";   shift 2 ;;
        --skip-download)  SKIP_DOWNLOAD=true; shift ;;
        --keep-tar)       KEEP_TAR=true;   shift   ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$CACHE_DIR"

# ---------------------------------------------------------------------------
# 1. Download from B2
# ---------------------------------------------------------------------------
if [[ "$SKIP_DOWNLOAD" == false ]]; then
    echo "==> Downloading from B2: $B2_URI"
    echo "    → $TAR_PATH"
    b2 file download "$B2_URI" "$TAR_PATH"
    TAR_SIZE=$(du -sh "$TAR_PATH" | cut -f1)
    echo "    Downloaded: $TAR_SIZE"
else
    if [[ ! -f "$TAR_PATH" ]]; then
        echo "ERROR: $TAR_PATH not found (--skip-download was set but file is missing)"
        exit 1
    fi
    echo "==> Skipping download, using existing: $TAR_PATH"
fi

# ---------------------------------------------------------------------------
# 2. Extract into CACHE_DIR
# ---------------------------------------------------------------------------
echo ""
echo "==> Extracting into $CACHE_DIR ..."
echo "    (use pv for progress: apt install pv)"

if command -v pv &>/dev/null; then
    TAR_SIZE_BYTES=$(stat -c%s "$TAR_PATH" 2>/dev/null || stat -f%z "$TAR_PATH")
    pv -s "$TAR_SIZE_BYTES" -petrab "$TAR_PATH" | tar -xf - -C "$CACHE_DIR"
else
    tar -xf "$TAR_PATH" -C "$CACHE_DIR"
fi

echo "    Extracted to $CACHE_DIR"

# ---------------------------------------------------------------------------
# 3. Clean up tar (unless --keep-tar)
# ---------------------------------------------------------------------------
if [[ "$KEEP_TAR" == false ]]; then
    echo ""
    echo "==> Removing $TAR_PATH  (pass --keep-tar to skip)"
    rm -f "$TAR_PATH"
fi

# ---------------------------------------------------------------------------
# 4. Summary
# ---------------------------------------------------------------------------
echo ""
echo "==> Datasets available:"
for ds in imagenet coco textvqa cc3m openimages; do
    dir="$CACHE_DIR/$ds"
    if [[ -d "$dir" ]]; then
        n=$(find "$dir" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" -o -name "*.JPEG" \) 2>/dev/null | wc -l)
        sz=$(du -sh "$dir" 2>/dev/null | cut -f1)
        printf "  %-20s %8s images  %s\n" "$ds" "$n" "$sz"
    fi
done
echo ""
echo "Done. Run training with:"
echo "  python framework/scripts/train_ensemble.py framework/configs/ensemble_broad.yaml"
