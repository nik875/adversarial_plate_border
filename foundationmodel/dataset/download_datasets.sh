#!/bin/bash
# Download datasets from B2 and extract to cache directories
# Downloads b2://licenseplate-dataset/opensourcedata.tar

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
B2_PATH="b2://licenseplate-dataset/opensourcedata.tar"
TEMP_FILE=$(mktemp)

trap "rm -f $TEMP_FILE" EXIT

echo "Downloading datasets from B2..."
echo "Source: $B2_PATH"

# Download from B2
b2 file download "$B2_PATH" "$TEMP_FILE"

if [ ! -f "$TEMP_FILE" ]; then
    echo "Error: Failed to download from B2"
    exit 1
fi

echo "✓ Download complete"
echo "Extracting datasets to cache directories..."

# Extract to home directory (will restore .cache structure)
cd "$HOME"
tar -xf "$TEMP_FILE"

echo "✓ Extraction complete"
echo ""
echo "Cache directories:"
if [ -d "$HOME/.cache/iiit5k" ]; then
    echo "  ✓ IIIT5K: $HOME/.cache/iiit5k"
    ls -la "$HOME/.cache/iiit5k/" | head -10
fi

if [ -d "$HOME/.cache/huggingface/datasets" ]; then
    echo "  ✓ Hugging Face datasets: $HOME/.cache/huggingface/datasets"
    ls -la "$HOME/.cache/huggingface/datasets/" | head -5
fi

if [ -d "$HOME/.cache/icdar2011" ]; then
    echo "  ✓ ICDAR 2011 (2013 Challenge 1): $HOME/.cache/icdar2011"
    ls -la "$HOME/.cache/icdar2011/" | head -5
fi

if [ -d "$HOME/.cache/icdar2013" ]; then
    echo "  ✓ ICDAR 2013 (2015 Challenge 2): $HOME/.cache/icdar2013"
    ls -la "$HOME/.cache/icdar2013/" | head -5
fi

if [ -d "$HOME/.cache/cocotext_crops" ]; then
    echo "  ✓ COCO Text crops: $HOME/.cache/cocotext_crops"
    ls -la "$HOME/.cache/cocotext_crops/" | head -5
fi

if [ -d "$HOME/.cache/roboflow_lpr_crops" ]; then
    echo "  ✓ Roboflow LPR crops: $HOME/.cache/roboflow_lpr_crops"
    ls -la "$HOME/.cache/roboflow_lpr_crops/" | head -5
fi

echo ""
echo "✓ Dataset download and extraction complete"
