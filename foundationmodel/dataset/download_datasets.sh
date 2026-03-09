#!/bin/bash
# Download datasets from B2 and extract to cache directories
# Downloads b2://licenseplate-dataset/opensourcedata.tar

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
B2_PATH="b2://licenseplate-dataset/opensourcedata.tar"
LOCAL_TAR="$SCRIPT_DIR/opensourcedata.tar"
TEMP_FILE=$(mktemp)
SKIP_DOWNLOAD=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --local)
            SKIP_DOWNLOAD=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --local       Skip B2 download and use local opensourcedata.tar in this directory"
            echo "  --help, -h    Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

trap "rm -f $TEMP_FILE" EXIT

if [ "$SKIP_DOWNLOAD" = true ]; then
    echo "Using local tarball..."
    if [ ! -f "$LOCAL_TAR" ]; then
        echo "Error: opensourcedata.tar not found at $LOCAL_TAR"
        exit 1
    fi
    echo "Source: $LOCAL_TAR"
    cp "$LOCAL_TAR" "$TEMP_FILE"
else
    echo "Downloading datasets from B2..."
    echo "Source: $B2_PATH"
    # Download from B2
    b2 file download "$B2_PATH" "$TEMP_FILE"
fi

if [ ! -f "$TEMP_FILE" ]; then
    echo "Error: Failed to get tarball"
    exit 1
fi

echo "✓ Tarball ready"
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

if [ -d "$HOME/.cache/kaggle_lp_crops" ]; then
    echo "  ✓ Kaggle LP crops: $HOME/.cache/kaggle_lp_crops"
    ls -la "$HOME/.cache/kaggle_lp_crops/" | head -5
fi

if [ -d "$HOME/.cache/indian_plates_kaggle_crops" ]; then
    echo "  ✓ Indian Plates Kaggle crops: $HOME/.cache/indian_plates_kaggle_crops"
    ls -la "$HOME/.cache/indian_plates_kaggle_crops/" | head -5
fi

if [ -d "$HOME/.cache/ccpd2019_crops" ]; then
    echo "  ✓ CCPD2019 crops (variants): $HOME/.cache/ccpd2019_crops"
    ls -la "$HOME/.cache/ccpd2019_crops/" | head -15
fi

if [ -d "$HOME/.cache/mercosur_crops" ]; then
    echo "  ✓ Mercosur crops: $HOME/.cache/mercosur_crops"
    ls -la "$HOME/.cache/mercosur_crops/" | head -10
fi

if [ -d "$HOME/.cache/crpd_crops" ]; then
    echo "  ✓ CRPD crops: $HOME/.cache/crpd_crops"
    ls -la "$HOME/.cache/crpd_crops/" | head -10
fi

echo ""
echo "Original Datasets:"
if [ -d "$HOME/.cache/coco_text" ]; then
    echo "  ✓ COCO Text: $HOME/.cache/coco_text"
fi

if [ -d "$HOME/.cache/roboflow_lpr_dataset" ]; then
    echo "  ✓ Roboflow LPR: $HOME/.cache/roboflow_lpr_dataset"
fi

if [ -d "$HOME/.cache/indian_plates_kaggle" ]; then
    echo "  ✓ Indian Plates Kaggle: $HOME/.cache/indian_plates_kaggle"
fi

if [ -d "$HOME/.cache/kaggle_lp_detection" ]; then
    echo "  ✓ Kaggle LP Detection: $HOME/.cache/kaggle_lp_detection"
fi

if [ -d "$HOME/.cache/CCPD2019" ]; then
    echo "  ✓ CCPD2019: $HOME/.cache/CCPD2019"
fi

if [ -d "$HOME/.cache/Mercosur" ]; then
    echo "  ✓ Mercosur: $HOME/.cache/Mercosur"
fi

if [ -d "$HOME/.cache/CRPD" ]; then
    echo "  ✓ CRPD: $HOME/.cache/CRPD"
fi

echo ""
echo "✓ Dataset download and extraction complete"
