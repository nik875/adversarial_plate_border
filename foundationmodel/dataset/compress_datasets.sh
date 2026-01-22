#!/bin/bash
# Compress downloaded datasets into a single tarball
# Output: opensourcedata.tar

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}"
OUTPUT_FILE="${OUTPUT_DIR}/opensourcedata.tar"

echo "Compressing datasets..."
echo "Output: $OUTPUT_FILE"

# Create temporary directory for staging
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# Create cache directory structure in temp
mkdir -p "$TEMP_DIR/.cache/iiit5k"
mkdir -p "$TEMP_DIR/.cache/huggingface/datasets"
mkdir -p "$TEMP_DIR/.cache/icdar2011"
mkdir -p "$TEMP_DIR/.cache/icdar2013"
mkdir -p "$TEMP_DIR/.cache/cocotext_crops"

# Copy IIIT5K dataset (if exists)
if [ -d "$HOME/.cache/iiit5k" ]; then
    echo "Copying IIIT5K dataset..."
    cp -r "$HOME/.cache/iiit5k"/* "$TEMP_DIR/.cache/iiit5k/" 2>/dev/null || true
else
    echo "Warning: IIIT5K dataset not found at $HOME/.cache/iiit5k"
fi

# Copy Hugging Face datasets (if exists)
if [ -d "$HOME/.cache/huggingface/datasets" ]; then
    echo "Copying Hugging Face datasets..."
    cp -r "$HOME/.cache/huggingface/datasets"/* "$TEMP_DIR/.cache/huggingface/datasets/" 2>/dev/null || true
else
    echo "Warning: Hugging Face datasets not found at $HOME/.cache/huggingface/datasets"
fi

# Copy ICDAR 2011 dataset (2013 Challenge 1) (if exists)
if [ -d "$HOME/.cache/icdar2011" ]; then
    echo "Copying ICDAR 2011 dataset (train & test)..."
    cp -r "$HOME/.cache/icdar2011"/* "$TEMP_DIR/.cache/icdar2011/" 2>/dev/null || true
else
    echo "Warning: ICDAR 2011 dataset not found at $HOME/.cache/icdar2011"
fi

# Copy ICDAR 2013 dataset (2015 Challenge 2) (if exists)
if [ -d "$HOME/.cache/icdar2013" ]; then
    echo "Copying ICDAR 2013 dataset (train & test)..."
    cp -r "$HOME/.cache/icdar2013"/* "$TEMP_DIR/.cache/icdar2013/" 2>/dev/null || true
else
    echo "Warning: ICDAR 2013 dataset not found at $HOME/.cache/icdar2013"
fi

# Copy COCO Text crops (if exists)
if [ -d "$HOME/.cache/cocotext_crops" ]; then
    echo "Copying COCO Text crops..."
    cp -r "$HOME/.cache/cocotext_crops"/* "$TEMP_DIR/.cache/cocotext_crops/" 2>/dev/null || true
else
    echo "Warning: COCO Text crops not found at $HOME/.cache/cocotext_crops"
fi

# Create tarball
echo "Creating tarball..."
cd "$TEMP_DIR"
tar -cf "$OUTPUT_FILE" .cache/
cd - > /dev/null

echo "✓ Dataset compression complete: $OUTPUT_FILE"
ls -lh "$OUTPUT_FILE"
