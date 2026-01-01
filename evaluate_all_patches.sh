#!/bin/bash

# Script to evaluate all trained patch variants using analyze_patch.py
# Runs digital evaluation on ALPR detection for all patches

set -e  # Exit on error

# Check if patch_variants directory exists
if [ ! -d "patch_variants_"* ]; then
    echo "Error: No patch_variants_* directory found!"
    echo "Usage: Run this script in the directory containing patch_variants_TIMESTAMP/"
    exit 1
fi

# Find the most recent patch_variants directory
PATCH_VARIANTS_DIR=$(ls -dt patch_variants_* 2>/dev/null | head -1)

if [ -z "$PATCH_VARIANTS_DIR" ]; then
    echo "Error: No patch_variants directory found!"
    exit 1
fi

echo "=========================================="
echo "Evaluating all patches in: $PATCH_VARIANTS_DIR"
echo "=========================================="

# Configuration
CSV_PATH="preproc_labels.csv"
DEVICE="cuda"
OUTPUT_BASE="evaluation_results_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$OUTPUT_BASE"

# Counter for progress
total_variants=$(find "$PATCH_VARIANTS_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
current=0

# Process each patch variant directory
for variant_dir in "$PATCH_VARIANTS_DIR"/patches_*; do
    if [ ! -d "$variant_dir" ]; then
        continue
    fi

    current=$((current + 1))
    variant_name=$(basename "$variant_dir")

    echo ""
    echo "=========================================="
    echo "[$current/$total_variants] Processing: $variant_name"
    echo "=========================================="

    # Find the best patch file
    best_patch_dir="$variant_dir/best_patches"
    if [ ! -d "$best_patch_dir" ]; then
        echo "⚠️  Warning: No best_patches directory found in $variant_name, skipping..."
        continue
    fi

    # Get the latest/best patch (highest epoch number)
    patch_file=$(ls -t "$best_patch_dir"/patch_epoch_*.png 2>/dev/null | head -1)

    if [ -z "$patch_file" ]; then
        echo "⚠️  Warning: No patch PNG found in $variant_name/best_patches/, skipping..."
        continue
    fi

    echo "Using patch: $(basename "$patch_file")"

    # Determine impersonation target from variant name
    impersonation_target=""
    if [[ "$variant_name" == *"VJJ7744"* ]]; then
        impersonation_target="VJJ7744"
    elif [[ "$variant_name" == *"SHX8459"* ]]; then
        impersonation_target="SHX8459"
    fi

    # Set output directory for this variant
    output_dir="$OUTPUT_BASE/$variant_name"

    # Build command
    cmd="python analyze_patch.py --csv \"$CSV_PATH\" --patch \"$patch_file\" --output \"$output_dir\" --device $DEVICE"

    if [ -n "$impersonation_target" ]; then
        cmd="$cmd --impersonating-plate $impersonation_target"
        echo "Target plate: $impersonation_target"
    fi

    echo "Command: $cmd"
    echo ""

    # Run evaluation
    if eval $cmd; then
        echo "✓ Completed: $variant_name"

        # Copy variant config to results
        if [ -f "$variant_dir/config.txt" ]; then
            cp "$variant_dir/config.txt" "$output_dir/"
        fi
    else
        echo "✗ Failed: $variant_name"
    fi
done

# Create summary
echo ""
echo "=========================================="
echo "EVALUATION COMPLETE"
echo "=========================================="
echo "All results saved to: $OUTPUT_BASE/"
echo ""
echo "Directory structure:"
tree -L 2 "$OUTPUT_BASE" 2>/dev/null || ls -R "$OUTPUT_BASE"

echo ""
echo "=========================================="
echo "Summary of evaluations:"
echo "=========================================="
for result_dir in "$OUTPUT_BASE"/patches_*; do
    if [ -f "$result_dir/results.csv" ]; then
        variant_name=$(basename "$result_dir")
        echo "✓ $variant_name"
    fi
done

echo ""
echo "To compare results, check the CSV files in each directory:"
echo "  $OUTPUT_BASE/patches_*/results.csv"
