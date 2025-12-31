#!/bin/bash

# Script to train all patch variants with different combinations of:
# - TV loss (enabled/disabled)
# - Homography (enabled/disabled)
# - Impersonation target (VJJ7744/SHX8459)

set -e  # Exit on error

# Create main output directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_BASE="patch_variants_${TIMESTAMP}"
mkdir -p "$OUTPUT_BASE"

echo "Starting adversarial patch training for all variants..."
echo "This will create 8 different patch sets (2 targets × 2 TV settings × 2 homography settings)"
echo "All outputs will be saved to: $OUTPUT_BASE/"
echo "=========================================="

# Base command - always includes --match-detection
BASE_CMD="python offensive_patch.py --match-detection"

# Array to track which runs have completed
declare -a completed_runs

# Function to run training and organize output
run_variant() {
    local target=$1
    local tv_flag=$2
    local homo_flag=$3
    local desc=$4

    echo ""
    echo "=========================================="
    echo "Running: $desc"
    echo "Target: $target"
    echo "TV Loss: $([ -z "$tv_flag" ] && echo "ENABLED" || echo "DISABLED")"
    echo "Homography: $([ -z "$homo_flag" ] && echo "ENABLED" || echo "DISABLED")"
    echo "=========================================="

    # Construct command
    cmd="$BASE_CMD --impersonation-target $target $tv_flag $homo_flag"

    echo "Command: $cmd"
    echo ""

    # Run training
    if $cmd; then
        # Create output directory name
        variant_name="patches_${target}"
        [ -n "$tv_flag" ] && variant_name="${variant_name}_notv"
        [ -n "$homo_flag" ] && variant_name="${variant_name}_nohomo"

        output_dir="$OUTPUT_BASE/$variant_name"

        # Move outputs to organized directory
        echo "Training completed successfully!"
        echo "Moving outputs to $output_dir/"

        # Create output directory
        mkdir -p "$output_dir"

        # Move patch directories if they exist
        [ -d "best_patches" ] && mv best_patches "$output_dir/"
        [ -d "checkpoint_patches" ] && mv checkpoint_patches "$output_dir/"

        # Move training results if they exist
        [ -f "adversarial_training_results.png" ] && mv adversarial_training_results.png "$output_dir/"

        # Create a metadata file
        cat > "$output_dir/config.txt" <<EOF
Target: $target
TV Loss: $([ -z "$tv_flag" ] && echo "ENABLED" || echo "DISABLED")
Homography: $([ -z "$homo_flag" ] && echo "ENABLED" || echo "DISABLED")
Command: $cmd
Timestamp: $(date)
EOF

        completed_runs+=("$desc")
        echo "✓ Completed: $desc"
    else
        echo "✗ Failed: $desc"
        return 1
    fi
}

# Run all 8 combinations
# Format: run_variant <target> <tv_flag> <homo_flag> <description>

echo "Starting batch training..."

# VJJ7744 variants
run_variant "VJJ7744" "" "" "VJJ7744 - Full (TV + Homography)"
run_variant "VJJ7744" "--disable-tv-loss" "" "VJJ7744 - No TV Loss"
run_variant "VJJ7744" "" "--disable-homography" "VJJ7744 - No Homography"
run_variant "VJJ7744" "--disable-tv-loss" "--disable-homography" "VJJ7744 - No TV + No Homography"

# SHX8459 variants
run_variant "SHX8459" "" "" "SHX8459 - Full (TV + Homography)"
run_variant "SHX8459" "--disable-tv-loss" "" "SHX8459 - No TV Loss"
run_variant "SHX8459" "" "--disable-homography" "SHX8459 - No Homography"
run_variant "SHX8459" "--disable-tv-loss" "--disable-homography" "SHX8459 - No TV + No Homography"

# Summary
echo ""
echo "=========================================="
echo "TRAINING COMPLETE"
echo "=========================================="
echo "Completed runs (${#completed_runs[@]}/8):"
for run in "${completed_runs[@]}"; do
    echo "  ✓ $run"
done

echo ""
echo "All outputs saved to: $OUTPUT_BASE/"
echo ""
echo "Directory structure:"
tree -L 2 "$OUTPUT_BASE" 2>/dev/null || ls -R "$OUTPUT_BASE"

echo ""
echo "=========================================="
echo "To export all results, simply copy/compress this directory:"
echo "  tar -czf patch_variants.tar.gz $OUTPUT_BASE/"
echo "  or: zip -r patch_variants.zip $OUTPUT_BASE/"
echo "=========================================="
