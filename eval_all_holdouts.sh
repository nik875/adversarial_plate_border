#!/usr/bin/env bash
set -e

FINETUNED_MODELS="finetuned_models"
DEVICE="cuda"
OUTPUT_BASE="results/holdout_eval"

declare -A RUNS=(
    [fasterrcnn]="runs/holdout_fasterrcnn_lprnet_20260403_214721"
    [rtdetr]="runs/holdout_rtdetr_doctr-vitstr_20260403_214832"
    [owlvit]="runs/holdout_owlvit_trocr_20260403_214846"
    [yolo-v9-608]="runs/holdout_yolo-v9-608_cct_20260403_214615"
)

for holdout in fasterrcnn rtdetr owlvit yolo-v9-608; do
    run_dir="${RUNS[$holdout]}"
    patch="$run_dir/patches/patch_ensemble_best.png"
    output="$OUTPUT_BASE/holdout_$holdout"

    echo "======================================================"
    echo "Evaluating holdout=$holdout"
    echo "  patch  : $patch"
    echo "  output : $output"
    echo "======================================================"

    python evaluate_finetuned.py \
        --finetuned-models "$FINETUNED_MODELS" \
        --patch "$patch" \
        --device "$DEVICE" \
        --output "$output"
done

echo ""
echo "All done. Results in $OUTPUT_BASE/"
