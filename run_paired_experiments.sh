#!/usr/bin/env bash
# =============================================================================
# run_paired_experiments.sh
#
# Train one adversarial patch per explicit detector+OCR pairing.
#
# Pairings configured by default:
#   1) fasterrcnn + doctr-vitstr (fine-tuned vitstr_small)
#   2) yolo-v9-384 + cct (FastALPR / open-image-models + fast-plate-ocr)
#   3) rtdetr     + trocr
#   4) yolov8     + lprnet
#
# Usage:
#   chmod +x run_paired_experiments.sh
#   ./run_paired_experiments.sh
#   ./run_paired_experiments.sh --epochs 20 --device cuda --limit 64
#   ./run_paired_experiments.sh --dry-run   # sanity check + debug images only, no training
# =============================================================================

set -euo pipefail

# Defaults
DEVICE="cuda"
CSV="preproc_labels.csv"
EPOCHS=100
LR=0.1
GRAD_ACCUM=64
NUM_WORKERS=$(nproc)
PIN_MEMORY=false
PRELOAD_IMAGES=false
LIMIT=0
DRY_RUN=false

PATCH_DIR="patches"
LOG_DIR="logs"

# Model paths
YOLOV8_WEIGHTS="weights/lp_yolov8.pt"
FASTERRCNN_WEIGHTS="weights/model.pt"
RTDETR_WEIGHTS="weights/rtdetr-v2-license-plates"
RTDETR_WEIGHTS_FALLBACK="weights/rtdetr-v2-license-plate"
YOLOV9_384_WEIGHTS="~/.cache/open-image-models/yolo-v9-t-384-license-plate-end2end/yolo-v9-t-384-license-plates-end2end.onnx"
CCT_WEIGHTS="~/.cache/fast-plate-ocr/cct-s-v1-global-model/cct_s_v1_global.onnx"

TROCR_WEIGHTS="none"
LPRNET_WEIGHTS="us_lprnet_patched.onnx"
DOCTR_VITSTR_WEIGHTS="weights/vitstr_small_finetuned.pt"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)          DEVICE="$2"; shift 2 ;;
        --csv)             CSV="$2"; shift 2 ;;
        --epochs)          EPOCHS="$2"; shift 2 ;;
        --lr)              LR="$2"; shift 2 ;;
        --grad-accumulate) GRAD_ACCUM="$2"; shift 2 ;;
        --num-workers)     NUM_WORKERS="$2"; shift 2 ;;
        --pin-memory)      PIN_MEMORY=true; shift ;;
        --preload-images)  PRELOAD_IMAGES=true; shift ;;
        --limit)           LIMIT="$2"; shift 2 ;;
        --dry-run)         DRY_RUN=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

mkdir -p "$PATCH_DIR" "$LOG_DIR" best_patches checkpoint_patches

# Keep requested RT-DETR path, but allow legacy singular folder name.
if [[ ! -e "$RTDETR_WEIGHTS" && -e "$RTDETR_WEIGHTS_FALLBACK" ]]; then
    RTDETR_WEIGHTS="$RTDETR_WEIGHTS_FALLBACK"
fi

log() { echo "[$(date '+%H:%M:%S')] $*"; }

run() {
    local label="$1"; shift
    local logfile="$LOG_DIR/${label}.log"
    log "START  $label"
    if $DRY_RUN; then
        # Pass --dry-run through to trainer (sanity check + debug output only, no training)
        "$@" --dry-run 2>&1 | tee "$logfile"
        if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
            log "FAILED $label — see $logfile"
            exit 1
        fi
    else
        "$@" 2>&1 | tee "$logfile"
        if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
            log "FAILED $label — see $logfile"
            exit 1
        fi
    fi
    log "DONE   $label"
}

# Find most recent patch artifact for a detector backend.
best_patch() {
    local det="$1"
    local candidates=()
    local d f

    for d in "$PATCH_DIR" "best_patches" "checkpoint_patches"; do
        [[ -d "$d" ]] || continue

        if [[ -f "$d/patch_${det}_best.pt" ]]; then
            echo "$d/patch_${det}_best.pt"
            return 0
        fi

        for f in "$d"/patch_${det}_epoch_*.pt; do
            [[ -f "$f" ]] || continue
            candidates+=("$f")
        done
    done

    if [[ ${#candidates[@]} -gt 0 ]]; then
        printf "%s\n" "${candidates[@]}" | sort -V | tail -n1
    else
        echo ""
    fi
}

BASE_ARGS=(
    --csv "$CSV"
    --device "$DEVICE"
    --epochs "$EPOCHS"
    --lr "$LR"
    --grad-accumulate "$GRAD_ACCUM"
    --num-workers "$NUM_WORKERS"
)

if $PIN_MEMORY; then
    BASE_ARGS+=(--pin-memory)
fi
if $PRELOAD_IMAGES; then
    BASE_ARGS+=(--preload-images)
fi
if [[ "$LIMIT" -gt 0 ]]; then
    BASE_ARGS+=(--limit "$LIMIT")
fi

# Pair rows:
# label|detector|det_weights|ocr|ocr_weights
PAIRS=(
    "pair_fasterrcnn_doctr_vitstr|fasterrcnn|$FASTERRCNN_WEIGHTS|doctr-vitstr|$DOCTR_VITSTR_WEIGHTS"
    "pair_fastalpr|yolo-v9-384|$YOLOV9_384_WEIGHTS|cct|$CCT_WEIGHTS"
    "pair_rtdetr_trocr|rtdetr|$RTDETR_WEIGHTS|trocr|$TROCR_WEIGHTS"
    "pair_yolov8_lprnet|yolov8|$YOLOV8_WEIGHTS|lprnet|$LPRNET_WEIGHTS"
)

# Removed: pair_fasterrcnn_dtrb (dtrb/ViTSTR checkpoint backend has not worked)

log "━━━━  Paired Training  ━━━━"

for row in "${PAIRS[@]}"; do
    IFS='|' read -r label det det_w ocr ocr_w <<< "$row"

    if [[ "$det_w" != "none" && ! -e "$det_w" ]]; then
        log "SKIP   $label — detector weights not found: $det_w"
        continue
    fi
    if [[ "$ocr_w" != "none" && ! -e "$ocr_w" ]]; then
        log "SKIP   $label — OCR weights not found: $ocr_w"
        continue
    fi

    run "$label" \
        python trainer.py \
            "${BASE_ARGS[@]}" \
            --backend "$det" \
            --model-path "$det_w" \
            --ocr-backend "$ocr" \
            --ocr-model-path "$ocr_w" \
            "${EXTRA_ARGS[@]}"

    # Skip artifact staging in dry-run mode (no patches are saved)
    $DRY_RUN && continue

    latest_patch=$(best_patch "$det")
    if [[ -z "$latest_patch" ]]; then
        log "WARN   no patch artifact found after $label"
        continue
    fi

    tag="${det}__${ocr}"
    base_name=$(basename "$latest_patch")
    staged_name="${base_name/${det}_/${tag}_}"
    staged_patch="$PATCH_DIR/$staged_name"

    cp "$latest_patch" "$staged_patch"
    log "Staged $latest_patch → $staged_patch"

    latest_png="${latest_patch%.pt}.png"
    if [[ -f "$latest_png" ]]; then
        staged_png="$PATCH_DIR/${staged_name%.pt}.png"
        cp "$latest_png" "$staged_png"
        log "Staged $latest_png → $staged_png"
    fi
done

log "━━━━  Done  ━━━━"
echo "Saved paired patch artifacts under: $PATCH_DIR"
