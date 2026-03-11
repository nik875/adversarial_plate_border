#!/usr/bin/env bash
# =============================================================================
# run_experiments.sh
#
# Trains one adversarial patch per detector model and one per OCR model, then
# runs the cross-model evaluator on all produced patches.
#
# Usage
# -----
#   chmod +x run_experiments.sh
#   ./run_experiments.sh                        # all defaults
#   ./run_experiments.sh --device mps           # override device
#   ./run_experiments.sh --epochs 50 --skip-ocr # faster, skip OCR sweep
#
# Outputs
# -------
#   patches/                   all saved patch checkpoints
#   results/detector_sweep/    evaluator output for detector patches
#   results/ocr_sweep/         evaluator output for OCR patches
#   logs/                      per-run stdout/stderr logs
# =============================================================================

set -euo pipefail

# ── Defaults (override via CLI flags below) ───────────────────────────────────
DEVICE="cuda"
CSV="updated_control_corners.csv"
EPOCHS=100
LR=0.1
GRAD_ACCUM=64
NUM_WORKERS=0
PIN_MEMORY=false
PRELOAD_IMAGES=false
LIMIT=0
PATCH_DIR="patches"
LOG_DIR="logs"
RESULTS_DIR="results"

# ── Weight paths — edit these to match your local files ───────────────────────
YOLOV8_WEIGHTS="weights/lp_yolov8.pt"
FASTERRCNN_WEIGHTS="weights/model.pt"
YOLOV11_WEIGHTS="weights/yolov11s-license-plate.pt"   # or set download_hf=True in code
RTDETR_WEIGHTS="weights/rtdetr-v2-license-plate"

# OCR weights / sources
# - crnn and dtrb use local checkpoints
# - trocr and fastanpr-ocr use "none"
CRNN_WEIGHTS="weights/crnn_synth90k.pt"
DTRB_WEIGHTS="weights/vitstr_small_patch16_224.pth"
DTRB_ROOT="/home/ubuntu/deep-text-recognition-benchmark"   # Path to DTRB repo for model definitions

# Default OCR backend used during detector sweep
# Use "fastanpr-ocr" or "trocr" if you don't have CRNN weights yet
DEFAULT_OCR="crnn"
DEFAULT_OCR_WEIGHTS="$CRNN_WEIGHTS"

# Default detector used during OCR sweep
DEFAULT_DET="yolov8"
DEFAULT_DET_WEIGHTS="$YOLOV8_WEIGHTS"

# ── Flags ─────────────────────────────────────────────────────────────────────
SKIP_OCR=false
SKIP_TRAIN=false    # set true to jump straight to evaluation (re-use existing patches)
DRY_RUN=false

# ── Parse CLI args ─────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)       DEVICE="$2";       shift 2 ;;
        --epochs)       EPOCHS="$2";       shift 2 ;;
        --lr)           LR="$2";           shift 2 ;;
        --csv)          CSV="$2";          shift 2 ;;
        --num-workers)  NUM_WORKERS="$2";  shift 2 ;;
        --pin-memory)   PIN_MEMORY=true;    shift   ;;
        --preload-images) PRELOAD_IMAGES=true; shift ;;
        --limit)        LIMIT="$2";        shift 2 ;;
        --dtrb-root)    DTRB_ROOT="$2";    shift 2 ;;
        --skip-ocr)     SKIP_OCR=true;     shift   ;;
        --skip-train)   SKIP_TRAIN=true;   shift   ;;
        --dry-run)      DRY_RUN=true;      shift   ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Helpers ───────────────────────────────────────────────────────────────────
mkdir -p "$PATCH_DIR" "$LOG_DIR" \
         "$RESULTS_DIR/detector_sweep" \
         "$RESULTS_DIR/ocr_sweep"

# Print with timestamp
log() { echo "[$(date '+%H:%M:%S')] $*"; }

# Run a command, tee output to a log file, exit on failure
run() {
    local label="$1"; shift
    local logfile="$LOG_DIR/${label}.log"
    log "START  $label"
    if $DRY_RUN; then
        echo "  DRY-RUN: $*"
    else
        "$@" 2>&1 | tee "$logfile"
        if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
            log "FAILED $label — see $logfile"
            exit 1
        fi
    fi
    log "DONE   $label"
}

# Find the best (latest-epoch) patch for a given backend name
best_patch() {
    local name="$1"
    local candidates=()
    local d f

    # Search all known trainer output dirs.
    for d in "$PATCH_DIR" "best_patches" "checkpoint_patches"; do
        [[ -d "$d" ]] || continue

        # Prefer explicit best file if present.
        if [[ -f "$d/patch_${name}_best.pt" ]]; then
            echo "$d/patch_${name}_best.pt"
            return 0
        fi

        # Collect epoch checkpoints.
        for f in "$d"/patch_${name}_epoch_*.pt; do
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

# ── Training base args ─────────────────────────────────────────────────────────
BASE_TRAIN_ARGS=(
    --csv "$CSV"
    --device "$DEVICE"
    --epochs "$EPOCHS"
    --lr "$LR"
    --grad-accumulate "$GRAD_ACCUM"
    --num-workers "$NUM_WORKERS"
)

if $PIN_MEMORY; then
    BASE_TRAIN_ARGS+=(--pin-memory)
fi
if $PRELOAD_IMAGES; then
    BASE_TRAIN_ARGS+=(--preload-images)
fi
if [[ "$LIMIT" -gt 0 ]]; then
    BASE_TRAIN_ARGS+=(--limit "$LIMIT")
fi

# =============================================================================
# PHASE 1 — Detector sweep
# One patch per detector, using the default OCR backend throughout.
# Goal: compare how effective the patch algorithm is across different detectors.
# =============================================================================

log "━━━━  PHASE 1: Detector sweep  ━━━━"

DETECTOR_RUNS=(
    "yolov8:$YOLOV8_WEIGHTS"
    "fasterrcnn:$FASTERRCNN_WEIGHTS"
    "yolov11:$YOLOV11_WEIGHTS"
    "rtdetr:$RTDETR_WEIGHTS"
)

if ! $SKIP_TRAIN; then
    if [[ "$DEFAULT_OCR_WEIGHTS" != "none" && ! -f "$DEFAULT_OCR_WEIGHTS" ]]; then
        log "SKIP   detector sweep training — default OCR '$DEFAULT_OCR' weights not found at '$DEFAULT_OCR_WEIGHTS'"
    else
    for entry in "${DETECTOR_RUNS[@]}"; do
        det="${entry%%:*}"
        wts="${entry##*:}"

        if [[ "$wts" != "none" && ! -e "$wts" ]]; then
            log "SKIP   $det — weights not found at '$wts'"
            continue
        fi

        run "train_det_${det}" \
            python trainer.py \
                "${BASE_TRAIN_ARGS[@]}" \
                --backend        "$det" \
                --model-path     "$wts" \
                --ocr-backend    "$DEFAULT_OCR" \
                --ocr-model-path "$DEFAULT_OCR_WEIGHTS"
    done
    fi
fi

# Collect all detector patches that exist
DETECTOR_PATCHES=()
DETECTOR_BACKEND_SPECS=()
for entry in "${DETECTOR_RUNS[@]}"; do
    det="${entry%%:*}"
    wts="${entry##*:}"
    patch=$(best_patch "$det")
    if [[ -n "$patch" && ( "$wts" == "none" || -e "$wts" ) ]]; then
        DETECTOR_PATCHES+=("$patch")
        DETECTOR_BACKEND_SPECS+=("${det}:${wts}")
        log "Found patch for $det: $patch"
    else
        log "No patch found for $det — skipping from eval"
    fi
done

if [[ ${#DETECTOR_PATCHES[@]} -gt 0 ]]; then
    log "Running cross-model evaluation for detector sweep..."
    run "eval_detector_sweep" \
        python evaluator.py \
            --csv       "$CSV" \
            --device    "$DEVICE" \
            --backends  "${DETECTOR_BACKEND_SPECS[@]}" \
            --patches   "${DETECTOR_PATCHES[@]}" \
            --output    "$RESULTS_DIR/detector_sweep"
else
    log "No detector patches found — skipping detector sweep evaluation"
fi

# =============================================================================
# PHASE 2 — OCR sweep
# One patch per OCR model (against the default detector).
# Goal: compare OCR backend sensitivity under the same detector.
# =============================================================================

log "━━━━  PHASE 2: OCR sweep  ━━━━"

OCR_RUNS=(
    "crnn:$CRNN_WEIGHTS"
    "trocr:none"
    "dtrb:$DTRB_WEIGHTS"
    "fastanpr-ocr:none"
)

if ! $SKIP_TRAIN && ! $SKIP_OCR; then
    if [[ ! -f "$DEFAULT_DET_WEIGHTS" ]]; then
        log "SKIP   OCR sweep — default detector weights not found at '$DEFAULT_DET_WEIGHTS'"
    else
        for entry in "${OCR_RUNS[@]}"; do
            ocr_name="${entry%%:*}"
            ocr_wts="${entry##*:}"

            if [[ "$ocr_wts" != "none" && ! -f "$ocr_wts" ]]; then
                log "SKIP   ocr=$ocr_name — weights not found at '$ocr_wts'"
                continue
            fi

            OCR_EXTRA_ARGS=()
            if [[ "$ocr_name" == "dtrb" && -n "$DTRB_ROOT" ]]; then
                OCR_EXTRA_ARGS+=(--ocr-repo-root "$DTRB_ROOT")
            fi
            if [[ "$ocr_name" == "dtrb" ]]; then
                # ViTSTR-specific configuration
                OCR_EXTRA_ARGS+=(--dtrb-feature-extraction "vitstr_small_patch16_224")
                OCR_EXTRA_ARGS+=(--dtrb-sequence-modeling "None")
                OCR_EXTRA_ARGS+=(--dtrb-transformation "None")
            fi

            # Train one OCR variant against DEFAULT_DET.
            run "train_ocr_${ocr_name}" \
                python trainer.py \
                    "${BASE_TRAIN_ARGS[@]}" \
                    --backend        "$DEFAULT_DET" \
                    --model-path     "$DEFAULT_DET_WEIGHTS" \
                    --ocr-backend    "$ocr_name" \
                    --ocr-model-path "$ocr_wts" \
                    "${OCR_EXTRA_ARGS[@]}"

            # Stage the newest/default-det patch under an OCR-specific name.
            latest_patch=$(best_patch "$DEFAULT_DET")
            if [[ -z "$latest_patch" ]]; then
                log "WARN   no patch artifact found for ${DEFAULT_DET} after ocr=${ocr_name}"
                continue
            fi

            base_name=$(basename "$latest_patch")
            staged_name="${base_name/${DEFAULT_DET}_/${DEFAULT_DET}_ocr-${ocr_name}_}"
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
    fi
fi

# Collect OCR sweep patches
OCR_PATCHES=()
for entry in "${OCR_RUNS[@]}"; do
    ocr_name="${entry%%:*}"
    ocr_wts="${entry##*:}"
    patch=$(best_patch "${DEFAULT_DET}_ocr-${ocr_name}")
    if [[ -n "$patch" ]]; then
        OCR_PATCHES+=("$patch")
        log "Found OCR patch for ocr=$ocr_name: $patch"
    fi
done

if [[ ${#OCR_PATCHES[@]} -gt 0 && -f "$DEFAULT_DET_WEIGHTS" ]]; then
    log "Running cross-OCR evaluation..."
    run "eval_ocr_sweep" \
        python evaluator.py \
            --csv      "$CSV" \
            --device   "$DEVICE" \
            --backends "${DEFAULT_DET}:${DEFAULT_DET_WEIGHTS}" \
            --patches  "${OCR_PATCHES[@]}" \
            --output   "$RESULTS_DIR/ocr_sweep"
else
    log "No OCR sweep patches found — skipping OCR sweep evaluation"
fi

# =============================================================================
# Done
# =============================================================================

log "━━━━  All experiments complete  ━━━━"
echo ""
echo "Results:"
echo "  Detector sweep : $RESULTS_DIR/detector_sweep/"
echo "    metrics.csv"
echo "    matrix_recall.png   ← cross-model heatmap"
echo "    matrix_attack_drop.png"
echo "    bar_chart.png"
echo ""
echo "  OCR sweep      : $RESULTS_DIR/ocr_sweep/"
echo "    metrics.csv"
echo "    bar_chart.png       ← clean vs each OCR variant"
echo ""
echo "  Logs           : $LOG_DIR/"
echo "  Patches        : $PATCH_DIR/"
