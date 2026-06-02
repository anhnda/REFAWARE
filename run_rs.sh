#!/bin/bash

LOG_FILE="rs_all.log"
OUT_DIR="out"

> "$LOG_FILE"
mkdir -p "$OUT_DIR"

for img in benchmark_50/*.JPEG; do
    echo "========================================" | tee -a "$LOG_FILE"
    echo "Running: $img" | tee -a "$LOG_FILE"
    echo "Start: $(date)" | tee -a "$LOG_FILE"

    python real_select.py \
        --image "$img" \
        --model resnet50 \
        --device cuda \
        --grid 12 \
        --n-samples 2000 \
        --insdel \
        --out-dir "$OUT_DIR" \
        >> "$LOG_FILE" 2>&1

    echo "Finished: $img" | tee -a "$LOG_FILE"
    echo "End: $(date)" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
done