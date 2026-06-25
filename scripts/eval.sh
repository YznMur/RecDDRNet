#!/bin/bash
# Multi-split evaluation (train, val, test) - single GPU
# Usage: ./scripts/eval.sh [val|train,val,test|all] [model_path]
# If model_path not provided, uses latest checkpoint from OUTPUT_DIR

SPLITS=${1:-val}
MODEL_FILE=${2:-""}

if [ -z "$MODEL_FILE" ]; then
    # Try to find latest checkpoint
    OUTPUT_DIR=$(grep OUTPUT_DIR configs/rsm_ddrnet23slim_convlstm.yaml | awk '{print $2}' | tr -d "'")
    if [ -d "$OUTPUT_DIR" ]; then
        MODEL_FILE=$(find "$OUTPUT_DIR" -name "checkpoint*.pth*" -type f | sort -V | tail -1)
    fi
fi

CMD="python3 tools/eval.py --cfg configs/rsm_ddrnet23slim_convlstm.yaml --splits $SPLITS"
if [ -n "$MODEL_FILE" ]; then
    CMD="$CMD --model-file $MODEL_FILE"
fi

echo "Running: $CMD"
eval $CMD