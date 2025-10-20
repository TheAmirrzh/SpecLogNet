#!/bin/bash

# Complete training pipeline for SpecLogicNet - FULLY FIXED
# Usage: bash run.sh

set -e

echo "================================"
echo "SpecLogicNet Training Pipeline"
echo "================================"

# Configuration - OPTIMIZED
DATA_DIR="data/horn"
EXP_DIR="experiments/run_$(date +%Y%m%d_%H%M%S)"
BATCH_SIZE=128          # Reduced for better gradients
EPOCHS=30
LR=1e-3                # Higher initial LR with scheduler
HIDDEN_DIM=256         # Smaller model for dataset size
NUM_LAYERS=4
DROPOUT=0.3            # Increased dropout
DEVICE="mps"

# Step 1: Generate dataset
echo ""
echo "Step 1: Generating dataset..."
python data_generator.py

# Step 2: Train model
echo ""
echo "Step 2: Training model..."
python train.py \
  --data-dir $DATA_DIR \
  --exp-dir $EXP_DIR \
  --epochs $EPOCHS \
  --batch-size $BATCH_SIZE \
  --lr $LR \
  --hidden-dim $HIDDEN_DIM \
  --num-layers $NUM_LAYERS \
  --dropout $DROPOUT \
  --device $DEVICE \
  --seed 42 \
  --use-type-aware


# Step 3: Results
echo ""
echo "================================"
echo "Training complete!"
echo "Results saved to: $EXP_DIR"
echo "================================"

if [ -f "$EXP_DIR/results.json" ]; then
    echo ""
    echo "Test Results:"
    cat "$EXP_DIR/results.json" | python -m json.tool | grep -A 5 "test_metrics"
fi