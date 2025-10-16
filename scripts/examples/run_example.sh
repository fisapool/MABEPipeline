#!/bin/bash
# MABE Pipeline Example Script
# Demonstrates end-to-end pipeline execution

set -e  # Exit on any error

echo "MABE Pipeline Example - End-to-End Demo"
echo "========================================"

# Configuration
CONFIG_FILE="configs/default.yaml"
OUTPUT_DIR="outputs"
LOG_DIR="outputs/logs"

# Create output directories
mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR

echo "Step 1: Data Preprocessing"
echo "-------------------------"
python bin/run_pipeline.py preprocess \
    --config $CONFIG_FILE \
    --override training.max_videos=2 \
    --verbose

echo ""
echo "Step 2: Model Training"
echo "----------------------"
python bin/run_pipeline.py train \
    --config $CONFIG_FILE \
    --override training.epochs=5 \
    --override training.batch_size=16 \
    --verbose

echo ""
echo "Step 3: Hyperparameter Tuning (Quick)"
echo "--------------------------------------"
python bin/run_pipeline.py tune \
    --config $CONFIG_FILE \
    --override optuna.n_trials=10 \
    --override optuna.ci_mode=true \
    --verbose

echo ""
echo "Step 4: Model Inference"
echo "-----------------------"
python bin/run_pipeline.py infer \
    --config $CONFIG_FILE \
    --verbose

echo ""
echo "Step 5: Evaluation (if ground truth available)"
echo "----------------------------------------------"
if [ -f "outputs/submissions/submission_*.csv" ]; then
    python bin/run_pipeline.py evaluate \
        --config $CONFIG_FILE \
        --predictions outputs/submissions/submission_*.csv \
        --verbose
else
    echo "No submission file found, skipping evaluation"
fi

echo ""
echo "Example Pipeline Completed Successfully!"
echo "========================================"
echo "Check the following directories for outputs:"
echo "  - Models: $OUTPUT_DIR/models/"
echo "  - Submissions: $OUTPUT_DIR/submissions/"
echo "  - Logs: $LOG_DIR/"
echo "  - Studies: $OUTPUT_DIR/studies/"
