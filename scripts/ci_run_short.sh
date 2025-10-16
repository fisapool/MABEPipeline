#!/bin/bash
# MABE Pipeline CI Test Script
# Short test suite for continuous integration

set -e  # Exit on any error

echo "MABE Pipeline CI Test Suite"
echo "============================"

# Configuration for CI
CONFIG_FILE="configs/default.yaml"
OUTPUT_DIR="outputs"
LOG_DIR="outputs/logs"

# Create output directories
mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR

echo "Step 1: Configuration Validation"
echo "--------------------------------"
python -c "
import sys
sys.path.insert(0, 'src')
from mabe.utils.config import load_config
cfg = load_config('$CONFIG_FILE')
print('✓ Configuration loaded successfully')
print(f'  Dataset path: {cfg[\"dataset\"][\"path\"]}')
print(f'  Model type: {cfg[\"training\"][\"model_type\"]}')
print(f'  Device: {cfg[\"device\"][\"device_str\"]}')
"

echo ""
echo "Step 2: Unit Tests"
echo "------------------"
python -m pytest tests/ -v --tb=short

echo ""
echo "Step 3: Data Preprocessing (CI Mode)"
echo "-------------------------------------"
python bin/run_pipeline.py preprocess \
    --config $CONFIG_FILE \
    --override training.max_videos=1 \
    --override optuna.ci_mode=true \
    --verbose

echo ""
echo "Step 4: Model Training (Quick)"
echo "------------------------------"
python bin/run_pipeline.py train \
    --config $CONFIG_FILE \
    --override training.epochs=2 \
    --override training.batch_size=8 \
    --override training.max_videos=1 \
    --verbose

echo ""
echo "Step 5: Hyperparameter Tuning (CI Mode)"
echo "----------------------------------------"
python bin/run_pipeline.py tune \
    --config $CONFIG_FILE \
    --override optuna.n_trials=3 \
    --override optuna.ci_mode=true \
    --override training.max_videos=1 \
    --verbose

echo ""
echo "Step 6: Model Inference (Quick)"
echo "--------------------------------"
python bin/run_pipeline.py infer \
    --config $CONFIG_FILE \
    --override training.max_videos=1 \
    --verbose

echo ""
echo "Step 7: Pipeline Validation"
echo "---------------------------"
# Check that outputs were created
if [ -d "$OUTPUT_DIR/models" ]; then
    echo "✓ Models directory created"
    ls -la $OUTPUT_DIR/models/
else
    echo "✗ Models directory not found"
    exit 1
fi

if [ -d "$OUTPUT_DIR/submissions" ]; then
    echo "✓ Submissions directory created"
    ls -la $OUTPUT_DIR/submissions/
else
    echo "✗ Submissions directory not found"
    exit 1
fi

if [ -d "$OUTPUT_DIR/logs" ]; then
    echo "✓ Logs directory created"
    ls -la $OUTPUT_DIR/logs/
else
    echo "✗ Logs directory not found"
    exit 1
fi

echo ""
echo "CI Test Suite Completed Successfully!"
echo "====================================="
echo "All tests passed ✓"
echo "Pipeline is ready for production use"
