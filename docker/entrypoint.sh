#!/bin/bash
# MABE Pipeline Docker Entrypoint

set -e

# Default values
CONFIG_FILE=${CONFIG_FILE:-"configs/default.yaml"}
COMMAND=${COMMAND:-"help"}
MAX_VIDEOS=${MAX_VIDEOS:-5}
EPOCHS=${EPOCHS:-30}
BATCH_SIZE=${BATCH_SIZE:-32}
DEVICE=${DEVICE:-"cpu"}

echo "MABE Pipeline Docker Container"
echo "=============================="
echo "Config: $CONFIG_FILE"
echo "Command: $COMMAND"
echo "Max Videos: $MAX_VIDEOS"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Device: $DEVICE"
echo ""

# Create output directories
mkdir -p outputs/models outputs/submissions outputs/logs outputs/studies

# Run the specified command
case $COMMAND in
    "preprocess")
        echo "Running data preprocessing..."
        python bin/run_pipeline.py preprocess \
            --config $CONFIG_FILE \
            --override training.max_videos=$MAX_VIDEOS \
            --device $DEVICE
        ;;
    "train")
        echo "Running model training..."
        python bin/run_pipeline.py train \
            --config $CONFIG_FILE \
            --override training.epochs=$EPOCHS \
            --override training.batch_size=$BATCH_SIZE \
            --override training.max_videos=$MAX_VIDEOS \
            --device $DEVICE
        ;;
    "tune")
        echo "Running hyperparameter tuning..."
        python bin/run_pipeline.py tune \
            --config $CONFIG_FILE \
            --override training.max_videos=$MAX_VIDEOS \
            --device $DEVICE
        ;;
    "infer")
        echo "Running inference..."
        python bin/run_pipeline.py infer \
            --config $CONFIG_FILE \
            --device $DEVICE
        ;;
    "evaluate")
        echo "Running evaluation..."
        python bin/run_pipeline.py evaluate \
            --config $CONFIG_FILE \
            --predictions outputs/submissions/submission_*.csv
        ;;
    "all")
        echo "Running full pipeline..."
        python bin/run_pipeline.py all \
            --config $CONFIG_FILE \
            --override training.max_videos=$MAX_VIDEOS \
            --override training.epochs=$EPOCHS \
            --override training.batch_size=$BATCH_SIZE \
            --device $DEVICE
        ;;
    "ci")
        echo "Running CI test suite..."
        ./scripts/ci_run_short.sh
        ;;
    "help"|*)
        echo "Available commands:"
        echo "  preprocess  - Run data preprocessing"
        echo "  train       - Train models"
        echo "  tune        - Run hyperparameter tuning"
        echo "  infer       - Run inference"
        echo "  evaluate    - Run evaluation"
        echo "  all         - Run full pipeline"
        echo "  ci          - Run CI test suite"
        echo "  help        - Show this help"
        echo ""
        echo "Environment variables:"
        echo "  CONFIG_FILE - Configuration file path (default: configs/default.yaml)"
        echo "  COMMAND     - Command to run (default: help)"
        echo "  MAX_VIDEOS  - Maximum videos to process (default: 5)"
        echo "  EPOCHS      - Number of training epochs (default: 30)"
        echo "  BATCH_SIZE  - Training batch size (default: 32)"
        echo "  DEVICE      - Device to use (default: cpu)"
        echo ""
        echo "Example usage:"
        echo "  docker run -e COMMAND=train -e EPOCHS=10 mabe-pipeline"
        echo "  docker run -e COMMAND=all -e MAX_VIDEOS=2 mabe-pipeline"
        ;;
esac
