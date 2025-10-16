# MABE Pipeline

A production-ready machine learning pipeline for the MABE (Mouse Behavior Analysis) competition. This pipeline provides a complete end-to-end solution for data preprocessing, model training, hyperparameter optimization, inference, and evaluation.

## Features

- **Centralized Configuration**: YAML-based configuration with CLI overrides
- **Modular Design**: Clean separation of concerns with reusable components
- **Reproducible**: Deterministic behavior with seed management
- **Backward Compatible**: Works with existing MABE datasets and models
- **Production Ready**: Comprehensive logging, error handling, and validation
- **CLI Interface**: Single command-line tool for all pipeline operations
- **Local Evaluation**: Comprehensive evaluation with ground truth data for model development
- **Flexible Evaluation**: Both competition-style and local evaluation options

## Quick Start

### Installation

1. Clone or download the pipeline:
```bash
cd C:\Users\MYPC\Documents\MABEPipeline
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Verify installation:
```bash
python bin/run_pipeline.py --help
```

### Basic Usage

Run the full pipeline:
```bash
python bin/run_pipeline.py all --config configs/default.yaml
```

Train models only:
```bash
python bin/run_pipeline.py train --config configs/default.yaml --override training.epochs=10
```

Run inference:
```bash
python bin/run_pipeline.py infer --config configs/default.yaml
```

## Pipeline Commands

### Preprocessing
```bash
python bin/run_pipeline.py preprocess --config configs/default.yaml --max-videos 5
```

### Training
```bash
python bin/run_pipeline.py train --config configs/default.yaml --resume
```

### Hyperparameter Tuning

#### Phase 1: Class Imbalance Optimization (Recommended)
```bash
# Multi-objective tuning focusing on behavior diversity
python bin/run_pipeline.py tune \
  --config configs/default.yaml \
  --phase class_imbalance \
  --n-trials 30 \
  --diversity-weight 0.2
```

#### Standard Hyperparameter Tuning
```bash
# Traditional single-objective tuning
python bin/run_pipeline.py tune --config configs/default.yaml --n-trials 50
```

### Inference
```bash
python bin/run_pipeline.py infer --config configs/default.yaml --confidence 0.4
```

### Evaluation

#### Pipeline Evaluation (for competition submissions)
```bash
python bin/run_pipeline.py evaluate --config configs/default.yaml --predictions outputs/submissions/submission_20241016_123456.csv
```

#### Local Evaluation (with ground truth data)
```bash
# Evaluate with all ground truth data
python scripts/evaluate_local.py --predictions outputs/submissions/submission_20241016_123456.csv

# Evaluate with limited videos (faster)
python scripts/evaluate_local.py --predictions outputs/submissions/submission_20241016_123456.csv --max-videos 10

# Specify custom output path
python scripts/evaluate_local.py --predictions outputs/submissions/submission_20241016_123456.csv --output results.json
```

**Note**: The local evaluation script provides comprehensive metrics using actual ground truth data from your MABE dataset, while the pipeline evaluation is designed for competition submissions without ground truth.

## Configuration

The pipeline uses YAML configuration files. Default settings are in `configs/default.yaml`:

```yaml
dataset:
  path: C:/Users/MYPC/Documents/MABEDatasets/MABe-extracted
  train_csv: train.csv
  test_csv: test.csv

training:
  model_type: both  # 'cnn'|'lstm'|'both'
  batch_size: 32
  epochs: 30
  learning_rate: 0.001

device:
  use_cuda: true
  device_str: cuda:0

seed: 42
```

### Override Configuration

Override any setting via command line:
```bash
python bin/run_pipeline.py train --override training.epochs=10 training.batch_size=16
```

Or use environment variables:
```bash
export MABE_EPOCHS=10
export MABE_BATCH_SIZE=16
python bin/run_pipeline.py train
```

## Project Structure

```
MABEPipeline/
├── bin/
│   └── run_pipeline.py          # Main CLI entry point
├── configs/
│   ├── default.yaml             # Default configuration
│   ├── paths.yaml               # Path mappings
│   └── example_overrides.yaml   # Override examples
├── src/
│   └── mabe/
│       ├── preprocessing.py     # Data preprocessing
│       ├── train.py             # Model training
│       ├── inference.py         # Model inference
│       ├── hyperparameter.py    # Hyperparameter tuning
│       ├── evaluate_local.py    # Local evaluation
│       ├── train_pipeline.py    # Training orchestrator
│       ├── infer_pipeline.py    # Inference orchestrator
│       ├── tune.py              # Tuning orchestrator
│       ├── evaluate.py          # Evaluation orchestrator
│       └── utils/
│           ├── config.py        # Configuration management
│           ├── logger.py        # Logging setup
│           ├── seed.py          # Random seed management
│           └── io_compat.py     # Legacy file compatibility
├── data/                        # Data directory
├── outputs/                     # Output directory
│   ├── models/                  # Model checkpoints
│   ├── submissions/             # Kaggle submissions
│   ├── logs/                    # Training logs
│   └── studies/                 # Optuna studies
├── tests/                       # Unit tests
├── scripts/                     # Utility scripts
│   └── evaluate_local.py        # Local evaluation with ground truth
└── requirements.txt             # Dependencies
```

## Advanced Usage

### Local Evaluation with Ground Truth

The local evaluation script provides comprehensive metrics for model development and validation:

```bash
# Basic evaluation
python scripts/evaluate_local.py --predictions your_predictions.csv

# Evaluate with video limit (faster for testing)
python scripts/evaluate_local.py --predictions your_predictions.csv --max-videos 5

# Custom output location
python scripts/evaluate_local.py --predictions your_predictions.csv --output my_results.json
```

**Features:**
- **Overall F-score**: Overall model performance
- **Per-behavior F-scores**: Performance for each behavior type (approach, attack, avoid, etc.)
- **Per-video F-scores**: Performance breakdown by video
- **Detailed metrics**: Precision, recall, and comprehensive statistics
- **Ground truth integration**: Uses actual MABE dataset annotations

**Requirements:**
- Predictions must be in the standard format: `video_id, agent_id, target_id, action, start_frame, stop_frame`
- Ground truth data must be available in the dataset path (training videos only)

### Custom Configuration

Create a custom config file:
```yaml
# my_config.yaml
dataset:
  path: /path/to/your/data

training:
  epochs: 50
  batch_size: 64
  model_type: cnn

device:
  use_cuda: true
```

Use it:
```bash
python bin/run_pipeline.py train --config my_config.yaml
```

### Hyperparameter Tuning

#### Multi-Objective Optimization (Phase 1: Class Imbalance)
Focus on improving behavior diversity and F1 score:
```bash
# Phase 1: Class imbalance parameters (focal loss, augmentation, class weights)
python bin/run_pipeline.py tune \
  --config configs/default.yaml \
  --phase class_imbalance \
  --n-trials 30 \
  --diversity-weight 0.2 \
  --timeout 3600
```

#### Analyze Tuning Results
```bash
# Analyze results and get recommendations
python scripts/analyze_tuning_results.py \
  --results outputs/optuna_results_*.json \
  --output-dir outputs/analysis
```

#### Train with Best Parameters
```bash
# Train with optimized hyperparameters
python bin/run_pipeline.py train \
  --config configs/default.yaml \
  --override training.focal_gamma=3.0 \
  --override training.class_weight_power=1.5 \
  --override training.augment_factor=3.0
```

#### Standard Single-Objective Tuning
```bash
# Traditional optimization (accuracy only)
python bin/run_pipeline.py tune --config configs/default.yaml --n-trials 100 --timeout 3600
```

### Resume Training

Resume from checkpoint:
```bash
python bin/run_pipeline.py train --config configs/default.yaml --resume --checkpoint outputs/models/checkpoint.pth
```

## Troubleshooting

### Common Issues

1. **CUDA not available**: Set `device.use_cuda: false` in config
2. **Out of memory**: Reduce `training.batch_size` in config
3. **Missing data**: Check `dataset.path` in config points to correct location
4. **Import errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`
5. **Local evaluation returns 0.0000 F-score**: This is expected for test predictions (no ground truth). Use training data for meaningful evaluation.
6. **Ground truth not found**: Ensure your dataset path contains the `train_annotation/` directory with parquet files.

### Debug Mode

Enable verbose logging:
```bash
python bin/run_pipeline.py train --config configs/default.yaml --verbose
```

### Log Files

Check log files in `outputs/logs/` for detailed information about pipeline execution.

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/ tests/
isort src/ tests/
flake8 src/ tests/
```

### Adding New Features

1. Add new modules to `src/mabe/`
2. Update `src/mabe/__init__.py` to export new functions
3. Add CLI commands to `bin/run_pipeline.py`
4. Add tests in `tests/`
5. Update documentation

## Migration from Legacy Code

If you have existing MABE code, the pipeline provides compatibility:

1. **Legacy frame labels**: Use `scripts/migrate_frame_labels.py` to convert old CSV files
2. **Existing models**: The pipeline can load models saved by the original code
3. **Data paths**: Update `dataset.path` in config to point to your data

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review log files in `outputs/logs/`
3. Create an issue with detailed error information
# MABEPipeline
