# MABE Pipeline

A production-ready machine learning pipeline for the MABE (Mouse Behavior Analysis) competition. This pipeline provides a complete end-to-end solution for data preprocessing, model training, hyperparameter optimization, inference, and evaluation.

## Features

- **Centralized Configuration**: YAML-based configuration with CLI overrides
- **Modular Design**: Clean separation of concerns with reusable components
- **Reproducible**: Deterministic behavior with seed management
- **Backward Compatible**: Works with existing MABE datasets and models
- **Production Ready**: Comprehensive logging, error handling, and validation
- **CLI Interface**: Single command-line tool for all pipeline operations

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
```bash
python bin/run_pipeline.py tune --config configs/default.yaml --n-trials 50
```

### Inference
```bash
python bin/run_pipeline.py infer --config configs/default.yaml --confidence 0.4
```

### Evaluation
```bash
python bin/run_pipeline.py evaluate --config configs/default.yaml --predictions outputs/submissions/submission_20241016_123456.csv
```

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
└── requirements.txt             # Dependencies
```

## Advanced Usage

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

Run Optuna optimization:
```bash
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
