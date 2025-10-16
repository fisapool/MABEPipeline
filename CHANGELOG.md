# Changelog

All notable changes to the MABE Pipeline project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-10-16

### Added
- Initial release of MABE Pipeline refactoring
- Centralized configuration system with YAML support
- Single CLI entry point (`bin/run_pipeline.py`) with subcommands:
  - `preprocess`: Data preprocessing and feature extraction
  - `train`: Model training with CNN and LSTM architectures
  - `tune`: Hyperparameter optimization with Optuna
  - `infer`: Model inference and prediction generation
  - `evaluate`: Local evaluation and metrics calculation
  - `all`: Complete end-to-end pipeline
- Modular architecture with clean separation of concerns:
  - `src/mabe/preprocessing.py`: Data loading and preprocessing
  - `src/mabe/train.py`: Model training components
  - `src/mabe/inference.py`: Model inference components
  - `src/mabe/hyperparameter.py`: Hyperparameter optimization
  - `src/mabe/evaluate_local.py`: Evaluation metrics
  - `src/mabe/train_pipeline.py`: Training orchestrator
  - `src/mabe/infer_pipeline.py`: Inference orchestrator
  - `src/mabe/tune.py`: Tuning orchestrator
  - `src/mabe/evaluate.py`: Evaluation orchestrator
- Utility modules:
  - `src/mabe/utils/config.py`: Configuration management with CLI/env overrides
  - `src/mabe/utils/logger.py`: Centralized logging with file rotation
  - `src/mabe/utils/seed.py`: Random seed management for reproducibility
  - `src/mabe/utils/io_compat.py`: Legacy file compatibility
- Configuration system:
  - `configs/default.yaml`: Default configuration with existing paths
  - `configs/paths.yaml`: Path mappings for consistent file handling
  - `configs/example_overrides.yaml`: Example configuration overrides
- Backward compatibility:
  - Support for existing `mabe_frame_labels_*.csv` files
  - Compatibility with models saved by original code
  - Legacy file migration utilities
- Reproducibility features:
  - Deterministic random seed management
  - RNG state logging and restoration
  - Consistent data loading and preprocessing
- Production-ready features:
  - Comprehensive error handling and logging
  - Configuration validation
  - Progress tracking and metrics
  - File output management
- Testing framework:
  - Unit tests for core components
  - Integration tests for pipeline stages
  - Configuration validation tests
- Documentation:
  - Comprehensive README with usage examples
  - API documentation for all modules
  - Troubleshooting guide
  - Migration guide from legacy code

### Changed
- Refactored existing MABE codebase into modular pipeline
- Replaced hardcoded paths with configuration-driven approach
- Improved error handling and logging throughout
- Enhanced model training with better checkpointing
- Streamlined inference pipeline with ensemble support
- Optimized hyperparameter tuning with stratified cross-validation

### Technical Details
- **Configuration Priority**: CLI args > env vars > user config > defaults
- **Supported Overrides**: Dotted key notation (e.g., `training.epochs=10`)
- **Model Compatibility**: Works with existing CNN and LSTM architectures
- **Data Formats**: Supports legacy CSV and new standardized formats
- **Device Support**: Automatic CUDA detection with CPU fallback
- **Logging**: Timed rotating file logs with console output
- **Testing**: Pytest-based testing with coverage reporting

### Migration Notes
- Original `MABEModel/` directory remains untouched
- New pipeline can read existing frame labels and models
- Gradual migration path with compatibility layer
- Configuration preserves existing hardcoded paths as defaults

### Breaking Changes
- None (this is the initial release)

### Dependencies
- Python 3.8+
- PyTorch 2.0+
- NumPy, Pandas, Scikit-learn
- Optuna for hyperparameter optimization
- PyYAML for configuration management
- PyArrow/FastParquet for data I/O

### Known Issues
- None at initial release

### Future Roadmap
- [ ] Docker containerization
- [ ] Weights & Biases integration
- [ ] Advanced ensemble methods
- [ ] Real-time inference API
- [ ] Automated hyperparameter optimization
- [ ] Model versioning and registry
- [ ] Distributed training support
