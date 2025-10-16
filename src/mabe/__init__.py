"""
MABE Pipeline - Machine Learning Pipeline for Mouse Behavior Analysis

This package provides a production-ready machine learning pipeline for the MABE
(Mouse Behavior Analysis) competition. It includes data preprocessing, model training,
hyperparameter tuning, inference, and evaluation components.

Main Components:
- preprocessing: Data loading and feature extraction
- train: Model training and validation
- inference: Model inference and prediction generation
- hyperparameter: Hyperparameter optimization with Optuna
- evaluate_local: Local evaluation and metrics calculation

Pipeline Orchestrators:
- train_pipeline: High-level training orchestration
- infer_pipeline: High-level inference orchestration
- tune: Hyperparameter tuning orchestration
- evaluate: Evaluation orchestration

Utilities:
- config: Configuration management
- logger: Centralized logging
- seed: Random seed management for reproducibility
- io_compat: Legacy file compatibility
"""

__version__ = "0.1.0"
__author__ = "MABE Pipeline Team"

# Import main pipeline functions
from .train_pipeline import run_training
from .infer_pipeline import run_inference
from .tune import run_optuna
from .evaluate import run_local_evaluation

# Import core modules
from . import preprocessing
from . import train
from . import inference
from . import hyperparameter
from . import evaluate_local

# Import utilities
from .utils import (
    load_config,
    merge_configs,
    get_logger,
    set_seed,
    read_legacy_frame_labels,
    convert_to_standard_format
)

__all__ = [
    # Pipeline orchestrators
    'run_training',
    'run_inference', 
    'run_optuna',
    'run_local_evaluation',
    
    # Core modules
    'preprocessing',
    'train',
    'inference',
    'hyperparameter',
    'evaluate_local',
    
    # Utilities
    'load_config',
    'merge_configs',
    'get_logger',
    'set_seed',
    'read_legacy_frame_labels',
    'convert_to_standard_format'
]
