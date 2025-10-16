# MABE Pipeline API Documentation

## Overview

The MABE Pipeline provides a comprehensive API for mouse behavior analysis. This document describes the main functions and classes available in the pipeline.

## Core Modules

### preprocessing

#### `preprocess_data(cfg: Dict, max_videos: int = None) -> pd.DataFrame`

Main preprocessing function that loads and processes data.

**Parameters:**
- `cfg`: Configuration dictionary
- `max_videos`: Maximum number of videos to process

**Returns:**
- DataFrame with frame labels

**Example:**
```python
from mabe import preprocess_data
frame_labels = preprocess_data(cfg, max_videos=5)
```

#### `extract_features(frame_labels_df: pd.DataFrame, cfg: Dict) -> int`

Extract features from frame labels.

**Parameters:**
- `frame_labels_df`: DataFrame with frame labels
- `cfg`: Configuration dictionary

**Returns:**
- Feature dimension (typically 26)

### train

#### `build_model(cfg: Dict, input_dim: int, model_type: str) -> nn.Module`

Build a model based on configuration.

**Parameters:**
- `cfg`: Configuration dictionary
- `input_dim`: Input feature dimension
- `model_type`: Model type ('cnn' or 'lstm')

**Returns:**
- PyTorch model

#### `train(model, train_loader, val_loader, cfg: Dict, resume_checkpoint: Optional[str] = None) -> Dict`

Train a model with given configuration.

**Parameters:**
- `model`: PyTorch model to train
- `train_loader`: Training data loader
- `val_loader`: Validation data loader
- `cfg`: Configuration dictionary
- `resume_checkpoint`: Optional checkpoint path to resume from

**Returns:**
- Dictionary with training results

### inference

#### `load_model(model_path: str, model_class: str = 'cnn', input_dim: int = 26) -> nn.Module`

Load trained model from file.

**Parameters:**
- `model_path`: Path to model file
- `model_class`: Model class ('cnn' or 'lstm')
- `input_dim`: Input feature dimension

**Returns:**
- Loaded PyTorch model

#### `predict_single(model, data_loader, device, return_proba: bool = True) -> np.ndarray`

Make predictions with a single model.

**Parameters:**
- `model`: PyTorch model
- `data_loader`: Data loader
- `device`: Device to run on
- `return_proba`: Whether to return probabilities or class predictions

**Returns:**
- Array of predictions

#### `ensemble_predictions(proba_list: List[np.ndarray], method: str = 'average') -> np.ndarray`

Ensemble multiple model predictions.

**Parameters:**
- `proba_list`: List of probability arrays
- `method`: Ensemble method ('average' or 'max')

**Returns:**
- Ensembled predictions

### hyperparameter

#### `objective_fn(trial, cfg: Dict, train_df: pd.DataFrame, val_df: pd.DataFrame) -> float`

Optuna objective function for hyperparameter optimization.

**Parameters:**
- `trial`: Optuna trial object
- `cfg`: Configuration dictionary
- `train_df`: Training data
- `val_df`: Validation data

**Returns:**
- Validation accuracy score

### evaluate_local

#### `evaluate_predictions(ground_truth: pd.DataFrame, predictions: pd.DataFrame) -> Dict`

Evaluate predictions against ground truth.

**Parameters:**
- `ground_truth`: Ground truth DataFrame
- `predictions`: Predictions DataFrame

**Returns:**
- Dictionary with evaluation metrics

#### `calculate_video_f_score(predictions: pd.DataFrame, ground_truth: pd.DataFrame) -> float`

Calculate F-score for a single video.

**Parameters:**
- `predictions`: Predictions DataFrame
- `ground_truth`: Ground truth DataFrame

**Returns:**
- F-score value

## Pipeline Orchestrators

### train_pipeline

#### `run_training(cfg: Dict, resume: bool = False, checkpoint_path: Optional[str] = None) -> Dict`

Run the complete training pipeline.

**Parameters:**
- `cfg`: Configuration dictionary
- `resume`: Whether to resume training from checkpoint
- `checkpoint_path`: Path to checkpoint file

**Returns:**
- Dictionary with training results

### infer_pipeline

#### `run_inference(cfg: Dict, test_csv: Optional[str] = None, output_path: Optional[str] = None) -> pd.DataFrame`

Run the complete inference pipeline.

**Parameters:**
- `cfg`: Configuration dictionary
- `test_csv`: Optional path to test CSV file
- `output_path`: Optional path for output submission

**Returns:**
- DataFrame with submission predictions

### tune

#### `run_optuna(cfg: Dict, n_trials: Optional[int] = None, timeout: Optional[int] = None) -> Dict`

Run Optuna hyperparameter optimization.

**Parameters:**
- `cfg`: Configuration dictionary
- `n_trials`: Number of trials (from config if None)
- `timeout`: Timeout in seconds (from config if None)

**Returns:**
- Dictionary with optimization results

### evaluate

#### `run_local_evaluation(cfg: Dict, predictions_csv: str) -> Dict`

Run local evaluation on predictions.

**Parameters:**
- `cfg`: Configuration dictionary
- `predictions_csv`: Path to predictions CSV file

**Returns:**
- Dictionary with evaluation results

## Utility Functions

### config

#### `load_config(config_path: Optional[Path] = None, overrides: Optional[List[str]] = None) -> Dict`

Load configuration from files and overrides.

**Parameters:**
- `config_path`: Path to configuration file
- `overrides`: List of override strings in format 'key=value'

**Returns:**
- Configuration dictionary

### logger

#### `get_logger(name: str, log_level: str = 'INFO') -> logging.Logger`

Get a logger instance.

**Parameters:**
- `name`: Logger name
- `log_level`: Logging level

**Returns:**
- Logger instance

### seed

#### `set_seed(seed: int) -> None`

Set random seed for reproducibility.

**Parameters:**
- `seed`: Random seed value

### io_compat

#### `read_legacy_frame_labels(path: str) -> pd.DataFrame`

Read legacy frame labels file.

**Parameters:**
- `path`: Path to legacy file

**Returns:**
- DataFrame with frame labels

#### `convert_to_standard_format(df: pd.DataFrame) -> pd.DataFrame`

Convert legacy DataFrame to standard format.

**Parameters:**
- `df`: Legacy DataFrame

**Returns:**
- Standardized DataFrame

## Data Classes

### MouseBehaviorDataset

Custom PyTorch dataset for mouse behavior detection.

**Methods:**
- `__len__()`: Return dataset length
- `__getitem__(idx)`: Get item at index
- `extract_tracking_features(video_id, frame, agent_id, target_id, idx)`: Extract features
- `extract_spatial_features(frame_data, agent_id, target_id)`: Extract spatial features
- `extract_temporal_features(tracking_df, frame)`: Extract temporal features
- `extract_interaction_features(frame_data, agent_id, target_id)`: Extract interaction features

### MABEDataPreprocessor

Data preprocessing class for MABE behavior detection.

**Methods:**
- `load_data(frame_labels_path: str = None, max_videos: int = None)`: Load data
- `extract_features(frame_labels_df)`: Extract features
- `create_dataset(frame_labels_df, batch_size=None, train_split=None, use_augmentation=None)`: Create datasets
- `save_preprocessing_info(feature_dim, frame_labels_df)`: Save preprocessing info

## Model Classes

### BehaviorCNN

CNN model for behavior classification.

**Parameters:**
- `input_dim`: Input feature dimension
- `num_classes`: Number of output classes
- `dropout`: Dropout rate

### BehaviorLSTM

LSTM model for temporal behavior classification.

**Parameters:**
- `input_dim`: Input feature dimension
- `hidden_dim`: Hidden dimension
- `num_classes`: Number of output classes
- `num_layers`: Number of LSTM layers
- `dropout`: Dropout rate

### EarlyStopping

Early stopping to prevent overfitting.

**Parameters:**
- `patience`: Number of epochs to wait
- `min_delta`: Minimum change to qualify as improvement
- `restore_best_weights`: Whether to restore best weights

## Configuration

The pipeline uses YAML configuration files with the following structure:

```yaml
dataset:
  path: /path/to/dataset
  train_csv: train.csv
  test_csv: test.csv

paths:
  models_dir: outputs/models
  outputs_dir: outputs
  logs_dir: outputs/logs
  submissions_dir: outputs/submissions

training:
  model_type: both
  batch_size: 32
  epochs: 30
  learning_rate: 0.001
  weight_decay: 0.0
  val_fraction: 0.2
  max_videos: 5
  use_augmentation: true

inference:
  confidence_threshold: 0.4
  ensemble_method: max
  min_duration: 1

optuna:
  n_trials: 50
  timeout: 3600
  n_splits: 3

device:
  use_cuda: true
  device_str: cuda:0

seed: 42
```

## Error Handling

The pipeline includes comprehensive error handling:

- **FileNotFoundError**: When required files are missing
- **ValueError**: When configuration values are invalid
- **RuntimeError**: When models fail to load or train
- **ImportError**: When required dependencies are missing

## Logging

The pipeline uses structured logging with different levels:

- **DEBUG**: Detailed debugging information
- **INFO**: General information about pipeline progress
- **WARNING**: Non-critical issues that don't stop execution
- **ERROR**: Critical errors that stop execution

Logs are written to both console and file (in `outputs/logs/`).
