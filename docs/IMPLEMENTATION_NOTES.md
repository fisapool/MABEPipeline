# MABE Pipeline Implementation Notes

## Overview

This document provides detailed implementation notes for the MABE Pipeline refactoring, including design decisions, technical details, and migration strategies.

## Architecture Decisions

### 1. Configuration System

**Decision**: YAML-based configuration with CLI and environment variable overrides.

**Rationale**:
- Provides flexibility for different environments
- Maintains backward compatibility with existing hardcoded paths
- Enables easy parameter tuning without code changes
- Supports both development and production configurations

**Implementation**:
```python
# Priority: CLI > Env Vars > User Config > Defaults
def load_config(config_path=None, overrides=None):
    # Load default config
    config = load_yaml('configs/default.yaml')
    
    # Apply user config
    if config_path:
        user_config = load_yaml(config_path)
        merge_configs(config, user_config)
    
    # Apply environment variables
    apply_env_overrides(config)
    
    # Apply CLI overrides
    if overrides:
        apply_cli_overrides(config, overrides)
    
    return config
```

### 2. Modular Design

**Decision**: Separate core modules from pipeline orchestrators.

**Rationale**:
- Core modules (`preprocessing.py`, `train.py`, etc.) handle specific functionality
- Pipeline orchestrators (`train_pipeline.py`, `infer_pipeline.py`, etc.) coordinate workflows
- Enables independent testing and development
- Supports both programmatic and CLI usage

**Structure**:
```
src/mabe/
├── preprocessing.py      # Core data processing
├── train.py             # Core training logic
├── inference.py         # Core inference logic
├── hyperparameter.py    # Core optimization logic
├── evaluate_local.py    # Core evaluation logic
├── train_pipeline.py    # Training orchestrator
├── infer_pipeline.py    # Inference orchestrator
├── tune.py              # Tuning orchestrator
├── evaluate.py          # Evaluation orchestrator
└── utils/               # Utility functions
```

### 3. Backward Compatibility

**Decision**: Maintain compatibility with existing data formats and models.

**Rationale**:
- Preserves existing workflows and investments
- Enables gradual migration
- Reduces risk of data loss
- Supports A/B testing between old and new pipelines

**Implementation**:
```python
# Legacy file detection and conversion
def load_legacy_frame_labels(search_dir):
    legacy_files = list(search_dir.glob("mabe_frame_labels_*.csv"))
    if legacy_files:
        latest_file = max(legacy_files, key=lambda x: x.stat().st_mtime)
        return read_legacy_frame_labels(latest_file)
    return None

# Model compatibility
def load_model(model_path, model_class='cnn', input_dim=26):
    if model_class.lower() == 'cnn':
        model = BehaviorCNN_Old(input_dim=input_dim, num_classes=8)
    elif model_class.lower() == 'lstm':
        model = BehaviorLSTM_Old(input_dim=input_dim, hidden_dim=128, num_classes=8)
    
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    return model
```

## Technical Implementation Details

### 1. Data Processing Pipeline

**Feature Extraction**:
- Maintains 26-feature architecture for compatibility
- Spatial features: 12 features (positions, distances, angles)
- Temporal features: 10 features (velocity, acceleration)
- Interaction features: 4 features (mouse-to-mouse interactions)

**Data Augmentation**:
```python
def apply_augmentation(self, features):
    # Add Gaussian noise (5% of feature std)
    noise_std = np.std(features) * 0.05
    noise = np.random.normal(0, noise_std, features.shape)
    features = features + noise
    
    # Random scaling (0.95 to 1.05)
    scale_factor = np.random.uniform(0.95, 1.05)
    features = features * scale_factor
    
    # Small random shifts for position features
    if len(features) >= 12:
        position_shift = np.random.normal(0, 0.1, 12)
        features[:12] = features[:12] + position_shift
    
    return features
```

### 2. Model Architecture

**CNN Model**:
```python
class BehaviorCNN_Old(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.3):
        super(BehaviorCNN_Old, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
```

**LSTM Model**:
```python
class BehaviorLSTM_Old(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=2, dropout=0.3):
        super(BehaviorLSTM_Old, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc1 = nn.Linear(hidden_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
```

### 3. Training Pipeline

**Class Weight Calculation**:
```python
def compute_class_weights(labels):
    unique_classes = np.unique(labels)
    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=labels
    )
    
    # Create full weight array for all 8 classes
    full_weights = np.ones(8)
    for i, weight in enumerate(class_weights_array):
        class_id = unique_classes[i]
        full_weights[class_id] = weight
    
    return torch.FloatTensor(full_weights)
```

**Early Stopping**:
```python
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
```

### 4. Inference Pipeline

**Ensemble Methods**:
```python
def ensemble_predictions(proba_list, method='average'):
    if method == 'average':
        return np.mean(proba_list, axis=0)
    elif method == 'max':
        return np.max(proba_list, axis=0)
    else:
        raise ValueError(f"Unknown ensemble method: {method}")
```

**Submission Format**:
```python
def create_kaggle_submission(predictions, cfg, min_duration=1):
    submission_data = []
    
    # Group predictions by video, agent, target, and behavior
    grouped_predictions = {}
    for pred in predictions:
        key = (pred['video_id'], pred['agent_id'], pred['target_id'], pred['behavior_name'])
        if key not in grouped_predictions:
            grouped_predictions[key] = []
        grouped_predictions[key].append(pred['frame'])
    
    # Create continuous ranges
    for (video_id, agent_id, target_id, behavior_name), frames in grouped_predictions.items():
        frames = sorted(frames)
        ranges = build_continuous_ranges(frames, min_duration)
        
        for start_frame, stop_frame in ranges:
            submission_data.append({
                'video_id': video_id,
                'agent_id': agent_id,
                'target_id': target_id,
                'action': behavior_name,
                'start_frame': start_frame,
                'stop_frame': stop_frame + 1  # Convert to exclusive
            })
    
    return pd.DataFrame(submission_data)
```

### 5. Hyperparameter Optimization

**Optuna Integration**:
```python
def objective_fn(trial, cfg, train_df, val_df):
    # Sample hyperparameters
    params = {
        'dropout': trial.suggest_categorical('dropout', [0.1, 0.2, 0.3, 0.4, 0.5]),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.0001, 0.001, 0.01]),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
        'weight_decay': trial.suggest_categorical('weight_decay', [0.0, 1e-4, 1e-3])
    }
    
    # Create and train model
    model = build_model(cfg, input_dim=26, model_type='cnn')
    score = train_and_evaluate(model, params, train_df, val_df)
    
    return score
```

### 6. Evaluation Metrics

**F-Score Calculation**:
```python
def calculate_overall_f_score(ground_truth, predictions):
    precision = calculate_precision(ground_truth, predictions)
    recall = calculate_recall(ground_truth, predictions)
    
    if precision + recall == 0:
        f_score = 0.0
    else:
        f_score = 2 * (precision * recall) / (precision + recall)
    
    return f_score
```

## Migration Strategy

### 1. Data Migration

**Legacy Frame Labels**:
- Detect existing `mabe_frame_labels_*.csv` files
- Convert to standardized format
- Maintain column compatibility
- Generate migration report

**Implementation**:
```python
def migrate_frame_labels(source_dir, output_dir):
    legacy_files = find_legacy_files(source_dir)
    converted_files = []
    
    for file_path in legacy_files:
        if validate_legacy_file(file_path):
            converted_path = convert_legacy_file(file_path, output_dir)
            if converted_path:
                converted_files.append(converted_path)
    
    return create_migration_report(legacy_files, converted_files, output_dir)
```

### 2. Model Migration

**Existing Models**:
- Load models saved by original pipeline
- Maintain architecture compatibility
- Support both CNN and LSTM models
- Handle different model versions

**Implementation**:
```python
def load_model(model_path, model_class='cnn', input_dim=26):
    try:
        if model_class.lower() == 'cnn':
            model = BehaviorCNN_Old(input_dim=input_dim, num_classes=8)
        elif model_class.lower() == 'lstm':
            model = BehaviorLSTM_Old(input_dim=input_dim, hidden_dim=128, num_classes=8)
        
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        return model
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        raise
```

### 3. Configuration Migration

**Path Mapping**:
- Map existing hardcoded paths to configuration
- Provide sensible defaults
- Support environment-specific overrides
- Maintain backward compatibility

**Implementation**:
```yaml
# configs/default.yaml
dataset:
  path: C:/Users/MYPC/Documents/MABEDatasets/MABe-extracted
  train_csv: train.csv
  test_csv: test.csv

paths:
  models_dir: outputs/models
  outputs_dir: outputs
  logs_dir: outputs/logs
  submissions_dir: outputs/submissions
```

## Testing Strategy

### 1. Unit Tests

**Feature Extraction Tests**:
- Test feature extraction returns correct shape (26 features)
- Test non-zero features on sample data
- Test spatial/temporal/interaction feature functions
- Test handling of missing data

**Preprocessing Tests**:
- Test data loading with valid/invalid paths
- Test tracking data loading
- Test dataset creation
- Test class weight calculation

**Configuration Tests**:
- Test config loading and merging
- Test override precedence
- Test dotted key parsing
- Test environment variable handling

### 2. Integration Tests

**End-to-End Pipeline**:
```bash
# Full pipeline test
python bin/run_pipeline.py preprocess --config configs/default.yaml --override training.max_videos=2
python bin/run_pipeline.py train --config configs/default.yaml --override training.epochs=1
python bin/run_pipeline.py infer --config configs/default.yaml
```

**CI Test Suite**:
```bash
# Quick test for CI
./scripts/ci_run_short.sh
```

### 3. Validation Tests

**Output Comparison**:
- Compare model outputs between old and new pipelines
- Validate submission file formats
- Check metric calculations
- Verify data processing consistency

## Performance Considerations

### 1. Data Loading

**Optimizations**:
- Use multiple workers for data loading
- Enable data caching for repeated access
- Use parquet format for faster I/O
- Implement data sampling for development

**Implementation**:
```python
train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                        sampler=weighted_sampler, num_workers=4)
```

### 2. Training

**Optimizations**:
- Use mixed precision training
- Enable gradient checkpointing
- Implement early stopping
- Use learning rate scheduling

**Implementation**:
```python
# Mixed precision training
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()
with autocast():
    output = model(data)
    loss = criterion(output, target)
```

### 3. Inference

**Optimizations**:
- Use batch inference
- Enable model quantization
- Implement ensemble methods
- Use GPU acceleration

**Implementation**:
```python
# Batch inference
def predict_batch(model, data_loader, device):
    model.eval()
    all_predictions = []
    
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            outputs = model(data)
            all_predictions.append(outputs.cpu().numpy())
    
    return np.concatenate(all_predictions, axis=0)
```

## Error Handling

### 1. Data Errors

**File Not Found**:
```python
def load_data(cfg):
    dataset_path = Path(cfg['dataset']['path'])
    if not dataset_path.exists():
        logger.error(f"Dataset path does not exist: {dataset_path}")
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
```

**Invalid Data Format**:
```python
def validate_dataframe(df):
    required_columns = ['video_id', 'frame', 'agent_id', 'target_id', 'behavior']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
```

### 2. Model Errors

**Model Loading**:
```python
def load_model(model_path, model_class='cnn', input_dim=26):
    try:
        model = create_model(model_class, input_dim)
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        return model
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        raise
```

**Training Errors**:
```python
def train_model(model, train_loader, val_loader, cfg):
    try:
        # Training logic
        pass
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            logger.error("CUDA out of memory. Try reducing batch size.")
            raise
        else:
            logger.error(f"Training error: {e}")
            raise
```

### 3. Configuration Errors

**Invalid Configuration**:
```python
def validate_config(cfg):
    required_keys = ['dataset', 'paths', 'training', 'device']
    missing_keys = [key for key in required_keys if key not in cfg]
    
    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {missing_keys}")
```

## Logging and Monitoring

### 1. Structured Logging

**Log Levels**:
- DEBUG: Detailed debugging information
- INFO: General information about pipeline progress
- WARNING: Non-critical issues that don't stop execution
- ERROR: Critical errors that stop execution

**Implementation**:
```python
def get_logger(name, log_level='INFO'):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(log_level.upper())
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler
        file_handler = TimedRotatingFileHandler('outputs/logs/pipeline.log')
        file_handler.setFormatter(console_formatter)
        logger.addHandler(file_handler)
    
    return logger
```

### 2. Metrics Collection

**Training Metrics**:
```python
def log_training_metrics(epoch, train_loss, val_loss, val_acc, lr):
    logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, "
               f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, LR: {lr:.6f}")
```

**Inference Metrics**:
```python
def log_inference_metrics(predictions_count, confidence_threshold, ensemble_method):
    logger.info(f"Generated {predictions_count} predictions with "
               f"confidence threshold {confidence_threshold} using {ensemble_method} ensemble")
```

## Future Enhancements

### 1. Advanced Features

**Model Architectures**:
- Transformer-based models
- Attention mechanisms
- Multi-scale feature extraction
- Temporal convolution networks

**Data Augmentation**:
- Advanced augmentation techniques
- Adversarial training
- Domain adaptation
- Synthetic data generation

### 2. Scalability

**Distributed Training**:
- Multi-GPU training
- Distributed data parallel
- Model parallel training
- Gradient synchronization

**Cloud Integration**:
- AWS SageMaker integration
- Google Cloud AI Platform
- Azure Machine Learning
- Kubernetes deployment

### 3. Monitoring

**Real-time Monitoring**:
- Prometheus metrics
- Grafana dashboards
- Alert systems
- Performance tracking

**Model Management**:
- Model versioning
- A/B testing
- Model registry
- Automated retraining

## Conclusion

The MABE Pipeline refactoring provides a robust, scalable, and maintainable solution for mouse behavior analysis. The modular design, comprehensive testing, and backward compatibility ensure a smooth transition from the original codebase while providing a foundation for future enhancements.

Key benefits:
- **Maintainability**: Clean, modular code structure
- **Scalability**: Support for distributed training and inference
- **Flexibility**: Configuration-driven approach
- **Compatibility**: Backward compatibility with existing data and models
- **Testing**: Comprehensive test coverage
- **Documentation**: Detailed API and deployment guides
