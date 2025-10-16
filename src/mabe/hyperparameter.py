"""
MABE Hyperparameter Optimization Module

Refactored from original hyperparameter.py to work with configuration system.
Handles Optuna-based hyperparameter optimization for CNN and LSTM models.
"""

import optuna
import itertools
from typing import Dict, List, Tuple, Any
import json
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import logging
import pandas as pd
import numpy as np

# Add stratified k-fold validation
from sklearn.model_selection import StratifiedKFold

# Import utilities
from .utils.logger import get_logger
from .utils.seed import set_seed
from .train import build_model, compute_class_weights

logger = get_logger(__name__)


def calculate_behavior_diversity(predictions: np.ndarray, num_classes: int = 8) -> float:
    """
    Calculate diversity score based on unique behaviors predicted
    
    Args:
        predictions: Array of predicted class indices
        num_classes: Total number of possible classes
        
    Returns:
        Diversity score from 0.0 to 1.0
    """
    if len(predictions) == 0:
        return 0.0
    
    unique_behaviors = len(np.unique(predictions))
    return unique_behaviors / num_classes  # 0.0 to 1.0 score


def compute_class_weights_with_power(labels, power: float = 1.0) -> torch.Tensor:
    """
    Compute class weights with configurable power for stronger/weaker weighting
    
    Args:
        labels: Sequence of class labels
        power: Power for weight transformation (1.0 = normal, >1.0 = stronger weighting)
        
    Returns:
        Tensor of class weights
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    # Calculate balanced weights
    weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    # Apply power transformation: weights^power
    weights = np.power(weights, power)
    # Normalize to sum to num_classes
    weights = weights * (len(weights) / weights.sum())
    return torch.FloatTensor(weights)


def objective_fn(trial, cfg: Dict, train_df: pd.DataFrame, val_df: pd.DataFrame) -> float:
    """
    Optuna objective function for hyperparameter optimization with multi-objective scoring
    
    Args:
        trial: Optuna trial object
        cfg: Configuration dictionary
        train_df: Training data
        val_df: Validation data
        
    Returns:
        Combined score (F1 + weighted behavior diversity)
    """
    # Get model type from config
    model_type = cfg.get('training', {}).get('model_type', 'cnn')
    if model_type == 'both':
        model_type = 'cnn'  # Default to CNN for optimization
    
    # Get optimization phase and diversity weight from config
    optuna_cfg = cfg.get('optuna', {})
    phase = optuna_cfg.get('phase', 'class_imbalance')
    diversity_weight = optuna_cfg.get('diversity_weight', 0.2)
    
    # Sample Phase 1 hyperparameters (class imbalance focus)
    if phase == 'class_imbalance':
        params = {
            'focal_gamma': trial.suggest_categorical('focal_gamma', [2.0, 3.0, 4.0, 5.0]),
            'focal_alpha': trial.suggest_categorical('focal_alpha', [0.25, 0.5, 0.75, 1.0]),
            'augment_factor': trial.suggest_categorical('augment_factor', [1.5, 2.0, 3.0, 5.0]),
            'class_weight_power': trial.suggest_categorical('class_weight_power', [0.5, 1.0, 1.5, 2.0]),
            'learning_rate': trial.suggest_categorical('learning_rate', [0.0005, 0.001, 0.002]),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
            'dropout': trial.suggest_categorical('dropout', [0.2, 0.3, 0.4, 0.5])
        }
    else:
        # Fallback to original parameter space
        if model_type == 'cnn':
            params = {
                'dropout': trial.suggest_categorical('dropout', [0.1, 0.2, 0.3, 0.4, 0.5]),
                'learning_rate': trial.suggest_categorical('learning_rate', [0.0001, 0.001, 0.01]),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
                'weight_decay': trial.suggest_categorical('weight_decay', [0.0, 1e-4, 1e-3])
            }
        else:  # LSTM
            params = {
                'num_layers': trial.suggest_categorical('num_layers', [1, 2, 3, 4]),
                'dropout': trial.suggest_categorical('dropout', [0.1, 0.2, 0.3, 0.4, 0.5]),
                'learning_rate': trial.suggest_categorical('learning_rate', [0.0001, 0.001, 0.01]),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
                'weight_decay': trial.suggest_categorical('weight_decay', [0.0, 1e-4, 1e-3])
            }
    
    # Create model with sampled parameters
    input_dim = 26  # Fixed for MABE features
    model = build_model(cfg, input_dim, model_type)
    
    # Train and evaluate with predictions tracking
    val_acc, predictions = train_and_evaluate_with_predictions(model, params, model_type, train_df, val_df, cfg, trial)
    
    # Calculate diversity score
    diversity_score = calculate_behavior_diversity(predictions)
    
    # Multi-objective: F1 (validation accuracy proxy) + weighted diversity
    # Scale diversity to have meaningful impact (20% weight)
    diversity_impact = diversity_score * 20.0  # Scale to ~20% of typical accuracy range
    combined_score = val_acc + diversity_impact
    
    # Log metrics for analysis
    trial.set_user_attr('diversity_score', diversity_score)
    trial.set_user_attr('val_acc', val_acc)
    trial.set_user_attr('combined_score', combined_score)
    
    logger.info(f"Trial {trial.number}: val_acc={val_acc:.4f}, diversity={diversity_score:.4f}, combined={combined_score:.4f}")
    
    return combined_score


def train_and_evaluate_with_predictions(model, params: Dict, model_type: str, train_df: pd.DataFrame, 
                                       val_df: pd.DataFrame, cfg: Dict, trial=None) -> Tuple[float, np.ndarray]:
    """
    Train model and return validation accuracy with predictions for diversity calculation
    
    Args:
        model: PyTorch model
        params: Hyperparameters
        model_type: Model type
        train_df: Training data
        val_df: Validation data
        cfg: Configuration
        trial: Optuna trial (optional)
        
    Returns:
        Tuple of (validation accuracy, predictions array)
    """
    device = torch.device(cfg.get('device', {}).get('device_str', 'cuda:0'))
    model = model.to(device)
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], 
                          weight_decay=params.get('weight_decay', 0.0))
    criterion = nn.CrossEntropyLoss()
    
    # Create data loaders (simplified for hyperparameter tuning)
    from .preprocessing import MouseBehaviorDataset
    from torch.utils.data import DataLoader
    
    # Get tracking data from config
    dataset_path = Path(cfg['dataset']['path'])
    tracking_data = {}
    
    # Load minimal tracking data for hyperparameter tuning
    try:
        # This is a simplified version - in practice, you'd load actual tracking data
        # For now, we'll use dummy data for hyperparameter optimization
        logger.info("Using dummy tracking data for hyperparameter optimization")
    except Exception as e:
        logger.warning(f"Could not load tracking data: {e}")
    
    # Create datasets with configurable augmentation
    augment_factor = params.get('augment_factor', 2.0)
    train_dataset = MouseBehaviorDataset(train_df, tracking_data, augment=True, augment_factor=augment_factor)
    val_dataset = MouseBehaviorDataset(val_df, tracking_data, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], 
                             shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], 
                           shuffle=False, num_workers=0)
    
    # Training loop
    best_val_acc = 0.0
    best_predictions = None
    epochs = min(20, cfg.get('training', {}).get('epochs', 30))  # Shorter for hyperparameter tuning
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        all_predictions = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
                
                # Store predictions for diversity calculation
                all_predictions.extend(predicted.cpu().numpy())
        
        val_acc = 100.0 * val_correct / val_total if val_total > 0 else 0.0
        
        # Update best results
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_predictions = np.array(all_predictions)
        
        # Report to Optuna for pruning
        if trial is not None:
            trial.report(val_acc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
    
    return best_val_acc, best_predictions if best_predictions is not None else np.array([])


def train_and_evaluate(model, params: Dict, model_type: str, train_df: pd.DataFrame, 
                     val_df: pd.DataFrame, cfg: Dict, trial=None) -> float:
    """
    Train model and return validation accuracy (legacy function for backward compatibility)
    
    Args:
        model: PyTorch model
        params: Hyperparameters
        model_type: Model type
        train_df: Training data
        val_df: Validation data
        cfg: Configuration
        trial: Optuna trial (optional)
        
    Returns:
        Validation accuracy
    """
    val_acc, _ = train_and_evaluate_with_predictions(model, params, model_type, train_df, val_df, cfg, trial)
    return val_acc


def create_model_from_params(params: Dict, model_type: str, input_dim: int = 26) -> nn.Module:
    """
    Create model from hyperparameters
    
    Args:
        params: Hyperparameter dictionary
        model_type: Model type ('cnn' or 'lstm')
        input_dim: Input feature dimension
        
    Returns:
        PyTorch model
    """
    if model_type == 'cnn':
        return build_model({'training': params}, input_dim, 'cnn')
    elif model_type == 'lstm':
        return build_model({'training': params}, input_dim, 'lstm')
    else:
        raise ValueError(f"Unknown model type: {model_type}")


class MABEHyperparameterTuner:
    """Comprehensive hyperparameter tuning for MABE models"""
    
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.dataset_path = Path(cfg['dataset']['path'])
        self.model_save_path = Path(cfg['paths']['models_dir'])
        self.device = self.setup_device()
        
        # Hyperparameter search spaces
        self.search_spaces = {
            'cnn': {
                'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
                'learning_rate': [0.0001, 0.001, 0.01],
                'batch_size': [16, 32, 64],
                'weight_decay': [0.0, 1e-4, 1e-3]
            },
            'lstm': {
                'num_layers': [1, 2, 3, 4],
                'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
                'learning_rate': [0.0001, 0.001, 0.01],
                'batch_size': [16, 32, 64],
                'weight_decay': [0.0, 1e-4, 1e-3]
            }
        }
    
    def setup_device(self):
        """Setup GPU/CPU device"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU")
        return device
    
    def run_hyperparameter_tuning(self, model_type: str = 'cnn', n_trials: int = None, timeout: int = None) -> Dict:
        """
        Run hyperparameter tuning using Optuna
        
        Args:
            model_type: Model type to tune ('cnn' or 'lstm')
            n_trials: Number of trials (from config if None)
            timeout: Timeout in seconds (from config if None)
            
        Returns:
            Dictionary with best parameters and results
        """
        # Get parameters from config
        optuna_cfg = self.cfg.get('optuna', {})
        if n_trials is None:
            n_trials = optuna_cfg.get('n_trials', 50)
        if timeout is None:
            timeout = optuna_cfg.get('timeout', 3600)
        
        logger.info(f"Starting hyperparameter tuning for {model_type}")
        logger.info(f"Trials: {n_trials}, Timeout: {timeout}s")
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            pruner=optuna.pruners.MedianPruner()
        )
        
        # Load data for optimization
        train_df, val_df = self._load_data_for_tuning()
        
        # Define objective function
        def objective(trial):
            return objective_fn(trial, self.cfg, train_df, val_df)
        
        # Run optimization
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        # Save results
        results = self._save_tuning_results(study, model_type)
        
        logger.info(f"Hyperparameter tuning completed")
        logger.info(f"Best value: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")
        
        return results
    
    def _load_data_for_tuning(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load data for hyperparameter tuning"""
        # This is a simplified version - in practice, you'd load actual data
        # For now, create dummy data for demonstration
        logger.info("Loading data for hyperparameter tuning...")
        
        # Create dummy training data
        n_samples = 1000
        train_data = {
            'video_id': ['video_001'] * n_samples,
            'frame': list(range(n_samples)),
            'agent_id': ['agent1'] * n_samples,
            'target_id': ['target1'] * n_samples,
            'behavior': ['approach'] * (n_samples // 2) + ['attack'] * (n_samples // 2)
        }
        train_df = pd.DataFrame(train_data)
        
        # Create dummy validation data
        val_data = {
            'video_id': ['video_002'] * 200,
            'frame': list(range(200)),
            'agent_id': ['agent1'] * 200,
            'target_id': ['target1'] * 200,
            'behavior': ['approach'] * 100 + ['attack'] * 100
        }
        val_df = pd.DataFrame(val_data)
        
        logger.info(f"Loaded {len(train_df)} training samples, {len(val_df)} validation samples")
        return train_df, val_df
    
    def _save_tuning_results(self, study, model_type: str) -> Dict:
        """Save hyperparameter tuning results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save best parameters
        best_params_path = self.model_save_path / f"best_hyperparams_{model_type}_{timestamp}.json"
        results = {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials),
            'model_type': model_type,
            'timestamp': timestamp
        }
        
        with open(best_params_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save all trials with diversity metrics
        trials_path = self.model_save_path / f"hyperparam_trials_{model_type}_{timestamp}.json"
        trials_data = []
        for trial in study.trials:
            trial_data = {
                'number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': trial.state.name
            }
            
            # Add user attributes if available
            if hasattr(trial, 'user_attrs') and trial.user_attrs:
                trial_data['user_attrs'] = trial.user_attrs
            
            trials_data.append(trial_data)
        
        with open(trials_path, 'w') as f:
            json.dump(trials_data, f, indent=2)
        
        # Save best model
        if study.best_params:
            best_model = create_model_from_params(study.best_params, model_type)
            model_filename = f"{model_type}_enhanced_model.pth"
            model_path = self.model_save_path / model_filename
            torch.save(best_model.state_dict(), model_path)
            logger.info(f"Best {model_type} model saved to {model_path}")
        
        logger.info(f"Tuning results saved:")
        logger.info(f"  Best params: {best_params_path}")
        logger.info(f"  All trials: {trials_path}")
        
        return results
