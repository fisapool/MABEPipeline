"""
MABE Hyperparameter Tuning Pipeline Orchestrator

High-level tuning orchestrator that coordinates Optuna optimization,
model training, and parameter saving.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging
import optuna

# Import utilities
from .utils.logger import get_logger
from .utils.seed import set_seed
from .utils.config import load_config

# Import core modules
from .hyperparameter import MABEHyperparameterTuner, objective_fn
from .preprocessing import preprocess_data, create_dataset
from .train import build_model, train

logger = get_logger(__name__)


def run_optuna(cfg: Dict, n_trials: Optional[int] = None, timeout: Optional[int] = None) -> Dict:
    """
    Run Optuna hyperparameter optimization
    
    Args:
        cfg: Configuration dictionary
        n_trials: Number of trials (from config if None)
        timeout: Timeout in seconds (from config if None)
        
    Returns:
        Dictionary with optimization results
    """
    logger.info("Starting MABE hyperparameter optimization...")
    
    # Set random seed for reproducibility
    set_seed(cfg.get('seed', 42))
    
    # Get parameters from config
    optuna_cfg = cfg.get('optuna', {})
    if n_trials is None:
        n_trials = optuna_cfg.get('n_trials', 50)
    if timeout is None:
        timeout = optuna_cfg.get('timeout', 3600)
    
    # Check if CI mode is enabled
    ci_mode = optuna_cfg.get('ci_mode', False)
    if ci_mode:
        n_trials = min(n_trials, 10)  # Limit trials in CI
        timeout = min(timeout, 1800)  # Limit timeout in CI
        logger.info("CI mode enabled: Reduced trials and timeout")
    
    # Step 1: Load and preprocess data
    logger.info("Step 1/4: Loading and preprocessing data")
    frame_labels_df = preprocess_data(cfg)
    
    if frame_labels_df.empty:
        logger.error("No frame labels generated. Check data paths and configuration.")
        return {}
    
    # Step 2: Create datasets
    logger.info("Step 2/4: Creating datasets")
    train_loader, val_loader, train_dataset, val_dataset = create_dataset(frame_labels_df, cfg)
    
    # Step 3: Run optimization
    logger.info("Step 3/4: Running hyperparameter optimization")
    tuner = MABEHyperparameterTuner(cfg)
    
    # Get model type to optimize
    model_type = cfg.get('training', {}).get('model_type', 'cnn')
    if model_type == 'both':
        model_type = 'cnn'  # Default to CNN for optimization
    
    # Run optimization
    results = tuner.run_hyperparameter_tuning(
        model_type=model_type,
        n_trials=n_trials,
        timeout=timeout
    )
    
    # Step 4: Save results
    logger.info("Step 4/4: Saving optimization results")
    results_path = save_optimization_results(results, cfg)
    
    logger.info("Hyperparameter optimization completed successfully!")
    logger.info(f"Best parameters: {results.get('best_params', {})}")
    logger.info(f"Best value: {results.get('best_value', 0.0):.4f}")
    logger.info(f"Results saved to: {results_path}")
    
    return results


def run_optuna_with_validation(cfg: Dict, n_splits: int = 3) -> Dict:
    """
    Run Optuna optimization with cross-validation
    
    Args:
        cfg: Configuration dictionary
        n_splits: Number of cross-validation splits
        
    Returns:
        Dictionary with cross-validation results
    """
    logger.info(f"Starting Optuna optimization with {n_splits}-fold cross-validation...")
    
    # Set random seed
    set_seed(cfg.get('seed', 42))
    
    # Load and preprocess data
    frame_labels_df = preprocess_data(cfg)
    
    if frame_labels_df.empty:
        logger.error("No frame labels generated.")
        return {}
    
    # Split data for cross-validation
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Get behavior labels for stratification
    behavior_map = {
        'approach': 0, 'attack': 1, 'avoid': 2, 'chase': 3,
        'chaseattack': 4, 'submit': 5, 'rear': 6, 'shepherd': 7
    }
    labels = frame_labels_df['behavior'].map(behavior_map).fillna(0)
    
    # Run cross-validation
    cv_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(frame_labels_df, labels)):
        logger.info(f"Running fold {fold + 1}/{n_splits}")
        
        # Split data
        train_df = frame_labels_df.iloc[train_idx]
        val_df = frame_labels_df.iloc[val_idx]
        
        # Create datasets
        train_loader, val_loader, train_dataset, val_dataset = create_dataset(train_df, cfg)
        
        # Run optimization for this fold
        fold_results = run_optuna_fold(cfg, train_loader, val_loader, fold)
        cv_results.append(fold_results)
    
    # Aggregate results
    aggregated_results = aggregate_cv_results(cv_results)
    
    # Save results
    results_path = save_optimization_results(aggregated_results, cfg)
    
    logger.info("Cross-validation optimization completed!")
    return aggregated_results


def run_optuna_fold(cfg: Dict, train_loader, val_loader, fold: int) -> Dict:
    """
    Run Optuna optimization for a single fold
    
    Args:
        cfg: Configuration dictionary
        train_loader: Training data loader
        val_loader: Validation data loader
        fold: Fold number
        
    Returns:
        Dictionary with fold results
    """
    # Create study for this fold
    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner()
    )
    
    # Define objective function for this fold
    def objective(trial):
        return objective_fn(trial, cfg, train_loader, val_loader)
    
    # Run optimization
    n_trials = cfg.get('optuna', {}).get('n_trials', 50)
    study.optimize(objective, n_trials=n_trials)
    
    return {
        'fold': fold,
        'best_params': study.best_params,
        'best_value': study.best_value,
        'n_trials': len(study.trials)
    }


def aggregate_cv_results(cv_results: List[Dict]) -> Dict:
    """
    Aggregate cross-validation results
    
    Args:
        cv_results: List of fold results
        
    Returns:
        Aggregated results dictionary
    """
    # Calculate mean and std of best values
    best_values = [result['best_value'] for result in cv_results]
    mean_best_value = np.mean(best_values)
    std_best_value = np.std(best_values)
    
    # Find most common best parameters
    all_params = [result['best_params'] for result in cv_results]
    param_counts = {}
    
    for params in all_params:
        for key, value in params.items():
            if key not in param_counts:
                param_counts[key] = {}
            if value not in param_counts[key]:
                param_counts[key][value] = 0
            param_counts[key][value] += 1
    
    # Get most common parameters
    best_params = {}
    for key, value_counts in param_counts.items():
        best_params[key] = max(value_counts, key=value_counts.get)
    
    return {
        'cv_results': cv_results,
        'mean_best_value': mean_best_value,
        'std_best_value': std_best_value,
        'best_params': best_params,
        'n_folds': len(cv_results)
    }


def save_optimization_results(results: Dict, cfg: Dict) -> str:
    """
    Save optimization results to file
    
    Args:
        results: Optimization results dictionary
        cfg: Configuration dictionary
        
    Returns:
        Path to saved results file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outputs_dir = Path(cfg.get('paths', {}).get('outputs_dir', 'outputs'))
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    results_path = outputs_dir / f"optuna_results_{timestamp}.json"
    
    # Convert numpy types to Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    results_serializable = convert_numpy_types(results)
    
    with open(results_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    logger.info(f"Optimization results saved to {results_path}")
    return str(results_path)


def run_optuna_with_sampling(cfg: Dict, sample_size: int = 1000) -> Dict:
    """
    Run Optuna optimization on a sampled subset of data
    
    Args:
        cfg: Configuration dictionary
        sample_size: Number of samples to use
        
    Returns:
        Dictionary with optimization results
    """
    logger.info(f"Starting Optuna optimization with sampling (sample_size={sample_size})...")
    
    # Set random seed
    set_seed(cfg.get('seed', 42))
    
    # Load and preprocess data
    frame_labels_df = preprocess_data(cfg)
    
    if frame_labels_df.empty:
        logger.error("No frame labels generated.")
        return {}
    
    # Sample data
    if len(frame_labels_df) > sample_size:
        frame_labels_df = frame_labels_df.sample(n=sample_size, random_state=42)
        logger.info(f"Sampled {sample_size} samples from {len(frame_labels_df)} total")
    
    # Create datasets
    train_loader, val_loader, train_dataset, val_dataset = create_dataset(frame_labels_df, cfg)
    
    # Run optimization
    results = run_optuna_fold(cfg, train_loader, val_loader, 0)
    
    # Save results
    results_path = save_optimization_results(results, cfg)
    
    logger.info("Sampling optimization completed!")
    return results


def validate_tuning_setup(cfg: Dict) -> bool:
    """
    Validate tuning configuration and data availability
    
    Args:
        cfg: Configuration dictionary
        
    Returns:
        True if setup is valid, False otherwise
    """
    logger.info("Validating tuning setup...")
    
    # Check required paths
    dataset_path = Path(cfg.get('dataset', {}).get('path', ''))
    if not dataset_path.exists():
        logger.error(f"Dataset path does not exist: {dataset_path}")
        return False
    
    # Check required files
    train_csv = dataset_path / cfg.get('dataset', {}).get('train_csv', 'train.csv')
    if not train_csv.exists():
        logger.error(f"Train CSV not found: {train_csv}")
        return False
    
    # Check output directories
    outputs_dir = Path(cfg.get('paths', {}).get('outputs_dir', 'outputs'))
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # Check Optuna configuration
    optuna_cfg = cfg.get('optuna', {})
    n_trials = optuna_cfg.get('n_trials', 50)
    timeout = optuna_cfg.get('timeout', 3600)
    
    if n_trials <= 0:
        logger.error("Number of trials must be positive")
        return False
    
    if timeout <= 0:
        logger.error("Timeout must be positive")
        return False
    
    logger.info("Tuning setup validation completed successfully!")
    return True


def get_tuning_summary(cfg: Dict) -> Dict:
    """
    Get summary of tuning configuration
    
    Args:
        cfg: Configuration dictionary
        
    Returns:
        Dictionary with tuning summary
    """
    optuna_cfg = cfg.get('optuna', {})
    
    summary = {
        'configuration': {
            'n_trials': optuna_cfg.get('n_trials', 50),
            'timeout': optuna_cfg.get('timeout', 3600),
            'ci_mode': optuna_cfg.get('ci_mode', False),
            'n_splits': optuna_cfg.get('n_splits', 3)
        },
        'data': {
            'dataset_path': cfg.get('dataset', {}).get('path', ''),
            'train_csv': cfg.get('dataset', {}).get('train_csv', 'train.csv'),
            'max_videos': cfg.get('training', {}).get('max_videos', 5)
        },
        'outputs': {
            'outputs_dir': cfg.get('paths', {}).get('outputs_dir', 'outputs'),
            'models_dir': cfg.get('paths', {}).get('models_dir', 'outputs/models')
        }
    }
    
    return summary
