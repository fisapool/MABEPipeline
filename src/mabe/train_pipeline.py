"""
MABE Training Pipeline Orchestrator

High-level training orchestrator that coordinates data loading, preprocessing,
model training, and saving.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging

# Import utilities
from .utils.logger import get_logger
from .utils.seed import set_seed
from .utils.config import load_config

# Import core modules
from .preprocessing import preprocess_data, create_dataset, MABEDataPreprocessor
from .train import build_model, train, train_multiple_models, save_training_results

logger = get_logger(__name__)


def run_training(cfg: Dict, resume: bool = False, checkpoint_path: Optional[str] = None) -> Dict:
    """
    Run the complete training pipeline
    
    Args:
        cfg: Configuration dictionary
        resume: Whether to resume training from checkpoint
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Dictionary with training results
    """
    logger.info("Starting MABE training pipeline...")
    
    # Set random seed for reproducibility
    set_seed(cfg.get('seed', 42))
    
    # Step 1: Data preprocessing
    logger.info("Step 1/5: Data preprocessing")
    frame_labels_df = preprocess_data(cfg)
    
    if frame_labels_df.empty:
        logger.error("No frame labels generated. Check data paths and configuration.")
        return {}
    
    # Step 2: Create datasets and data loaders
    logger.info("Step 2/5: Creating datasets and data loaders")
    train_loader, val_loader, train_dataset, val_dataset = create_dataset(frame_labels_df, cfg)
    
    # Step 3: Build models
    logger.info("Step 3/5: Building models")
    training_cfg = cfg.get('training', {})
    model_type = training_cfg.get('model_type', 'both')
    input_dim = 26  # Fixed for MABE features
    
    models = {}
    if model_type in ['cnn', 'both']:
        models['cnn'] = build_model(cfg, input_dim, 'cnn')
        logger.info("CNN model built")
    
    if model_type in ['lstm', 'both']:
        models['lstm'] = build_model(cfg, input_dim, 'lstm')
        logger.info("LSTM model built")
    
    # Step 4: Train models
    logger.info("Step 4/5: Training models")
    training_results = {}
    
    for model_name, model in models.items():
        logger.info(f"Training {model_name.upper()} model...")
        
        # Train single model
        result = train(model, train_loader, val_loader, cfg, 
                      resume_checkpoint=checkpoint_path if resume else None)
        training_results[model_name] = result
        
        logger.info(f"{model_name.upper()} training completed: "
                   f"Best Val Acc = {result['best_val_acc']:.2f}%")
    
    # Step 5: Save results and metrics
    logger.info("Step 5/5: Saving results and metrics")
    metrics_path = save_training_results(training_results, cfg)
    
    # Create summary
    summary = {
        'training_completed': True,
        'models_trained': list(models.keys()),
        'total_samples': len(frame_labels_df),
        'training_samples': len(train_dataset),
        'validation_samples': len(val_dataset),
        'metrics_path': metrics_path,
        'timestamp': datetime.now().isoformat()
    }
    
    # Add model-specific results
    for model_name, result in training_results.items():
        summary[f'{model_name}_best_val_acc'] = result['best_val_acc']
        summary[f'{model_name}_epochs_trained'] = result['epochs_trained']
        summary[f'{model_name}_model_path'] = result['model_path']
    
    logger.info("Training pipeline completed successfully!")
    logger.info(f"Models trained: {list(models.keys())}")
    logger.info(f"Training metrics saved to: {metrics_path}")
    
    return summary


def run_training_with_validation(cfg: Dict, test_size: float = 0.2) -> Dict:
    """
    Run training with cross-validation
    
    Args:
        cfg: Configuration dictionary
        test_size: Fraction of data to use for testing
        
    Returns:
        Dictionary with cross-validation results
    """
    logger.info("Starting training with cross-validation...")
    
    # Set random seed
    set_seed(cfg.get('seed', 42))
    
    # Load and preprocess data
    frame_labels_df = preprocess_data(cfg)
    
    if frame_labels_df.empty:
        logger.error("No frame labels generated.")
        return {}
    
    # Split data
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(frame_labels_df, test_size=test_size, random_state=42)
    
    logger.info(f"Data split: {len(train_df)} training, {len(val_df)} validation")
    
    # Create datasets
    train_loader, val_loader, train_dataset, val_dataset = create_dataset(train_df, cfg)
    
    # Train models
    training_cfg = cfg.get('training', {})
    model_type = training_cfg.get('model_type', 'both')
    input_dim = 26
    
    results = {}
    
    if model_type in ['cnn', 'both']:
        logger.info("Training CNN with cross-validation...")
        cnn_model = build_model(cfg, input_dim, 'cnn')
        cnn_result = train(cnn_model, train_loader, val_loader, cfg)
        results['cnn'] = cnn_result
    
    if model_type in ['lstm', 'both']:
        logger.info("Training LSTM with cross-validation...")
        lstm_model = build_model(cfg, input_dim, 'lstm')
        lstm_result = train(lstm_model, train_loader, val_loader, cfg)
        results['lstm'] = lstm_result
    
    # Save results
    metrics_path = save_training_results(results, cfg)
    
    logger.info("Cross-validation training completed!")
    return {
        'cross_validation_completed': True,
        'results': results,
        'metrics_path': metrics_path
    }


def run_training_with_sampling(cfg: Dict, sample_size: int = 1000) -> Dict:
    """
    Run training on a sampled subset of data (useful for testing)
    
    Args:
        cfg: Configuration dictionary
        sample_size: Number of samples to use
        
    Returns:
        Dictionary with training results
    """
    logger.info(f"Starting training with sampling (sample_size={sample_size})...")
    
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
    
    # Train models
    training_cfg = cfg.get('training', {})
    model_type = training_cfg.get('model_type', 'both')
    input_dim = 26
    
    results = {}
    
    if model_type in ['cnn', 'both']:
        logger.info("Training CNN with sampling...")
        cnn_model = build_model(cfg, input_dim, 'cnn')
        cnn_result = train(cnn_model, train_loader, val_loader, cfg)
        results['cnn'] = cnn_result
    
    if model_type in ['lstm', 'both']:
        logger.info("Training LSTM with sampling...")
        lstm_model = build_model(cfg, input_dim, 'lstm')
        lstm_result = train(lstm_model, train_loader, val_loader, cfg)
        results['lstm'] = lstm_result
    
    # Save results
    metrics_path = save_training_results(results, cfg)
    
    logger.info("Sampling training completed!")
    return {
        'sampling_completed': True,
        'sample_size': sample_size,
        'results': results,
        'metrics_path': metrics_path
    }


def validate_training_setup(cfg: Dict) -> bool:
    """
    Validate training configuration and data availability
    
    Args:
        cfg: Configuration dictionary
        
    Returns:
        True if setup is valid, False otherwise
    """
    logger.info("Validating training setup...")
    
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
    models_dir = Path(cfg.get('paths', {}).get('models_dir', 'outputs/models'))
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Check device availability
    device_str = cfg.get('device', {}).get('device_str', 'cuda:0')
    if 'cuda' in device_str:
        import torch
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            cfg['device']['device_str'] = 'cpu'
            cfg['device']['use_cuda'] = False
    
    logger.info("Training setup validation completed successfully!")
    return True


def get_training_summary(cfg: Dict) -> Dict:
    """
    Get summary of training configuration and data
    
    Args:
        cfg: Configuration dictionary
        
    Returns:
        Dictionary with training summary
    """
    summary = {
        'configuration': {
            'model_type': cfg.get('training', {}).get('model_type', 'both'),
            'epochs': cfg.get('training', {}).get('epochs', 30),
            'batch_size': cfg.get('training', {}).get('batch_size', 32),
            'learning_rate': cfg.get('training', {}).get('learning_rate', 0.001),
            'device': cfg.get('device', {}).get('device_str', 'cuda:0')
        },
        'data': {
            'dataset_path': cfg.get('dataset', {}).get('path', ''),
            'train_csv': cfg.get('dataset', {}).get('train_csv', 'train.csv'),
            'max_videos': cfg.get('training', {}).get('max_videos', 5)
        },
        'outputs': {
            'models_dir': cfg.get('paths', {}).get('models_dir', 'outputs/models'),
            'logs_dir': cfg.get('paths', {}).get('logs_dir', 'outputs/logs')
        }
    }
    
    return summary
