"""
MABE Evaluation Pipeline Orchestrator

High-level evaluation orchestrator that coordinates prediction loading,
ground truth loading, metric calculation, and result saving.
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
from .evaluate_local import (
    evaluate_predictions, calculate_video_f_score, 
    save_evaluation_results, load_ground_truth, compare_predictions
)

logger = get_logger(__name__)


def run_local_evaluation(cfg: Dict, predictions_csv: str) -> Dict:
    """
    Run local evaluation on predictions
    
    Args:
        cfg: Configuration dictionary
        predictions_csv: Path to predictions CSV file
        
    Returns:
        Dictionary with evaluation results
    """
    logger.info("Starting MABE local evaluation...")
    
    # Set random seed for reproducibility
    set_seed(cfg.get('seed', 42))
    
    # Step 1: Load predictions
    logger.info("Step 1/4: Loading predictions")
    predictions_df = load_predictions(cfg, predictions_csv)
    
    if predictions_df.empty:
        logger.error("No predictions loaded. Check file path and format.")
        return {}
    
    # Step 2: Load ground truth
    logger.info("Step 2/4: Loading ground truth")
    ground_truth_df = load_ground_truth(cfg)
    
    if ground_truth_df.empty:
        logger.warning("No ground truth available. Skipping evaluation.")
        return {'evaluation_skipped': True, 'reason': 'No ground truth available'}
    
    # Step 3: Calculate metrics
    logger.info("Step 3/4: Calculating evaluation metrics")
    metrics = evaluate_predictions(ground_truth_df, predictions_df)
    
    # Step 4: Save results
    logger.info("Step 4/4: Saving evaluation results")
    results_path = save_evaluation_results(metrics, cfg)
    
    logger.info("Local evaluation completed successfully!")
    logger.info(f"Overall F-score: {metrics.get('overall_f_score', 0.0):.4f}")
    logger.info(f"Results saved to: {results_path}")
    
    return metrics


def load_predictions(cfg: Dict, predictions_csv: str) -> pd.DataFrame:
    """
    Load predictions from CSV file
    
    Args:
        cfg: Configuration dictionary
        predictions_csv: Path to predictions CSV file
        
    Returns:
        Predictions DataFrame
    """
    predictions_path = Path(predictions_csv)
    
    if not predictions_path.exists():
        logger.error(f"Predictions file not found: {predictions_path}")
        return pd.DataFrame()
    
    logger.info(f"Loading predictions from {predictions_path}")
    
    try:
        predictions_df = pd.read_csv(predictions_path)
        logger.info(f"Loaded {len(predictions_df)} predictions")
        return predictions_df
    except Exception as e:
        logger.error(f"Error loading predictions: {e}")
        return pd.DataFrame()


def run_evaluation_with_ground_truth(cfg: Dict, predictions_csv: str, ground_truth_csv: str) -> Dict:
    """
    Run evaluation with specific ground truth file
    
    Args:
        cfg: Configuration dictionary
        predictions_csv: Path to predictions CSV file
        ground_truth_csv: Path to ground truth CSV file
        
    Returns:
        Dictionary with evaluation results
    """
    logger.info("Starting evaluation with specific ground truth...")
    
    # Set random seed
    set_seed(cfg.get('seed', 42))
    
    # Load predictions
    predictions_df = load_predictions(cfg, predictions_csv)
    if predictions_df.empty:
        return {}
    
    # Load ground truth
    ground_truth_path = Path(ground_truth_csv)
    if not ground_truth_path.exists():
        logger.error(f"Ground truth file not found: {ground_truth_path}")
        return {}
    
    logger.info(f"Loading ground truth from {ground_truth_path}")
    try:
        ground_truth_df = pd.read_csv(ground_truth_path)
        logger.info(f"Loaded {len(ground_truth_df)} ground truth entries")
    except Exception as e:
        logger.error(f"Error loading ground truth: {e}")
        return {}
    
    # Calculate metrics
    metrics = evaluate_predictions(ground_truth_df, predictions_df)
    
    # Save results
    results_path = save_evaluation_results(metrics, cfg)
    
    logger.info("Evaluation completed successfully!")
    logger.info(f"Overall F-score: {metrics.get('overall_f_score', 0.0):.4f}")
    
    return metrics


def run_comparative_evaluation(cfg: Dict, predictions1_csv: str, predictions2_csv: str, 
                              name1: str = "Predictions 1", name2: str = "Predictions 2") -> Dict:
    """
    Run comparative evaluation between two prediction sets
    
    Args:
        cfg: Configuration dictionary
        predictions1_csv: Path to first predictions CSV file
        predictions2_csv: Path to second predictions CSV file
        name1: Name for first predictions
        name2: Name for second predictions
        
    Returns:
        Dictionary with comparison results
    """
    logger.info(f"Starting comparative evaluation: {name1} vs {name2}")
    
    # Set random seed
    set_seed(cfg.get('seed', 42))
    
    # Load both prediction sets
    predictions1_df = load_predictions(cfg, predictions1_csv)
    predictions2_df = load_predictions(cfg, predictions2_csv)
    
    if predictions1_df.empty or predictions2_df.empty:
        logger.error("Could not load one or both prediction sets")
        return {}
    
    # Run comparison
    comparison_results = compare_predictions(
        predictions1_df, predictions2_df, name1, name2
    )
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outputs_dir = Path(cfg.get('paths', {}).get('outputs_dir', 'outputs'))
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = outputs_dir / f"comparative_evaluation_{timestamp}.json"
    
    with open(results_path, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    logger.info(f"Comparative evaluation completed: {results_path}")
    return comparison_results


def run_video_evaluation(cfg: Dict, predictions_csv: str, video_id: str) -> Dict:
    """
    Run evaluation for a specific video
    
    Args:
        cfg: Configuration dictionary
        predictions_csv: Path to predictions CSV file
        video_id: Video ID to evaluate
        
    Returns:
        Dictionary with video-specific results
    """
    logger.info(f"Starting evaluation for video {video_id}")
    
    # Set random seed
    set_seed(cfg.get('seed', 42))
    
    # Load predictions
    predictions_df = load_predictions(cfg, predictions_csv)
    if predictions_df.empty:
        return {}
    
    # Filter predictions for specific video
    video_predictions = predictions_df[predictions_df['video_id'] == video_id]
    
    if video_predictions.empty:
        logger.warning(f"No predictions found for video {video_id}")
        return {'video_id': video_id, 'predictions_count': 0}
    
    # Load ground truth for this video
    ground_truth_df = load_ground_truth(cfg)
    if not ground_truth_df.empty:
        ground_truth_df = ground_truth_df[ground_truth_df['video_id'] == video_id]
    
    if ground_truth_df.empty:
        logger.warning(f"No ground truth available for video {video_id}")
        return {
            'video_id': video_id,
            'predictions_count': len(video_predictions),
            'ground_truth_available': False
        }
    
    # Calculate video-specific metrics
    video_f_score = calculate_video_f_score(ground_truth_df, video_predictions)
    
    # Calculate additional metrics
    metrics = evaluate_predictions(ground_truth_df, video_predictions)
    
    results = {
        'video_id': video_id,
        'predictions_count': len(video_predictions),
        'ground_truth_count': len(ground_truth_df),
        'video_f_score': video_f_score,
        'overall_f_score': metrics.get('overall_f_score', 0.0),
        'behavior_f_scores': metrics.get('behavior_f_scores', {}),
        'ground_truth_available': True
    }
    
    logger.info(f"Video {video_id} evaluation completed: F-score = {video_f_score:.4f}")
    return results


def run_behavior_evaluation(cfg: Dict, predictions_csv: str, behavior: str) -> Dict:
    """
    Run evaluation for a specific behavior
    
    Args:
        cfg: Configuration dictionary
        predictions_csv: Path to predictions CSV file
        behavior: Behavior to evaluate
        
    Returns:
        Dictionary with behavior-specific results
    """
    logger.info(f"Starting evaluation for behavior: {behavior}")
    
    # Set random seed
    set_seed(cfg.get('seed', 42))
    
    # Load predictions
    predictions_df = load_predictions(cfg, predictions_csv)
    if predictions_df.empty:
        return {}
    
    # Filter predictions for specific behavior
    behavior_predictions = predictions_df[predictions_df['action'] == behavior]
    
    if behavior_predictions.empty:
        logger.warning(f"No predictions found for behavior {behavior}")
        return {'behavior': behavior, 'predictions_count': 0}
    
    # Load ground truth for this behavior
    ground_truth_df = load_ground_truth(cfg)
    if not ground_truth_df.empty:
        ground_truth_df = ground_truth_df[ground_truth_df['action'] == behavior]
    
    if ground_truth_df.empty:
        logger.warning(f"No ground truth available for behavior {behavior}")
        return {
            'behavior': behavior,
            'predictions_count': len(behavior_predictions),
            'ground_truth_available': False
        }
    
    # Calculate behavior-specific metrics
    metrics = evaluate_predictions(ground_truth_df, behavior_predictions)
    
    results = {
        'behavior': behavior,
        'predictions_count': len(behavior_predictions),
        'ground_truth_count': len(ground_truth_df),
        'overall_f_score': metrics.get('overall_f_score', 0.0),
        'precision': metrics.get('precision', 0.0),
        'recall': metrics.get('recall', 0.0),
        'ground_truth_available': True
    }
    
    logger.info(f"Behavior {behavior} evaluation completed: F-score = {results['overall_f_score']:.4f}")
    return results


def validate_evaluation_setup(cfg: Dict) -> bool:
    """
    Validate evaluation configuration and data availability
    
    Args:
        cfg: Configuration dictionary
        
    Returns:
        True if setup is valid, False otherwise
    """
    logger.info("Validating evaluation setup...")
    
    # Check output directories
    outputs_dir = Path(cfg.get('paths', {}).get('outputs_dir', 'outputs'))
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if ground truth is available
    ground_truth_df = load_ground_truth(cfg)
    if ground_truth_df.empty:
        logger.warning("No ground truth available. Evaluation will be limited.")
    
    logger.info("Evaluation setup validation completed successfully!")
    return True


def get_evaluation_summary(cfg: Dict) -> Dict:
    """
    Get summary of evaluation configuration
    
    Args:
        cfg: Configuration dictionary
        
    Returns:
        Dictionary with evaluation summary
    """
    summary = {
        'configuration': {
            'seed': cfg.get('seed', 42),
            'device': cfg.get('device', {}).get('device_str', 'cuda:0')
        },
        'data': {
            'dataset_path': cfg.get('dataset', {}).get('path', ''),
            'ground_truth_available': not load_ground_truth(cfg).empty
        },
        'outputs': {
            'outputs_dir': cfg.get('paths', {}).get('outputs_dir', 'outputs'),
            'submissions_dir': cfg.get('paths', {}).get('submissions_dir', 'outputs/submissions')
        }
    }
    
    return summary
