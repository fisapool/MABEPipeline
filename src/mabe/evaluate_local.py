"""
MABE Local Evaluation Module

Refactored from original evaluate_local.py to work with configuration system.
Handles evaluation metrics and F-score calculation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import json
from datetime import datetime

# Import utilities
from .utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_predictions(ground_truth: pd.DataFrame, predictions: pd.DataFrame, use_kaggle_metric: bool = True) -> Dict:
    """
    Evaluate predictions against ground truth
    
    Args:
        ground_truth: Ground truth DataFrame
        predictions: Predictions DataFrame
        use_kaggle_metric: Whether to use Kaggle-compatible metric (default: True)
        
    Returns:
        Dictionary with evaluation metrics
    """
    logger.info("Evaluating predictions...")
    
    # Convert to standard format if needed
    gt_df = standardize_dataframe(ground_truth)
    pred_df = standardize_dataframe(predictions)
    
    # Try Kaggle metric first if requested
    if use_kaggle_metric:
        try:
            from .kaggle_metric import evaluate_with_kaggle_metric, validate_kaggle_compatibility
            
            # Check if data is compatible with Kaggle metric
            if validate_kaggle_compatibility(gt_df, pred_df):
                logger.info("Using Kaggle-compatible metric for evaluation")
                return evaluate_with_kaggle_metric(gt_df, pred_df)
            else:
                logger.warning("Data not compatible with Kaggle metric, falling back to simple metric")
        except ImportError:
            logger.warning("Kaggle metric module not available, falling back to simple metric")
        except Exception as e:
            logger.warning(f"Kaggle metric evaluation failed: {e}, falling back to simple metric")
    
    # Fall back to simple metric
    logger.info("Using simple annotation-level metric for evaluation")
    
    # Calculate metrics
    metrics = {}
    
    # Overall F-score
    overall_f_score = calculate_overall_f_score(gt_df, pred_df)
    metrics['overall_f_score'] = overall_f_score
    
    # Per-behavior F-scores
    behavior_f_scores = calculate_behavior_f_scores(gt_df, pred_df)
    metrics['behavior_f_scores'] = behavior_f_scores
    
    # Per-video F-scores
    video_f_scores = calculate_video_f_scores(gt_df, pred_df)
    metrics['video_f_scores'] = video_f_scores
    
    # Summary statistics
    metrics['summary'] = {
        'total_gt_interactions': len(gt_df),
        'total_pred_interactions': len(pred_df),
        'overall_f_score': overall_f_score,
        'mean_behavior_f_score': np.mean(list(behavior_f_scores.values())),
        'mean_video_f_score': np.mean(list(video_f_scores.values()))
    }
    
    logger.info(f"Evaluation completed: Overall F-score = {overall_f_score:.4f}")
    return metrics


def calculate_video_f_score(predictions: pd.DataFrame, ground_truth: pd.DataFrame) -> float:
    """
    Calculate F-score for a single video
    
    Args:
        predictions: Predictions DataFrame
        ground_truth: Ground truth DataFrame
        
    Returns:
        F-score value
    """
    # Convert to standard format
    pred_df = standardize_dataframe(predictions)
    gt_df = standardize_dataframe(ground_truth)
    
    # Calculate F-score
    f_score = calculate_overall_f_score(gt_df, pred_df)
    return f_score


def calculate_overall_f_score(ground_truth: pd.DataFrame, predictions: pd.DataFrame) -> float:
    """
    Calculate overall F-score between ground truth and predictions
    
    Args:
        ground_truth: Ground truth DataFrame
        predictions: Predictions DataFrame
        
    Returns:
        F-score value
    """
    # Convert to standard format
    gt_df = standardize_dataframe(ground_truth)
    pred_df = standardize_dataframe(predictions)
    
    # Calculate precision and recall
    precision = calculate_precision(gt_df, pred_df)
    recall = calculate_recall(gt_df, pred_df)
    
    # Calculate F-score
    if precision + recall == 0:
        f_score = 0.0
    else:
        f_score = 2 * (precision * recall) / (precision + recall)
    
    return f_score


def calculate_precision(ground_truth: pd.DataFrame, predictions: pd.DataFrame) -> float:
    """
    Calculate precision of predictions
    
    Args:
        ground_truth: Ground truth DataFrame
        predictions: Predictions DataFrame
        
    Returns:
        Precision value
    """
    if len(predictions) == 0:
        return 0.0
    
    # Count true positives and false positives
    tp = 0
    fp = 0
    
    for _, pred in predictions.iterrows():
        # Check if prediction matches any ground truth
        matches = find_matching_ground_truth(pred, ground_truth)
        if matches:
            tp += 1
        else:
            fp += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    return precision


def calculate_recall(ground_truth: pd.DataFrame, predictions: pd.DataFrame) -> float:
    """
    Calculate recall of predictions
    
    Args:
        ground_truth: Ground truth DataFrame
        predictions: Predictions DataFrame
        
    Returns:
        Recall value
    """
    if len(ground_truth) == 0:
        return 0.0
    
    # Count true positives and false negatives
    tp = 0
    fn = 0
    
    for _, gt in ground_truth.iterrows():
        # Check if ground truth matches any prediction
        matches = find_matching_predictions(gt, predictions)
        if matches:
            tp += 1
        else:
            fn += 1
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return recall


def find_matching_ground_truth(prediction: pd.Series, ground_truth: pd.DataFrame) -> List[int]:
    """
    Find ground truth entries that match a prediction
    
    Args:
        prediction: Single prediction row
        ground_truth: Ground truth DataFrame
        
    Returns:
        List of matching ground truth indices
    """
    matches = []
    
    for idx, gt in ground_truth.iterrows():
        if (prediction['video_id'] == gt['video_id'] and
            prediction['agent_id'] == gt['agent_id'] and
            prediction['target_id'] == gt['target_id'] and
            prediction['action'] == gt['action'] and
            frames_overlap(prediction, gt)):
            matches.append(idx)
    
    return matches


def find_matching_predictions(ground_truth: pd.Series, predictions: pd.DataFrame) -> List[int]:
    """
    Find prediction entries that match a ground truth
    
    Args:
        ground_truth: Single ground truth row
        predictions: Predictions DataFrame
        
    Returns:
        List of matching prediction indices
    """
    matches = []
    
    for idx, pred in predictions.iterrows():
        if (pred['video_id'] == ground_truth['video_id'] and
            pred['agent_id'] == ground_truth['agent_id'] and
            pred['target_id'] == ground_truth['target_id'] and
            pred['action'] == ground_truth['action'] and
            frames_overlap(pred, ground_truth)):
            matches.append(idx)
    
    return matches


def frames_overlap(pred: pd.Series, gt: pd.Series) -> bool:
    """
    Check if prediction and ground truth frames overlap
    
    Args:
        pred: Prediction row
        gt: Ground truth row
        
    Returns:
        True if frames overlap
    """
    pred_start = pred['start_frame']
    pred_end = pred['stop_frame']
    gt_start = gt['start_frame']
    gt_end = gt['stop_frame']
    
    # Check for overlap
    return not (pred_end <= gt_start or gt_end <= pred_start)


def calculate_behavior_f_scores(ground_truth: pd.DataFrame, predictions: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate F-scores for each behavior type
    
    Args:
        ground_truth: Ground truth DataFrame
        predictions: Predictions DataFrame
        
    Returns:
        Dictionary with F-scores for each behavior
    """
    behaviors = set(ground_truth['action'].unique()) | set(predictions['action'].unique())
    behavior_f_scores = {}
    
    for behavior in behaviors:
        gt_behavior = ground_truth[ground_truth['action'] == behavior]
        pred_behavior = predictions[predictions['action'] == behavior]
        
        if len(gt_behavior) == 0 and len(pred_behavior) == 0:
            f_score = 1.0  # Perfect score if both are empty
        elif len(gt_behavior) == 0:
            f_score = 0.0  # No ground truth, but predictions exist
        elif len(pred_behavior) == 0:
            f_score = 0.0  # No predictions, but ground truth exists
        else:
            f_score = calculate_overall_f_score(gt_behavior, pred_behavior)
        
        behavior_f_scores[behavior] = f_score
    
    return behavior_f_scores


def calculate_video_f_scores(ground_truth: pd.DataFrame, predictions: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate F-scores for each video
    
    Args:
        ground_truth: Ground truth DataFrame
        predictions: Predictions DataFrame
        
    Returns:
        Dictionary with F-scores for each video
    """
    videos = set(ground_truth['video_id'].unique()) | set(predictions['video_id'].unique())
    video_f_scores = {}
    
    for video_id in videos:
        gt_video = ground_truth[ground_truth['video_id'] == video_id]
        pred_video = predictions[predictions['video_id'] == video_id]
        
        if len(gt_video) == 0 and len(pred_video) == 0:
            f_score = 1.0  # Perfect score if both are empty
        elif len(gt_video) == 0:
            f_score = 0.0  # No ground truth, but predictions exist
        elif len(pred_video) == 0:
            f_score = 0.0  # No predictions, but ground truth exists
        else:
            f_score = calculate_overall_f_score(gt_video, pred_video)
        
        video_f_scores[str(video_id)] = f_score
    
    return video_f_scores


def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize DataFrame format for evaluation
    
    Args:
        df: Input DataFrame
        
    Returns:
        Standardized DataFrame
    """
    # Ensure required columns exist
    required_columns = ['video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame']
    
    # Check if columns exist with different names
    column_mapping = {}
    for col in required_columns:
        if col not in df.columns:
            # Try alternative names
            if col == 'action' and 'behavior' in df.columns:
                column_mapping[col] = 'behavior'
            elif col == 'start_frame' and 'start' in df.columns:
                column_mapping[col] = 'start'
            elif col == 'stop_frame' and 'stop' in df.columns:
                column_mapping[col] = 'stop'
            else:
                logger.warning(f"Required column '{col}' not found in DataFrame")
                continue
    
    # Rename columns if needed
    if column_mapping:
        df = df.rename(columns=column_mapping)
    
    # Ensure all required columns exist
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Convert to appropriate types
    df = df.copy()
    df['video_id'] = df['video_id'].astype(str)
    df['agent_id'] = df['agent_id'].astype(str)
    df['target_id'] = df['target_id'].astype(str)
    df['action'] = df['action'].astype(str)
    df['start_frame'] = df['start_frame'].astype(int)
    df['stop_frame'] = df['stop_frame'].astype(int)
    
    return df


def save_evaluation_results(metrics: Dict, cfg: Dict, output_path: Optional[str] = None) -> str:
    """
    Save evaluation results to file
    
    Args:
        metrics: Evaluation metrics dictionary
        cfg: Configuration dictionary
        output_path: Optional output path
        
    Returns:
        Path to saved results file
    """
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outputs_dir = Path(cfg['paths']['outputs_dir'])
        outputs_dir.mkdir(parents=True, exist_ok=True)
        output_path = outputs_dir / f"evaluation_results_{timestamp}.json"
    
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
    
    metrics_serializable = convert_numpy_types(metrics)
    
    with open(output_path, 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    
    logger.info(f"Evaluation results saved to {output_path}")
    return str(output_path)


def load_ground_truth(cfg: Dict) -> pd.DataFrame:
    """
    Load ground truth data from configuration
    
    Args:
        cfg: Configuration dictionary
        
    Returns:
        Ground truth DataFrame
    """
    from .ground_truth_loader import load_ground_truth_for_evaluation
    
    logger.info("Loading ground truth data...")
    return load_ground_truth_for_evaluation(cfg)


def compare_predictions(predictions1: pd.DataFrame, predictions2: pd.DataFrame, 
                       name1: str = "Predictions 1", name2: str = "Predictions 2") -> Dict:
    """
    Compare two sets of predictions
    
    Args:
        predictions1: First predictions DataFrame
        predictions2: Second predictions DataFrame
        name1: Name for first predictions
        name2: Name for second predictions
        
    Returns:
        Comparison metrics dictionary
    """
    logger.info(f"Comparing {name1} vs {name2}...")
    
    # Standardize both DataFrames
    pred1_df = standardize_dataframe(predictions1)
    pred2_df = standardize_dataframe(predictions2)
    
    # Calculate metrics for each
    metrics1 = evaluate_predictions(pred1_df, pred1_df)  # Self-evaluation
    metrics2 = evaluate_predictions(pred2_df, pred2_df)  # Self-evaluation
    
    # Calculate cross-evaluation
    cross_metrics = {
        f"{name1}_vs_{name2}": calculate_overall_f_score(pred1_df, pred2_df),
        f"{name2}_vs_{name1}": calculate_overall_f_score(pred2_df, pred1_df)
    }
    
    comparison = {
        'predictions1': {
            'name': name1,
            'count': len(pred1_df),
            'f_score': metrics1['overall_f_score']
        },
        'predictions2': {
            'name': name2,
            'count': len(pred2_df),
            'f_score': metrics2['overall_f_score']
        },
        'cross_evaluation': cross_metrics
    }
    
    logger.info(f"Comparison completed: {name1} F-score = {metrics1['overall_f_score']:.4f}, "
                f"{name2} F-score = {metrics2['overall_f_score']:.4f}")
    
    return comparison
