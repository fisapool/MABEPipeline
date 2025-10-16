"""
Kaggle-Compatible Evaluation Metric for MABE Challenge

Implements the exact F-beta metric used by Kaggle for the MABE competition.
This uses frame-level evaluation (not annotation-level) and supports lab_id
and behaviors_labeled fields as required by the competition.
"""

import json
from collections import defaultdict
from typing import Dict, List, Optional, Union
import pandas as pd
import polars as pl
import numpy as np
from pathlib import Path

from .utils.logger import get_logger

logger = get_logger(__name__)


class HostVisibleError(Exception):
    """Error that should be visible to the host in Kaggle environment"""
    pass


def single_lab_f1(lab_solution: pl.DataFrame, lab_submission: pl.DataFrame, beta: float = 1) -> float:
    """
    Calculate F1 score for a single lab using frame-level evaluation.
    
    This is the core function that implements Kaggle's exact metric logic.
    
    Args:
        lab_solution: Ground truth DataFrame for one lab (with lab_id, behaviors_labeled)
        lab_submission: Predictions DataFrame for one lab
        beta: F-beta parameter (1.0 for F1 score)
        
    Returns:
        F1 score for this lab
    """
    label_frames: defaultdict[str, set[int]] = defaultdict(set)
    prediction_frames: defaultdict[str, set[int]] = defaultdict(set)

    # Build ground truth frame sets
    for row in lab_solution.to_dicts():
        label_frames[row['label_key']].update(range(row['start_frame'], row['stop_frame']))

    # Process predictions with active labels filtering
    for video in lab_solution['video_id'].unique():
        active_labels: str = lab_solution.filter(pl.col('video_id') == video)['behaviors_labeled'].first()
        active_labels: set[str] = set(json.loads(active_labels))
        predicted_mouse_pairs: defaultdict[str, set[int]] = defaultdict(set)

        for row in lab_submission.filter(pl.col('video_id') == video).to_dicts():
            # Only evaluate predictions for behaviors that were actually labeled
            prediction_key = ','.join([str(row['agent_id']), str(row['target_id']), row['action']])
            if prediction_key not in active_labels:
                continue

            new_frames = set(range(row['start_frame'], row['stop_frame']))
            # Remove frames already predicted for this key
            new_frames = new_frames.difference(prediction_frames[row['prediction_key']])
            prediction_pair = ','.join([str(row['agent_id']), str(row['target_id'])])
            
            # Check for multiple predictions from same agent/target pair in same frame
            if predicted_mouse_pairs[prediction_pair].intersection(new_frames):
                raise HostVisibleError('Multiple predictions for the same frame from one agent/target pair')
            
            prediction_frames[row['prediction_key']].update(new_frames)
            predicted_mouse_pairs[prediction_pair].update(new_frames)

    # Calculate frame-level precision and recall
    tps = defaultdict(int)
    fns = defaultdict(int)
    fps = defaultdict(int)
    
    for key, pred_frames in prediction_frames.items():
        action = key.split('_')[-1]
        matched_label_frames = label_frames[key]
        tps[action] += len(pred_frames.intersection(matched_label_frames))
        fns[action] += len(matched_label_frames.difference(pred_frames))
        fps[action] += len(pred_frames.difference(matched_label_frames))

    # Handle missing predictions for ground truth
    distinct_actions = set()
    for key, frames in label_frames.items():
        action = key.split('_')[-1]
        distinct_actions.add(action)
        if key not in prediction_frames:
            fns[action] += len(frames)

    # Calculate F1 scores per action
    action_f1s = []
    for action in distinct_actions:
        if tps[action] + fns[action] + fps[action] == 0:
            action_f1s.append(0)
        else:
            action_f1s.append((1 + beta**2) * tps[action] / ((1 + beta**2) * tps[action] + beta**2 * fns[action] + fps[action]))
    
    return sum(action_f1s) / len(action_f1s) if action_f1s else 0.0


def mouse_fbeta(solution: pd.DataFrame, submission: pd.DataFrame, beta: float = 1) -> float:
    """
    Calculate F-beta score for mouse behavior predictions.
    
    This is the main function that matches Kaggle's evaluation exactly.
    
    Args:
        solution: Ground truth DataFrame with lab_id, behaviors_labeled columns
        submission: Predictions DataFrame
        beta: F-beta parameter (1.0 for F1 score)
        
    Returns:
        F-beta score
    """
    if len(solution) == 0 or len(submission) == 0:
        raise ValueError('Missing solution or submission data')

    expected_cols = ['video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame']
    for col in expected_cols:
        if col not in solution.columns:
            raise ValueError(f'Solution is missing column {col}')
        if col not in submission.columns:
            raise ValueError(f'Submission is missing column {col}')

    # Convert to polars DataFrames
    solution: pl.DataFrame = pl.DataFrame(solution)
    submission: pl.DataFrame = pl.DataFrame(submission)
    
    # Validate frame ranges
    assert (solution['start_frame'] <= solution['stop_frame']).all()
    assert (submission['start_frame'] <= submission['stop_frame']).all()
    
    # Filter submission to only include videos in solution
    solution_videos = set(solution['video_id'].unique())
    submission = submission.filter(pl.col('video_id').is_in(solution_videos))

    # Create label and prediction keys
    solution = solution.with_columns(
        pl.concat_str(
            [
                pl.col('video_id').cast(pl.Utf8),
                pl.col('agent_id').cast(pl.Utf8),
                pl.col('target_id').cast(pl.Utf8),
                pl.col('action'),
            ],
            separator='_',
        ).alias('label_key'),
    )
    submission = submission.with_columns(
        pl.concat_str(
            [
                pl.col('video_id').cast(pl.Utf8),
                pl.col('agent_id').cast(pl.Utf8),
                pl.col('target_id').cast(pl.Utf8),
                pl.col('action'),
            ],
            separator='_',
        ).alias('prediction_key'),
    )

    # Calculate F1 score for each lab and average
    lab_scores = []
    for lab in solution['lab_id'].unique():
        lab_solution = solution.filter(pl.col('lab_id') == lab).clone()
        lab_videos = set(lab_solution['video_id'].unique())
        lab_submission = submission.filter(pl.col('video_id').is_in(lab_videos)).clone()
        lab_scores.append(single_lab_f1(lab_solution, lab_submission, beta=beta))

    return sum(lab_scores) / len(lab_scores) if lab_scores else 0.0


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str, beta: float = 1) -> float:
    """
    F1 score for the MABe Challenge - main entry point for Kaggle.
    
    Args:
        solution: Ground truth DataFrame
        submission: Predictions DataFrame  
        row_id_column_name: Name of row ID column to drop
        beta: F-beta parameter
        
    Returns:
        F-beta score
    """
    solution = solution.drop(row_id_column_name, axis='columns', errors='ignore')
    submission = submission.drop(row_id_column_name, axis='columns', errors='ignore')
    return mouse_fbeta(solution, submission, beta=beta)


def evaluate_with_kaggle_metric(ground_truth: pd.DataFrame, predictions: pd.DataFrame, 
                               beta: float = 1.0) -> Dict:
    """
    Evaluate predictions using Kaggle's exact metric.
    
    Args:
        ground_truth: Ground truth DataFrame (must have lab_id, behaviors_labeled)
        predictions: Predictions DataFrame
        beta: F-beta parameter
        
    Returns:
        Dictionary with evaluation results
    """
    logger.info("Evaluating with Kaggle-compatible metric...")
    
    # Validate required columns
    required_gt_cols = ['video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame', 'lab_id', 'behaviors_labeled']
    missing_cols = [col for col in required_gt_cols if col not in ground_truth.columns]
    if missing_cols:
        raise ValueError(f"Ground truth missing required columns: {missing_cols}")
    
    required_pred_cols = ['video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame']
    missing_cols = [col for col in required_pred_cols if col not in predictions.columns]
    if missing_cols:
        raise ValueError(f"Predictions missing required columns: {missing_cols}")
    
    # Calculate overall F-beta score
    overall_f_score = mouse_fbeta(ground_truth, predictions, beta=beta)
    
    # Calculate per-behavior F-scores
    behavior_f_scores = {}
    behaviors = set(ground_truth['action'].unique()) | set(predictions['action'].unique())
    
    for behavior in behaviors:
        gt_behavior = ground_truth[ground_truth['action'] == behavior]
        pred_behavior = predictions[predictions['action'] == behavior]
        
        if len(gt_behavior) == 0 and len(pred_behavior) == 0:
            behavior_f_scores[behavior] = 1.0
        elif len(gt_behavior) == 0 or len(pred_behavior) == 0:
            behavior_f_scores[behavior] = 0.0
        else:
            behavior_f_scores[behavior] = mouse_fbeta(gt_behavior, pred_behavior, beta=beta)
    
    # Calculate per-video F-scores
    video_f_scores = {}
    videos = set(ground_truth['video_id'].unique()) | set(predictions['video_id'].unique())
    
    for video_id in videos:
        gt_video = ground_truth[ground_truth['video_id'] == video_id]
        pred_video = predictions[predictions['video_id'] == video_id]
        
        if len(gt_video) == 0 and len(pred_video) == 0:
            video_f_scores[str(video_id)] = 1.0
        elif len(gt_video) == 0 or len(pred_video) == 0:
            video_f_scores[str(video_id)] = 0.0
        else:
            video_f_scores[str(video_id)] = mouse_fbeta(gt_video, pred_video, beta=beta)
    
    # Calculate per-lab F-scores
    lab_f_scores = {}
    labs = ground_truth['lab_id'].unique()
    
    for lab_id in labs:
        gt_lab = ground_truth[ground_truth['lab_id'] == lab_id]
        pred_lab = predictions[predictions['video_id'].isin(gt_lab['video_id'].unique())]
        
        if len(gt_lab) == 0 and len(pred_lab) == 0:
            lab_f_scores[lab_id] = 1.0
        elif len(gt_lab) == 0 or len(pred_lab) == 0:
            lab_f_scores[lab_id] = 0.0
        else:
            lab_f_scores[lab_id] = mouse_fbeta(gt_lab, pred_lab, beta=beta)
    
    results = {
        'overall_f_score': overall_f_score,
        'behavior_f_scores': behavior_f_scores,
        'video_f_scores': video_f_scores,
        'lab_f_scores': lab_f_scores,
        'summary': {
            'total_gt_interactions': len(ground_truth),
            'total_pred_interactions': len(predictions),
            'overall_f_score': overall_f_score,
            'mean_behavior_f_score': np.mean(list(behavior_f_scores.values())),
            'mean_video_f_score': np.mean(list(video_f_scores.values())),
            'mean_lab_f_score': np.mean(list(lab_f_scores.values()))
        }
    }
    
    logger.info(f"Kaggle metric evaluation completed: Overall F-score = {overall_f_score:.4f}")
    return results


def validate_kaggle_compatibility(ground_truth: pd.DataFrame, predictions: pd.DataFrame) -> bool:
    """
    Validate that data is compatible with Kaggle metric requirements.
    
    Args:
        ground_truth: Ground truth DataFrame
        predictions: Predictions DataFrame
        
    Returns:
        True if compatible, False otherwise
    """
    # Check ground truth has required columns
    required_gt_cols = ['video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame', 'lab_id', 'behaviors_labeled']
    missing_gt_cols = [col for col in required_gt_cols if col not in ground_truth.columns]
    if missing_gt_cols:
        logger.error(f"Ground truth missing columns: {missing_gt_cols}")
        return False
    
    # Check predictions have required columns
    required_pred_cols = ['video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame']
    missing_pred_cols = [col for col in required_pred_cols if col not in predictions.columns]
    if missing_pred_cols:
        logger.error(f"Predictions missing columns: {missing_pred_cols}")
        return False
    
    # Check frame ranges are valid
    if not (ground_truth['start_frame'] <= ground_truth['stop_frame']).all():
        logger.error("Ground truth has invalid frame ranges")
        return False
    
    if not (predictions['start_frame'] <= predictions['stop_frame']).all():
        logger.error("Predictions have invalid frame ranges")
        return False
    
    # Check behaviors_labeled is valid JSON
    try:
        for behaviors_str in ground_truth['behaviors_labeled'].unique():
            json.loads(behaviors_str)
    except (json.JSONDecodeError, TypeError) as e:
        logger.error(f"Invalid behaviors_labeled JSON: {e}")
        return False
    
    logger.info("Data is compatible with Kaggle metric")
    return True
