#!/usr/bin/env python3
"""
Local evaluation script using ground truth data

This script loads ground truth annotations from the MABE dataset and evaluates
predictions against them using comprehensive metrics.
"""

import sys
import os
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
from mabe.evaluate_local import evaluate_predictions, save_evaluation_results
from mabe.ground_truth_loader import load_ground_truth_for_evaluation
from mabe.utils.config import load_config
from mabe.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """Run local evaluation with ground truth data"""
    parser = argparse.ArgumentParser(description="Local evaluation with ground truth data")
    parser.add_argument("--predictions", required=True, help="Path to predictions CSV file")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--output", help="Output path for evaluation results")
    parser.add_argument("--max-videos", type=int, help="Maximum number of videos to evaluate")
    
    args = parser.parse_args()
    
    # Load configuration
    cfg = load_config(args.config)
    
    # Override max_videos if specified
    if args.max_videos:
        if 'data' not in cfg:
            cfg['data'] = {}
        cfg['data']['max_videos'] = args.max_videos
    
    # Load predictions
    predictions_path = args.predictions
    logger.info(f"Loading predictions from {predictions_path}")
    
    if not Path(predictions_path).exists():
        logger.error(f"Predictions file not found: {predictions_path}")
        return 1
    
    predictions = pd.read_csv(predictions_path)
    logger.info(f"Loaded {len(predictions)} predictions")
    
    # Load ground truth
    logger.info("Loading ground truth data...")
    try:
        ground_truth = load_ground_truth_for_evaluation(cfg)
    except Exception as e:
        logger.error(f"Failed to load ground truth: {e}")
        return 1
    
    if ground_truth.empty:
        logger.error("No ground truth data found!")
        return 1
    
    logger.info(f"Loaded {len(ground_truth)} ground truth annotations")
    
    # Run evaluation
    logger.info("Running evaluation...")
    try:
        metrics = evaluate_predictions(ground_truth, predictions)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1
    
    # Save results
    output_path = args.output
    if output_path:
        results_path = save_evaluation_results(metrics, cfg, output_path)
    else:
        results_path = save_evaluation_results(metrics, cfg)
    
    # Print summary
    print("\n" + "="*60)
    print("LOCAL EVALUATION RESULTS")
    print("="*60)
    print(f"Overall F-score: {metrics['overall_f_score']:.4f}")
    print(f"Mean behavior F-score: {metrics['summary']['mean_behavior_f_score']:.4f}")
    print(f"Mean video F-score: {metrics['summary']['mean_video_f_score']:.4f}")
    print(f"Total ground truth interactions: {metrics['summary']['total_gt_interactions']}")
    print(f"Total predicted interactions: {metrics['summary']['total_pred_interactions']}")
    print(f"Results saved to: {results_path}")
    print("="*60)
    
    # Print per-behavior F-scores
    print("\nPer-behavior F-scores:")
    for behavior, f_score in sorted(metrics['behavior_f_scores'].items()):
        print(f"  {behavior}: {f_score:.4f}")
    
    # Print per-video F-scores (top 10)
    print(f"\nPer-video F-scores (showing top 10):")
    video_scores = sorted(metrics['video_f_scores'].items(), key=lambda x: x[1], reverse=True)
    for video_id, f_score in video_scores[:10]:
        print(f"  {video_id}: {f_score:.4f}")
    
    if len(video_scores) > 10:
        print(f"  ... and {len(video_scores) - 10} more videos")
    
    return 0


if __name__ == "__main__":
    exit(main())
