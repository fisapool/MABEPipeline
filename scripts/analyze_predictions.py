#!/usr/bin/env python3
"""
Diagnostic script to analyze why predictions don't match ground truth.

This script loads ground truth annotations and predictions, then analyzes
the temporal alignment to understand why F-score is 0.0.
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mabe.utils.logger import get_logger
from mabe.ground_truth_loader import load_ground_truth_from_parquet
from mabe.kaggle_metric import mouse_fbeta

logger = get_logger(__name__)


def analyze_predictions(predictions_csv: str, video_id: str, dataset_path: str):
    """
    Analyze predictions vs ground truth for a specific video.
    
    Args:
        predictions_csv: Path to predictions CSV file
        video_id: Video ID to analyze
        dataset_path: Path to MABE dataset
    """
    logger.info(f"Analyzing predictions for video {video_id}")
    
    # Load predictions
    predictions_df = pd.read_csv(predictions_csv)
    video_predictions = predictions_df[predictions_df['video_id'] == int(video_id)]
    logger.info(f"Loaded {len(video_predictions)} predictions for video {video_id}")
    
    # Load ground truth for this video
    ground_truth = load_ground_truth_from_parquet(dataset_path, [video_id])
    video_gt = ground_truth[ground_truth['video_id'] == video_id]
    logger.info(f"Loaded {len(video_gt)} ground truth annotations for video {video_id}")
    
    if len(video_predictions) == 0:
        logger.error(f"No predictions found for video {video_id}")
        return
    
    if len(video_gt) == 0:
        logger.error(f"No ground truth found for video {video_id}")
        return
    
    # Analyze temporal alignment
    print(f"\n{'='*80}")
    print(f"DIAGNOSTIC ANALYSIS FOR VIDEO {video_id}")
    print(f"{'='*80}")
    
    # 1. Ground Truth Analysis
    print(f"\n1. GROUND TRUTH ANALYSIS")
    print(f"   Total annotations: {len(video_gt)}")
    print(f"   Behaviors: {sorted(video_gt['action'].unique())}")
    print(f"   Agent-Target pairs: {sorted(video_gt[['agent_id', 'target_id']].drop_duplicates().values.tolist())}")
    
    # Show sample ground truth intervals
    print(f"\n   Sample ground truth intervals:")
    for _, row in video_gt.head(5).iterrows():
        duration = row['stop_frame'] - row['start_frame']
        print(f"   - {row['agent_id']}->{row['target_id']} {row['action']}: frames {row['start_frame']}-{row['stop_frame']} ({duration} frames)")
    
    # 2. Predictions Analysis
    print(f"\n2. PREDICTIONS ANALYSIS")
    print(f"   Total predictions: {len(video_predictions)}")
    print(f"   Behaviors: {sorted(video_predictions['action'].unique())}")
    print(f"   Agent-Target pairs: {sorted(video_predictions[['agent_id', 'target_id']].drop_duplicates().values.tolist())}")
    
    # Show sample predictions
    print(f"\n   Sample predictions:")
    for _, row in video_predictions.head(10).iterrows():
        duration = row['stop_frame'] - row['start_frame']
        print(f"   - {row['agent_id']}->{row['target_id']} {row['action']}: frames {row['start_frame']}-{row['stop_frame']} ({duration} frames)")
    
    # 3. Temporal Overlap Analysis
    print(f"\n3. TEMPORAL OVERLAP ANALYSIS")
    
    # Calculate overlaps for each behavior
    behaviors = set(video_gt['action'].unique()) | set(video_predictions['action'].unique())
    
    for behavior in sorted(behaviors):
        gt_behavior = video_gt[video_gt['action'] == behavior]
        pred_behavior = video_predictions[video_predictions['action'] == behavior]
        
        if len(gt_behavior) == 0:
            print(f"   {behavior}: No ground truth, {len(pred_behavior)} predictions (all false positives)")
            continue
        if len(pred_behavior) == 0:
            print(f"   {behavior}: {len(gt_behavior)} ground truth, no predictions (all false negatives)")
            continue
        
        # Calculate frame-level overlap
        gt_frames = set()
        for _, row in gt_behavior.iterrows():
            gt_frames.update(range(row['start_frame'], row['stop_frame']))
        
        pred_frames = set()
        for _, row in pred_behavior.iterrows():
            pred_frames.update(range(row['start_frame'], row['stop_frame']))
        
        overlap = gt_frames.intersection(pred_frames)
        tp = len(overlap)
        fp = len(pred_frames - gt_frames)
        fn = len(gt_frames - pred_frames)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"   {behavior}:")
        print(f"     Ground truth frames: {len(gt_frames)}")
        print(f"     Prediction frames: {len(pred_frames)}")
        print(f"     Overlapping frames: {len(overlap)}")
        print(f"     True Positives: {tp}, False Positives: {fp}, False Negatives: {fn}")
        print(f"     Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
        
        # Show specific overlaps
        if len(overlap) > 0:
            overlap_list = sorted(list(overlap))
            if len(overlap_list) <= 10:
                print(f"     Overlapping frames: {overlap_list}")
            else:
                print(f"     Overlapping frames: {overlap_list[:5]} ... {overlap_list[-5:]} ({len(overlap_list)} total)")
    
    # 4. Interval Duration Analysis
    print(f"\n4. INTERVAL DURATION ANALYSIS")
    
    # Ground truth durations
    gt_durations = []
    for _, row in video_gt.iterrows():
        duration = row['stop_frame'] - row['start_frame']
        gt_durations.append(duration)
    
    # Prediction durations
    pred_durations = []
    for _, row in video_predictions.iterrows():
        duration = row['stop_frame'] - row['start_frame']
        pred_durations.append(duration)
    
    print(f"   Ground truth interval durations:")
    print(f"     Mean: {np.mean(gt_durations):.1f}, Median: {np.median(gt_durations):.1f}")
    print(f"     Min: {np.min(gt_durations)}, Max: {np.max(gt_durations)}")
    print(f"     Distribution: {np.histogram(gt_durations, bins=5)[0]}")
    
    print(f"   Prediction interval durations:")
    print(f"     Mean: {np.mean(pred_durations):.1f}, Median: {np.median(pred_durations):.1f}")
    print(f"     Min: {np.min(pred_durations)}, Max: {np.max(pred_durations)}")
    print(f"     Distribution: {np.histogram(pred_durations, bins=5)[0]}")
    
    # 5. Duplicate Analysis
    print(f"\n5. DUPLICATE ANALYSIS")
    
    # Check for duplicate predictions
    pred_key_counts = video_predictions.groupby(['agent_id', 'target_id', 'action', 'start_frame', 'stop_frame']).size()
    duplicates = pred_key_counts[pred_key_counts > 1]
    
    if len(duplicates) > 0:
        print(f"   Found {len(duplicates)} duplicate prediction intervals:")
        for (agent, target, action, start, stop), count in duplicates.head(5).items():
            print(f"     {agent}->{target} {action} frames {start}-{stop}: {count} duplicates")
        if len(duplicates) > 5:
            print(f"     ... and {len(duplicates) - 5} more")
    else:
        print(f"   No duplicate prediction intervals found")
    
    # 6. Kaggle Metric Calculation
    print(f"\n6. KAGGLE METRIC CALCULATION")
    
    try:
        # Calculate F-beta score using Kaggle metric
        f_score = mouse_fbeta(video_gt, video_predictions, beta=1.0)
        print(f"   Kaggle F1 score: {f_score:.6f}")
        
        if f_score == 0.0:
            print(f"   ❌ F-score is 0.0 - no frame-level overlap detected")
        else:
            print(f"   ✅ F-score > 0 - some frame-level overlap detected")
            
    except Exception as e:
        print(f"   Error calculating Kaggle metric: {e}")
    
    # 7. Recommendations
    print(f"\n7. RECOMMENDATIONS")
    
    if len(duplicates) > 0:
        print(f"   - Fix duplicate predictions in inference generation")
    
    if np.mean(pred_durations) < 5:
        print(f"   - Prediction intervals too short (mean: {np.mean(pred_durations):.1f} frames)")
        print(f"   - Improve interval merging in post-processing")
    
    if len(overlap) == 0:
        print(f"   - No frame-level overlap detected")
        print(f"   - Check if model is predicting correct behaviors")
        print(f"   - Verify frame alignment between predictions and ground truth")
    
    # Save detailed report
    report_path = Path("outputs/diagnostic_report.txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write(f"DIAGNOSTIC REPORT FOR VIDEO {video_id}\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n\n")
        f.write(f"Ground Truth: {len(video_gt)} annotations\n")
        f.write(f"Predictions: {len(video_predictions)} predictions\n")
        f.write(f"Overlapping frames: {len(overlap) if 'overlap' in locals() else 0}\n")
        f.write(f"F1 Score: {f_score if 'f_score' in locals() else 'Error'}\n")
    
    logger.info(f"Diagnostic report saved to {report_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Analyze predictions vs ground truth")
    parser.add_argument("--predictions", required=True, help="Path to predictions CSV file")
    parser.add_argument("--video", required=True, help="Video ID to analyze")
    parser.add_argument("--dataset-path", default="C:/Users/MYPC/Documents/MABEDatasets/MABe-extracted", 
                       help="Path to MABE dataset")
    
    args = parser.parse_args()
    
    try:
        analyze_predictions(args.predictions, args.video, args.dataset_path)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
