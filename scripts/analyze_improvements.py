"""
Analysis Script for MABE Pipeline Improvements

Compares before/after metrics for class imbalance and model calibration improvements.
Analyzes class distribution, confidence scores, behavior diversity, and model performance.
"""

import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_training_metrics(metrics_path: str) -> Dict:
    """Load training metrics from JSON file"""
    try:
        with open(metrics_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading metrics from {metrics_path}: {e}")
        return {}


def analyze_class_distribution(frame_labels_df: pd.DataFrame) -> Dict:
    """Analyze class distribution in the dataset"""
    behavior_counts = frame_labels_df['behavior'].value_counts()
    
    # Get all 8 behavior classes
    all_behaviors = ['approach', 'attack', 'avoid', 'chase', 'chaseattack', 'submit', 'rear', 'shepherd']
    
    distribution = {}
    for behavior in all_behaviors:
        count = behavior_counts.get(behavior, 0)
        distribution[behavior] = {
            'count': count,
            'percentage': (count / len(frame_labels_df)) * 100 if len(frame_labels_df) > 0 else 0
        }
    
    return distribution


def analyze_confidence_scores(submission_path: str) -> Dict:
    """Analyze confidence scores from submission file"""
    try:
        submission_df = pd.read_csv(submission_path)
        
        # Extract confidence scores if available
        # Note: This is a simplified analysis - actual confidence scores would need to be stored separately
        
        analysis = {
            'total_predictions': len(submission_df),
            'unique_behaviors': submission_df['action'].nunique() if 'action' in submission_df.columns else 0,
            'behavior_distribution': submission_df['action'].value_counts().to_dict() if 'action' in submission_df.columns else {}
        }
        
        return analysis
    except Exception as e:
        logger.error(f"Error analyzing submission file {submission_path}: {e}")
        return {}


def calculate_behavior_diversity(predictions: List[str]) -> float:
    """Calculate behavior diversity score"""
    if not predictions:
        return 0.0
    
    unique_behaviors = len(set(predictions))
    total_classes = 8  # Total number of behavior classes
    return unique_behaviors / total_classes


def create_comparison_plots(before_metrics: Dict, after_metrics: Dict, output_dir: Path):
    """Create comparison plots for before/after analysis"""
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('MABE Pipeline Improvements Analysis', fontsize=16, fontweight='bold')
    
    # 1. Class Distribution Comparison
    ax1 = axes[0, 0]
    behaviors = ['approach', 'attack', 'avoid', 'chase', 'chaseattack', 'submit', 'rear', 'shepherd']
    
    before_counts = [before_metrics.get('class_distribution', {}).get(behavior, {}).get('count', 0) for behavior in behaviors]
    after_counts = [after_metrics.get('class_distribution', {}).get(behavior, {}).get('count', 0) for behavior in behaviors]
    
    x = np.arange(len(behaviors))
    width = 0.35
    
    ax1.bar(x - width/2, before_counts, width, label='Before', alpha=0.8, color='lightcoral')
    ax1.bar(x + width/2, after_counts, width, label='After', alpha=0.8, color='lightblue')
    
    ax1.set_xlabel('Behavior Classes')
    ax1.set_ylabel('Sample Count')
    ax1.set_title('Class Distribution Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(behaviors, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Model Performance Comparison
    ax2 = axes[0, 1]
    metrics = ['best_val_acc', 'final_train_loss', 'final_val_loss']
    before_values = [before_metrics.get(metric, 0) for metric in metrics]
    after_values = [after_metrics.get(metric, 0) for metric in metrics]
    
    x = np.arange(len(metrics))
    ax2.bar(x - width/2, before_values, width, label='Before', alpha=0.8, color='lightcoral')
    ax2.bar(x + width/2, after_values, width, label='After', alpha=0.8, color='lightblue')
    
    ax2.set_xlabel('Metrics')
    ax2.set_ylabel('Values')
    ax2.set_title('Model Performance Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Val Accuracy', 'Train Loss', 'Val Loss'])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Behavior Diversity
    ax3 = axes[1, 0]
    before_diversity = before_metrics.get('behavior_diversity', 0)
    after_diversity = after_metrics.get('behavior_diversity', 0)
    
    diversities = [before_diversity, after_diversity]
    labels = ['Before', 'After']
    colors = ['lightcoral', 'lightblue']
    
    bars = ax3.bar(labels, diversities, color=colors, alpha=0.8)
    ax3.set_ylabel('Behavior Diversity Score')
    ax3.set_title('Behavior Diversity Comparison')
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, diversities):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Training Progress
    ax4 = axes[1, 1]
    if 'val_accuracies' in before_metrics and 'val_accuracies' in after_metrics:
        before_epochs = range(len(before_metrics['val_accuracies']))
        after_epochs = range(len(after_metrics['val_accuracies']))
        
        ax4.plot(before_epochs, before_metrics['val_accuracies'], 
                label='Before', marker='o', alpha=0.8, color='lightcoral')
        ax4.plot(after_epochs, after_metrics['val_accuracies'], 
                label='After', marker='s', alpha=0.8, color='lightblue')
        
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Validation Accuracy (%)')
        ax4.set_title('Training Progress Comparison')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'improvements_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_improvement_report(before_metrics: Dict, after_metrics: Dict, output_dir: Path):
    """Generate a comprehensive improvement report"""
    
    report = []
    report.append("# MABE Pipeline Improvements Analysis Report")
    report.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Class Distribution Analysis
    report.append("## Class Distribution Analysis")
    report.append("")
    
    behaviors = ['approach', 'attack', 'avoid', 'chase', 'chaseattack', 'submit', 'rear', 'shepherd']
    report.append("| Behavior | Before Count | After Count | Improvement |")
    report.append("|----------|--------------|-------------|-------------|")
    
    for behavior in behaviors:
        before_count = before_metrics.get('class_distribution', {}).get(behavior, {}).get('count', 0)
        after_count = after_metrics.get('class_distribution', {}).get(behavior, {}).get('count', 0)
        
        if before_count > 0:
            improvement = ((after_count - before_count) / before_count) * 100
        else:
            improvement = float('inf') if after_count > 0 else 0
        
        report.append(f"| {behavior} | {before_count} | {after_count} | {improvement:.1f}% |")
    
    report.append("")
    
    # Model Performance Analysis
    report.append("## Model Performance Analysis")
    report.append("")
    
    before_acc = before_metrics.get('best_val_acc', 0)
    after_acc = after_metrics.get('best_val_acc', 0)
    acc_improvement = ((after_acc - before_acc) / before_acc) * 100 if before_acc > 0 else 0
    
    report.append(f"- **Validation Accuracy**: {before_acc:.2f}% → {after_acc:.2f}% ({acc_improvement:+.1f}%)")
    
    # Behavior Diversity Analysis
    before_diversity = before_metrics.get('behavior_diversity', 0)
    after_diversity = after_metrics.get('behavior_diversity', 0)
    diversity_improvement = ((after_diversity - before_diversity) / before_diversity) * 100 if before_diversity > 0 else 0
    
    report.append(f"- **Behavior Diversity**: {before_diversity:.3f} → {after_diversity:.3f} ({diversity_improvement:+.1f}%)")
    
    # Training Stability Analysis
    before_epochs = len(before_metrics.get('val_accuracies', []))
    after_epochs = len(after_metrics.get('val_accuracies', []))
    
    report.append(f"- **Training Epochs**: {before_epochs} → {after_epochs}")
    
    report.append("")
    
    # Key Improvements Summary
    report.append("## Key Improvements Summary")
    report.append("")
    
    improvements = []
    
    if acc_improvement > 0:
        improvements.append(f"✅ Validation accuracy improved by {acc_improvement:.1f}%")
    
    if diversity_improvement > 0:
        improvements.append(f"✅ Behavior diversity improved by {diversity_improvement:.1f}%")
    
    # Check for missing classes being addressed
    before_missing = sum(1 for behavior in behaviors 
                        if before_metrics.get('class_distribution', {}).get(behavior, {}).get('count', 0) == 0)
    after_missing = sum(1 for behavior in behaviors 
                       if after_metrics.get('class_distribution', {}).get(behavior, {}).get('count', 0) == 0)
    
    if after_missing < before_missing:
        improvements.append(f"✅ Reduced missing classes from {before_missing} to {after_missing}")
    
    if not improvements:
        improvements.append("⚠️ No significant improvements detected")
    
    for improvement in improvements:
        report.append(improvement)
    
    report.append("")
    
    # Recommendations
    report.append("## Recommendations")
    report.append("")
    
    if after_acc < 40:
        report.append("- Consider increasing training epochs or adjusting learning rate")
    
    if after_diversity < 0.5:
        report.append("- Implement more aggressive data augmentation for minority classes")
    
    if after_missing > 0:
        report.append("- Consider synthetic data generation for missing classes")
    
    # Save report
    with open(output_dir / 'improvements_report.md', 'w') as f:
        f.write('\n'.join(report))
    
    logger.info(f"Improvement report saved to {output_dir / 'improvements_report.md'}")


def main():
    parser = argparse.ArgumentParser(description="Analyze MABE pipeline improvements")
    parser.add_argument('--before-metrics', type=str, required=True,
                        help='Path to before training metrics JSON file')
    parser.add_argument('--after-metrics', type=str, required=True,
                        help='Path to after training metrics JSON file')
    parser.add_argument('--before-submission', type=str, required=True,
                        help='Path to before submission CSV file')
    parser.add_argument('--after-submission', type=str, required=True,
                        help='Path to after submission CSV file')
    parser.add_argument('--output-dir', type=str, default='outputs/analysis',
                        help='Directory to save analysis results')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting MABE pipeline improvements analysis...")
    
    # Load metrics
    before_metrics = load_training_metrics(args.before_metrics)
    after_metrics = load_training_metrics(args.after_metrics)
    
    if not before_metrics or not after_metrics:
        logger.error("Failed to load training metrics")
        return
    
    # Analyze submissions
    before_submission = analyze_confidence_scores(args.before_submission)
    after_submission = analyze_confidence_scores(args.after_submission)
    
    # Add submission analysis to metrics
    before_metrics.update(before_submission)
    after_metrics.update(after_submission)
    
    # Create comparison plots
    logger.info("Creating comparison plots...")
    create_comparison_plots(before_metrics, after_metrics, output_dir)
    
    # Generate improvement report
    logger.info("Generating improvement report...")
    generate_improvement_report(before_metrics, after_metrics, output_dir)
    
    logger.info(f"Analysis complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
