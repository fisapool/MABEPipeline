#!/usr/bin/env python3
"""
Analyze hyperparameter tuning results and behavior diversity

This script analyzes Optuna tuning results to understand the trade-offs
between F1 score and behavior diversity, and recommends optimal hyperparameters.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_optuna_results(results_path: str) -> Dict:
    """
    Load Optuna results JSON
    
    Args:
        results_path: Path to Optuna results JSON file
        
    Returns:
        Dictionary with results data
    """
    try:
        with open(results_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading results from {results_path}: {e}")
        return {}


def analyze_diversity_tradeoffs(results: Dict) -> Dict:
    """
    Analyze F1 vs diversity trade-offs from trials
    
    Args:
        results: Optuna results dictionary
        
    Returns:
        Analysis results dictionary
    """
    if 'trials' not in results:
        logger.warning("No trials data found in results")
        return {}
    
    trials = results['trials']
    
    # Extract trial data with user attributes
    trial_data = []
    for trial in trials:
        if trial.get('state') == 'COMPLETE' and 'user_attrs' in trial:
            user_attrs = trial['user_attrs']
            trial_data.append({
                'trial_number': trial['number'],
                'value': trial['value'],
                'val_acc': user_attrs.get('val_acc', 0.0),
                'diversity_score': user_attrs.get('diversity_score', 0.0),
                'combined_score': user_attrs.get('combined_score', 0.0),
                'params': trial['params']
            })
    
    if not trial_data:
        logger.warning("No complete trials with user attributes found")
        return {}
    
    df = pd.DataFrame(trial_data)
    
    # Calculate statistics
    analysis = {
        'total_trials': len(df),
        'mean_val_acc': df['val_acc'].mean(),
        'std_val_acc': df['val_acc'].std(),
        'mean_diversity': df['diversity_score'].mean(),
        'std_diversity': df['diversity_score'].std(),
        'max_diversity': df['diversity_score'].max(),
        'min_diversity': df['diversity_score'].min(),
        'correlation': df['val_acc'].corr(df['diversity_score'])
    }
    
    # Find Pareto front (trials with best combined scores)
    df_sorted = df.sort_values('combined_score', ascending=False)
    pareto_trials = df_sorted.head(min(5, len(df_sorted)))
    
    analysis['pareto_trials'] = pareto_trials.to_dict('records')
    
    logger.info(f"Analyzed {analysis['total_trials']} trials")
    logger.info(f"Mean validation accuracy: {analysis['mean_val_acc']:.4f} ± {analysis['std_val_acc']:.4f}")
    logger.info(f"Mean diversity score: {analysis['mean_diversity']:.4f} ± {analysis['std_diversity']:.4f}")
    logger.info(f"Correlation between accuracy and diversity: {analysis['correlation']:.4f}")
    
    return analysis


def recommend_hyperparameters(results: Dict, analysis: Dict) -> Dict:
    """
    Recommend hyperparameters based on multi-objective analysis
    
    Args:
        results: Optuna results dictionary
        analysis: Analysis results dictionary
        
    Returns:
        Dictionary with recommendations
    """
    if 'pareto_trials' not in analysis or not analysis['pareto_trials']:
        logger.warning("No Pareto trials found for recommendations")
        return {}
    
    # Get best trial from Pareto front
    best_trial = analysis['pareto_trials'][0]
    
    # Analyze parameter patterns from top trials
    top_trials = analysis['pareto_trials'][:3]  # Top 3 trials
    
    # Count parameter frequency in top trials
    param_frequency = {}
    for trial in top_trials:
        for param, value in trial['params'].items():
            if param not in param_frequency:
                param_frequency[param] = {}
            if value not in param_frequency[param]:
                param_frequency[param][value] = 0
            param_frequency[param][value] += 1
    
    # Get most common values
    recommendations = {}
    for param, values in param_frequency.items():
        most_common = max(values, key=values.get)
        recommendations[param] = most_common
    
    # Add best trial info
    recommendations['best_trial'] = {
        'trial_number': best_trial['trial_number'],
        'val_acc': best_trial['val_acc'],
        'diversity_score': best_trial['diversity_score'],
        'combined_score': best_trial['combined_score']
    }
    
    logger.info("=== Hyperparameter Recommendations ===")
    for param, value in recommendations.items():
        if param != 'best_trial':
            logger.info(f"{param}: {value}")
    
    logger.info(f"Best trial: #{best_trial['trial_number']} "
                f"(val_acc={best_trial['val_acc']:.4f}, "
                f"diversity={best_trial['diversity_score']:.4f})")
    
    return recommendations


def create_visualization(results: Dict, analysis: Dict, output_dir: str) -> None:
    """
    Create visualization plots for tuning results
    
    Args:
        results: Optuna results dictionary
        analysis: Analysis results dictionary
        output_dir: Output directory for plots
    """
    if 'pareto_trials' not in analysis or not analysis['pareto_trials']:
        logger.warning("No data available for visualization")
        return
    
    df = pd.DataFrame(analysis['pareto_trials'])
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Accuracy vs Diversity scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df['val_acc'], df['diversity_score'], alpha=0.7, s=50)
    plt.xlabel('Validation Accuracy')
    plt.ylabel('Behavior Diversity Score')
    plt.title('F1 Score vs Behavior Diversity Trade-off')
    plt.grid(True, alpha=0.3)
    
    # Highlight best trial
    best_trial = df.iloc[0]
    plt.scatter(best_trial['val_acc'], best_trial['diversity_score'], 
                color='red', s=100, marker='*', label='Best Trial')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path / 'accuracy_vs_diversity.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Parameter importance (if enough trials)
    if len(df) >= 5:
        # Analyze parameter ranges in top trials
        param_ranges = {}
        for trial in analysis['pareto_trials']:
            for param, value in trial['params'].items():
                if param not in param_ranges:
                    param_ranges[param] = []
                param_ranges[param].append(value)
        
        # Create parameter distribution plot
        n_params = len(param_ranges)
        if n_params > 0:
            fig, axes = plt.subplots((n_params + 2) // 3, 3, figsize=(15, 5 * ((n_params + 2) // 3)))
            if n_params == 1:
                axes = [axes]
            elif n_params <= 3:
                axes = axes.reshape(1, -1)
            
            for i, (param, values) in enumerate(param_ranges.items()):
                row, col = i // 3, i % 3
                ax = axes[row, col] if n_params > 3 else axes[col] if n_params <= 3 else axes[i]
                
                if isinstance(values[0], (int, float)):
                    ax.hist(values, bins=10, alpha=0.7)
                    ax.set_title(f'{param} Distribution')
                    ax.set_xlabel(param)
                    ax.set_ylabel('Frequency')
                else:
                    # Categorical parameter
                    unique_values, counts = np.unique(values, return_counts=True)
                    ax.bar(range(len(unique_values)), counts, alpha=0.7)
                    ax.set_title(f'{param} Distribution')
                    ax.set_xlabel(param)
                    ax.set_ylabel('Frequency')
                    ax.set_xticks(range(len(unique_values)))
                    ax.set_xticklabels(unique_values, rotation=45)
            
            # Hide empty subplots
            for i in range(n_params, len(axes.flat)):
                axes.flat[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(output_path / 'parameter_distributions.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    logger.info(f"Visualizations saved to {output_path}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Analyze hyperparameter tuning results')
    parser.add_argument('--results', type=str, required=True,
                       help='Path to Optuna results JSON file')
    parser.add_argument('--output-dir', type=str, default='outputs/analysis',
                       help='Output directory for analysis and plots')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load results
    logger.info(f"Loading results from {args.results}")
    results = load_optuna_results(args.results)
    
    if not results:
        logger.error("Failed to load results")
        return 1
    
    # Analyze diversity trade-offs
    logger.info("Analyzing diversity trade-offs...")
    analysis = analyze_diversity_tradeoffs(results)
    
    if not analysis:
        logger.error("No analysis data available")
        return 1
    
    # Generate recommendations
    logger.info("Generating hyperparameter recommendations...")
    recommendations = recommend_hyperparameters(results, analysis)
    
    # Create visualizations
    logger.info("Creating visualizations...")
    create_visualization(results, analysis, args.output_dir)
    
    # Save analysis results
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    analysis_results = {
        'analysis': analysis,
        'recommendations': recommendations,
        'summary': {
            'total_trials': analysis.get('total_trials', 0),
            'best_val_acc': analysis.get('pareto_trials', [{}])[0].get('val_acc', 0.0),
            'best_diversity': analysis.get('pareto_trials', [{}])[0].get('diversity_score', 0.0),
            'correlation': analysis.get('correlation', 0.0)
        }
    }
    
    with open(output_path / 'analysis_results.json', 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    logger.info(f"Analysis complete! Results saved to {output_path}")
    logger.info("=== Summary ===")
    logger.info(f"Total trials analyzed: {analysis_results['summary']['total_trials']}")
    logger.info(f"Best validation accuracy: {analysis_results['summary']['best_val_acc']:.4f}")
    logger.info(f"Best diversity score: {analysis_results['summary']['best_diversity']:.4f}")
    logger.info(f"Accuracy-Diversity correlation: {analysis_results['summary']['correlation']:.4f}")
    
    return 0


if __name__ == '__main__':
    exit(main())
