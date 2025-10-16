#!/usr/bin/env python3
"""
Quick analysis of hyperparameter tuning results
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

def analyze_results():
    # Load the trials data
    trials_path = "outputs/models/hyperparam_trials_cnn_20251016_202820.json"
    
    with open(trials_path, 'r') as f:
        trials = json.load(f)
    
    print(f"Loaded {len(trials)} trials")
    
    # Extract data
    data = []
    for trial in trials:
        if trial.get('state') == 'COMPLETE' and 'user_attrs' in trial:
            user_attrs = trial['user_attrs']
            data.append({
                'trial_number': trial['number'],
                'val_acc': user_attrs.get('val_acc', 0.0),
                'diversity_score': user_attrs.get('diversity_score', 0.0),
                'combined_score': user_attrs.get('combined_score', 0.0),
                'focal_gamma': trial['params'].get('focal_gamma'),
                'focal_alpha': trial['params'].get('focal_alpha'),
                'augment_factor': trial['params'].get('augment_factor'),
                'class_weight_power': trial['params'].get('class_weight_power'),
                'learning_rate': trial['params'].get('learning_rate'),
                'batch_size': trial['params'].get('batch_size'),
                'dropout': trial['params'].get('dropout')
            })
    
    df = pd.DataFrame(data)
    
    print("\n=== ANALYSIS RESULTS ===")
    print(f"Total trials: {len(df)}")
    print(f"Mean validation accuracy: {df['val_acc'].mean():.4f} ± {df['val_acc'].std():.4f}")
    print(f"Mean diversity score: {df['diversity_score'].mean():.4f} ± {df['diversity_score'].std():.4f}")
    print(f"Max diversity score: {df['diversity_score'].max():.4f}")
    print(f"Min diversity score: {df['diversity_score'].min():.4f}")
    print(f"Correlation (val_acc vs diversity): {df['val_acc'].corr(df['diversity_score']):.4f}")
    
    # Best trials
    print("\n=== TOP 5 TRIALS ===")
    top_trials = df.nlargest(5, 'combined_score')
    for idx, row in top_trials.iterrows():
        print(f"Trial {row['trial_number']}: val_acc={row['val_acc']:.4f}, diversity={row['diversity_score']:.4f}, combined={row['combined_score']:.4f}")
        print(f"  Params: focal_gamma={row['focal_gamma']}, focal_alpha={row['focal_alpha']}, augment_factor={row['augment_factor']}")
        print(f"           class_weight_power={row['class_weight_power']}, lr={row['learning_rate']}, batch_size={row['batch_size']}, dropout={row['dropout']}")
        print()
    
    # Parameter analysis
    print("=== PARAMETER ANALYSIS ===")
    for param in ['focal_gamma', 'focal_alpha', 'augment_factor', 'class_weight_power', 'learning_rate', 'batch_size', 'dropout']:
        if param in df.columns:
            print(f"{param}:")
            value_counts = df[param].value_counts().sort_index()
            for value, count in value_counts.items():
                print(f"  {value}: {count} trials")
            print()

if __name__ == "__main__":
    analyze_results()
