#!/usr/bin/env python3
"""
Debug script to isolate the training error
"""

import sys
import traceback
sys.path.insert(0, 'src')

from mabe.utils.config import load_config
from mabe.train import build_model, train
from mabe.preprocessing import create_dataset

def main():
    try:
        print("Loading configuration...")
        cfg = load_config('configs/default.yaml')
        print("Configuration loaded")
        
        print("Building model...")
        model = build_model(cfg, 26, 'cnn')
        print("Model built")
        
        print("Creating dataset...")
        # Create a minimal dataset for testing
        import pandas as pd
        import numpy as np
        
        # Create dummy data with correct structure
        n_samples = 100
        dummy_data = pd.DataFrame({
            'video_id': ['test_video'] * n_samples,
            'agent_id': ['agent1'] * n_samples,
            'target_id': ['target1'] * n_samples,
            'action': np.random.randint(0, 8, n_samples),
            'start_frame': np.random.randint(0, 1000, n_samples),
            'stop_frame': np.random.randint(1000, 2000, n_samples),
            'behavior': np.random.choice(['approach', 'attack', 'avoid', 'chase', 'chaseattack', 'submit', 'rear', 'shepherd'], n_samples),
            'frame': np.random.randint(0, 1000, n_samples)
        })
        
        # Add dummy features
        for i in range(26):
            dummy_data[f'feature_{i}'] = np.random.randn(n_samples)
        
        print("Creating data loaders...")
        train_loader, val_loader, train_dataset, val_dataset = create_dataset(dummy_data, cfg)
        print("Data loaders created")
        
        print("Starting training...")
        result = train(model, train_loader, val_loader, cfg, resume_checkpoint=None)
        print("Training completed")
        print(f"Result: {result}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
