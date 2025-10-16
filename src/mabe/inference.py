"""
MABE Model Inference Module

Refactored from original inference.py to work with configuration system.
Handles model loading, prediction generation, and submission creation.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import json
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# GPU/CUDA Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Import utilities
from .utils.logger import get_logger
from .utils.seed import set_seed

logger = get_logger(__name__)


class TestMouseBehaviorDataset(Dataset):
    """Custom dataset for test data inference - F-SCORE COMPLIANT VERSION"""
    
    def __init__(self, test_df, tracking_data_dict, annotation_maps=None, training_patterns=None):
        self.test_df = test_df
        self.tracking_data = tracking_data_dict
        self.annotation_maps = annotation_maps or {}
        self.training_patterns = training_patterns or {}
        self.behavior_map = {
            'approach': 0, 'attack': 1, 'avoid': 2, 'chase': 3,
            'chaseattack': 4, 'submit': 5, 'rear': 6, 'shepherd': 7
        }
        self.reverse_behavior_map = {v: k for k, v in self.behavior_map.items()}
        
        # Generate F-Score compliant predictions (only annotated interactions)
        self.predictions_data = self.generate_annotation_aware_predictions()
        
    def generate_annotation_aware_predictions(self):
        """Generate predictions for test data - create predictions for all possible mouse interactions"""
        predictions_data = []
        
        for _, video_row in self.test_df.iterrows():
            video_id = video_row['video_id']
            fps = video_row['frames_per_second']
            duration_sec = video_row['video_duration_sec']
            total_frames = int(fps * duration_sec)
            
            # For test data, we need to generate predictions for all possible mouse interactions
            # since we don't have annotations. We'll use a standard approach.
            logger.info(f"Generating predictions for test video {video_id} ({total_frames} frames)")
            
            # Get available mouse IDs from tracking data
            available_mice = self.get_available_mice(video_id)
            
            if not available_mice:
                logger.warning(f"No mouse tracking data found for {video_id}")
                # Create default mouse pair
                available_mice = [('mouse1', 'mouse2')]
            
            # Generate predictions for all mouse pairs
            for agent_id, target_id in available_mice:
                if agent_id == target_id:
                    continue  # Skip self-interactions
                    
                logger.info(f"Generating predictions for {video_id}: {agent_id}->{target_id}")
                
                # Use sliding window approach for all interactions
                window_size = 30  # frames
                step_size = 10    # frames (less overlap for faster processing)
                
                for start_frame in range(0, total_frames - window_size + 1, step_size):
                    end_frame = min(start_frame + window_size, total_frames)
                    center_frame = (start_frame + end_frame) // 2
                    
                    predictions_data.append({
                        'video_id': video_id,
                        'frame': center_frame,
                        'start_frame': start_frame,
                        'end_frame': end_frame,
                        'agent_id': agent_id,
                        'target_id': target_id,
                        'annotated_behaviors': []  # No annotations for test data
                    })
        
        logger.info(f"Generated {len(predictions_data)} test predictions")
        
        # If no predictions were generated, create a minimal prediction to avoid empty dataset
        if not predictions_data:
            logger.warning("No predictions generated - creating minimal prediction to avoid empty dataset")
            # Create a single dummy prediction to prevent empty dataset issues
            if not self.test_df.empty:
                video_id = self.test_df.iloc[0]['video_id']
                predictions_data.append({
                    'video_id': video_id,
                    'frame': 0,
                    'start_frame': 0,
                    'end_frame': 1,
                    'agent_id': 'mouse1',
                    'target_id': 'mouse2',
                    'annotated_behaviors': []  # No annotations for test data
                })
        
        return predictions_data
    
    def get_available_mice(self, video_id):
        """Get available mouse IDs from tracking data"""
        if video_id not in self.tracking_data:
            return []
        
        tracking_df = self.tracking_data[video_id]
        
        # Find all mouse columns (ending with _body_center_x)
        mouse_columns = [col for col in tracking_df.columns if col.endswith('_body_center_x')]
        mice = []
        
        for col in mouse_columns:
            mouse_id = col.split('_')[0]  # Extract mouse ID
            mice.append(mouse_id)
        
        # Create all possible pairs
        mouse_pairs = []
        for i, mouse1 in enumerate(mice):
            for mouse2 in mice[i+1:]:
                mouse_pairs.append((mouse1, mouse2))
                mouse_pairs.append((mouse2, mouse1))  # Both directions
        
        return mouse_pairs
    
    def __len__(self):
        return len(self.predictions_data)
    
    def __getitem__(self, idx):
        pred_data = self.predictions_data[idx]
        
        video_id = pred_data['video_id']
        frame = pred_data['frame']
        agent_id = pred_data['agent_id']
        target_id = pred_data['target_id']
        
        # Extract features from tracking data
        features = self.extract_tracking_features(video_id, frame, agent_id, target_id)
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features)
        
        return features_tensor, video_id, frame, agent_id, target_id
    
    def extract_tracking_features(self, video_id, frame, agent_id, target_id):
        """Extract features from tracking data - EXACTLY matches train.py"""
        features = []
        
        if video_id not in self.tracking_data:
            # Return dummy features if tracking data not available
            return [0.0] * 26
        
        tracking_df = self.tracking_data[video_id]
        
        # Get frame data - handle different column names
        if 'frame' in tracking_df.columns:
            frame_data = tracking_df[tracking_df['frame'] == frame]
        elif 'Frame' in tracking_df.columns:
            frame_data = tracking_df[tracking_df['Frame'] == frame]
        else:
            # If no frame column, use index as frame
            if frame < len(tracking_df):
                frame_data = tracking_df.iloc[frame:frame+1]
            else:
                return [0.0] * 26
        
        if frame_data.empty:
            return [0.0] * 26
        
        # Extract spatial features
        spatial_features = self.extract_spatial_features(frame_data, agent_id, target_id)
        features.extend(spatial_features)
        
        # Extract temporal features
        temporal_features = self.extract_temporal_features(tracking_df, frame)
        features.extend(temporal_features)
        
        # Extract interaction features
        interaction_features = self.extract_interaction_features(frame_data, agent_id, target_id)
        features.extend(interaction_features)
        
        return features
    
    def extract_spatial_features(self, frame_data, agent_id, target_id):
        """Extract spatial features from frame data"""
        features = []
        
        # Agent position
        agent_x_col = f'{agent_id}_body_center_x'
        agent_y_col = f'{agent_id}_body_center_y'
        
        if agent_x_col in frame_data.columns and agent_y_col in frame_data.columns:
            agent_x = frame_data[agent_x_col].iloc[0] if not frame_data.empty else 0
            agent_y = frame_data[agent_y_col].iloc[0] if not frame_data.empty else 0
            features.extend([agent_x, agent_y])
        else:
            features.extend([0.0, 0.0])
        
        # Target position
        target_x_col = f'{target_id}_body_center_x'
        target_y_col = f'{target_id}_body_center_y'
        
        if target_x_col in frame_data.columns and target_y_col in frame_data.columns:
            target_x = frame_data[target_x_col].iloc[0] if not frame_data.empty else 0
            target_y = frame_data[target_y_col].iloc[0] if not frame_data.empty else 0
            features.extend([target_x, target_y])
        else:
            features.extend([0.0, 0.0])
        
        # Distance and angle
        if len(features) >= 4:
            distance = np.sqrt((features[0] - features[2])**2 + (features[1] - features[3])**2)
            angle = np.arctan2(features[3] - features[1], features[2] - features[0])
            features.extend([distance, angle, np.sin(angle), np.cos(angle)])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # Add more spatial features (padding to 12 features)
        while len(features) < 12:
            features.append(0.0)
        
        return features[:12]
    
    def extract_temporal_features(self, tracking_df, frame, window_size=5):
        """Extract temporal features using sliding window - FIXED VERSION"""
        features = []
        
        # Get window around current frame
        start_frame = max(0, frame - window_size)
        end_frame = min(len(tracking_df), frame + window_size + 1)
        window_data = tracking_df.iloc[start_frame:end_frame]
        
        if len(window_data) < 2:
            return [0.0] * 10  # Return dummy features
        
        # Find the first available mouse column
        mouse_columns = [col for col in window_data.columns if col.endswith('_body_center_x')]
        
        if mouse_columns:
            mouse_id = mouse_columns[0].split('_')[0]  # Extract mouse ID
            x_col = f'{mouse_id}_body_center_x'
            y_col = f'{mouse_id}_body_center_y'
            
            if x_col in window_data.columns and y_col in window_data.columns:
                dx = window_data[x_col].diff()
                dy = window_data[y_col].diff()
                velocity = np.sqrt(dx**2 + dy**2)
                
                features.extend([
                    velocity.mean(), velocity.std(), velocity.max(),
                    velocity.min(), velocity.median()
                ])
                
                # Acceleration
                accel = velocity.diff()
                features.extend([
                    accel.mean(), accel.std(), accel.max(),
                    accel.min(), accel.median()
                ])
            else:
                features.extend([0.0] * 10)
        else:
            features.extend([0.0] * 10)
        
        return features
    
    def extract_interaction_features(self, frame_data, agent_id, target_id):
        """Extract interaction features between mice"""
        features = []
        
        # Distance between mice
        agent_x_col = f'{agent_id}_body_center_x'
        agent_y_col = f'{agent_id}_body_center_y'
        target_x_col = f'{target_id}_body_center_x'
        target_y_col = f'{target_id}_body_center_y'
        
        if all(col in frame_data for col in [agent_x_col, agent_y_col, target_x_col, target_y_col]):
            agent_x = frame_data[agent_x_col].iloc[0] if not frame_data.empty else 0
            agent_y = frame_data[agent_y_col].iloc[0] if not frame_data.empty else 0
            target_x = frame_data[target_x_col].iloc[0] if not frame_data.empty else 0
            target_y = frame_data[target_y_col].iloc[0] if not frame_data.empty else 0
            
            distance = np.sqrt((agent_x - target_x)**2 + (agent_y - target_y)**2)
            angle = np.arctan2(target_y - agent_y, target_x - agent_x)
            
            features.extend([distance, angle, np.sin(angle), np.cos(angle)])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        return features


# FIXED MODEL ARCHITECTURES - EXACTLY MATCHING train.py
class BehaviorCNN_Old(nn.Module):
    """Old CNN model architecture that matches saved models"""
    
    def __init__(self, input_dim, num_classes, dropout=0.3):
        super(BehaviorCNN_Old, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Fully connected layers - OLD ARCHITECTURE
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class BehaviorLSTM_Old(nn.Module):
    """Old LSTM model architecture that matches saved models"""
    
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=2, dropout=0.3):
        super(BehaviorLSTM_Old, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # LSTM layers - OLD ARCHITECTURE
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Fully connected layers - OLD ARCHITECTURE
        self.fc1 = nn.Linear(hidden_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Ensure x has the right shape for LSTM
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last output
        output = lstm_out[:, -1, :]
        
        # Fully connected layers
        output = self.relu(self.fc1(output))
        output = self.dropout(output)
        output = self.relu(self.fc2(output))
        output = self.dropout(output)
        output = self.fc3(output)
        
        return output


def load_model(model_path: str, model_class: str = 'cnn', input_dim: int = 26) -> nn.Module:
    """
    Load trained model from file
    
    Args:
        model_path: Path to model file
        model_class: Model class ('cnn' or 'lstm')
        input_dim: Input feature dimension
        
    Returns:
        Loaded PyTorch model
    """
    logger.info(f"Loading {model_class} model from {model_path}")
    
    try:
        if model_class.lower() == 'cnn':
            model = BehaviorCNN_Old(input_dim=input_dim, num_classes=8)
        elif model_class.lower() == 'lstm':
            model = BehaviorLSTM_Old(input_dim=input_dim, hidden_dim=128, num_classes=8)
        else:
            raise ValueError(f"Unknown model class: {model_class}")
        
        # Load state dict
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        
        logger.info(f"Successfully loaded {model_class} model")
        return model
        
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        raise


def predict_single(model, data_loader, device, return_proba: bool = True) -> np.ndarray:
    """
    Make predictions with a single model
    
    Args:
        model: PyTorch model
        data_loader: Data loader
        device: Device to run on
        return_proba: Whether to return probabilities or class predictions
        
    Returns:
        Array of predictions
    """
    model.eval()
    all_predictions = []
    
    with torch.no_grad():
        for data, video_ids, frames, agent_ids, target_ids in data_loader:
            data = data.to(device)
            outputs = model(data)
            
            if return_proba:
                probabilities = F.softmax(outputs, dim=1)
                all_predictions.append(probabilities.cpu().numpy())
            else:
                _, predicted = torch.max(outputs, 1)
                all_predictions.append(predicted.cpu().numpy())
    
    # Handle empty predictions case
    if not all_predictions:
        logger.warning("No predictions generated - returning empty array")
        return np.array([])
    
    return np.concatenate(all_predictions, axis=0)


def ensemble_predictions(proba_list: List[np.ndarray], method: str = 'average') -> np.ndarray:
    """
    Ensemble multiple model predictions
    
    Args:
        proba_list: List of probability arrays
        method: Ensemble method ('average' or 'max')
        
    Returns:
        Ensembled predictions
    """
    if not proba_list:
        raise ValueError("No predictions to ensemble")
    
    if method == 'average':
        return np.mean(proba_list, axis=0)
    elif method == 'max':
        return np.max(proba_list, axis=0)
    else:
        raise ValueError(f"Unknown ensemble method: {method}")


def create_kaggle_submission(predictions: List[Dict], cfg: Dict, min_duration: int = 1) -> Tuple[str, pd.DataFrame]:
    """
    Create Kaggle submission file in the correct format
    
    Args:
        predictions: List of prediction dictionaries
        cfg: Configuration dictionary
        min_duration: Minimum duration for predictions
        
    Returns:
        Tuple of (submission_path, submission_df)
    """
    logger.info("Creating Kaggle submission...")
    
    submission_data = []
    
    # Group predictions by video, agent, target, and behavior_name
    grouped_predictions = {}
    for pred in predictions:
        # Ensure types are native Python ints/strs for grouping
        video_id = int(pred['video_id']) if hasattr(pred['video_id'], 'item') else int(pred['video_id'])
        agent_id = int(pred['agent_id']) if hasattr(pred['agent_id'], 'item') else int(pred['agent_id'])
        target_id = int(pred['target_id']) if hasattr(pred['target_id'], 'item') else int(pred['target_id'])
        behavior_name = str(pred.get('behavior_name', pred.get('behavior')))
        
        key = (video_id, agent_id, target_id, behavior_name)
        if key not in grouped_predictions:
            grouped_predictions[key] = []
        grouped_predictions[key].append(int(pred['frame']))
    
    logger.info(f"Processing {len(grouped_predictions)} prediction groups...")
    
    for (video_id, agent_id, target_id, behavior_name), frames in grouped_predictions.items():
        if not frames:
            continue
        frames = sorted(frames)
        # Build continuous ranges (frames are individual frame indices)
        ranges = []
        start = frames[0]
        end = frames[0]
        for i in range(1, len(frames)):
            if frames[i] == frames[i-1] + 1:
                end = frames[i]
            else:
                if (end - start + 1) >= min_duration:
                    ranges.append((start, end))
                start = end = frames[i]
        if (end - start + 1) >= min_duration:
            ranges.append((start, end))
        
        for start_frame, stop_frame_inclusive in ranges:
            # Convert inclusive stop_frame to exclusive as scorer expects
            stop_frame_exclusive = stop_frame_inclusive + 1
            submission_data.append({
                'video_id': int(video_id),
                'agent_id': int(agent_id),  # No "mouse" prefix
                'target_id': int(target_id),  # No "mouse" prefix
                'action': behavior_name,
                'start_frame': int(start_frame),
                'stop_frame': int(stop_frame_exclusive)
            })
    
    submission_df = pd.DataFrame(submission_data)
    
    # Insert row_id (optional)
    submission_df.insert(0, 'row_id', range(len(submission_df)))
    
    # Save submission
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submissions_dir = Path(cfg['paths']['submissions_dir'])
    submissions_dir.mkdir(parents=True, exist_ok=True)
    submission_path = submissions_dir / f"submission_{timestamp}.csv"
    submission_df.to_csv(submission_path, index=False)
    
    logger.info(f"Submission saved to {submission_path} ({len(submission_df)} rows)")
    return str(submission_path), submission_df
