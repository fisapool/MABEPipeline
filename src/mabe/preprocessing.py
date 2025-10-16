"""
MABE Data Preprocessing Module

Refactored from original preprocessing.py to work with configuration system.
Handles data loading, feature extraction, dataset creation, and augmentation.
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

# PyTorch for dataset creation
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight

# Import utilities
from .utils.logger import get_logger
from .utils.io_compat import read_legacy_frame_labels, convert_to_standard_format

logger = get_logger(__name__)


class MouseBehaviorDataset(Dataset):
    """Enhanced dataset with data augmentation for mouse behavior detection"""
    
    def __init__(self, frame_labels_df, tracking_data_dict, augment=False, augment_factor=2.0):
        self.frame_labels = frame_labels_df
        self.tracking_data = tracking_data_dict
        self.augment = augment
        self.augment_factor = augment_factor
        self.behavior_map = {
            'approach': 0, 'attack': 1, 'avoid': 2, 'chase': 3,
            'chaseattack': 4, 'submit': 5, 'rear': 6, 'shepherd': 7
        }
        
        # Calculate class distribution for augmentation
        if augment:
            self._calculate_augmentation_needs()
        
    def _calculate_augmentation_needs(self):
        """Calculate which classes need augmentation"""
        labels = self.frame_labels['behavior'].map(self.behavior_map).fillna(0)
        class_counts = labels.value_counts().sort_index()
        
        # Find majority class count
        max_count = class_counts.max()
        
        # Calculate balanced augmentation multipliers using configurable factor
        self.augment_multipliers = {}
        for class_id, count in class_counts.items():
            if count < 1000:  # Very small classes
                multiplier = self.augment_factor  # Use configurable factor
            elif count < 5000:  # Small classes
                multiplier = self.augment_factor * 0.75  # 75% of configurable factor
            elif count < max_count * 0.1:  # Less than 10% of max
                multiplier = self.augment_factor * 0.5  # 50% of configurable factor
            else:
                multiplier = 1.0  # No augmentation
                
            self.augment_multipliers[class_id] = multiplier
            logger.info(f"Class {class_id} will be augmented by factor {multiplier:.2f} (base factor: {self.augment_factor})")
        
    def __len__(self):
        return len(self.frame_labels)
    
    def __getitem__(self, idx):
        row = self.frame_labels.iloc[idx]
        
        video_id = row['video_id']
        frame = row['frame']
        agent_id = row['agent_id']
        target_id = row['target_id']
        behavior = row['behavior']
        
        # Extract features from tracking data
        features = self.extract_tracking_features(video_id, frame, agent_id, target_id, idx)
        
        # Apply data augmentation if enabled
        if self.augment:
            behavior_encoded = self.behavior_map.get(behavior, 0)
            if behavior_encoded in self.augment_multipliers:
                features = self.apply_augmentation(features)
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features)
        
        # Encode behavior
        behavior_encoded = self.behavior_map.get(behavior, 0)
        
        return features_tensor, behavior_encoded
    
    def apply_augmentation(self, features):
        """Apply data augmentation to features"""
        features = np.array(features)
        
        # Add Gaussian noise (5% of feature std)
        noise_std = np.std(features) * 0.05
        noise = np.random.normal(0, noise_std, features.shape)
        features = features + noise
        
        # Random scaling (0.95 to 1.05)
        scale_factor = np.random.uniform(0.95, 1.05)
        features = features * scale_factor
        
        # Small random shifts for position features (first 12 features are positions)
        if len(features) >= 12:
            position_shift = np.random.normal(0, 0.1, 12)
            features[:12] = features[:12] + position_shift
        
        return features.tolist()
        
    def extract_tracking_features(self, video_id, frame, agent_id, target_id, idx):
        """Extract comprehensive features from tracking data - MATCHES inference.py"""
        features = []
        dummy_vector_returned = False
        
        if self.tracking_data is None or video_id not in self.tracking_data:
            # Return dummy features if tracking data not available
            logger.warning(f"Tracking data not available for video {video_id}, returning dummy features")
            dummy_vector_returned = True
            return [0.0] * 26
        
        tracking_df = self.tracking_data[video_id]
        
        # Log tracking data info for debugging
        logger.debug(f"Extracting features for video {video_id}, frame {frame}, agent {agent_id}, target {target_id}")
        logger.debug(f"Tracking data shape: {tracking_df.shape}, columns: {list(tracking_df.columns)}")
        
        # Try different ways to access frame data
        try:
            # Method 1: Check for 'frame' or 'Frame' column (case-insensitive)
            frame_col = None
            for col in tracking_df.columns:
                if col.lower() == 'frame':
                    frame_col = col
                    break
            
            if frame_col is not None:
                # Handle both int and float frame values
                frame_data = tracking_df[tracking_df[frame_col] == frame]
                logger.debug(f"Using '{frame_col}' column, found {len(frame_data)} rows")
            else:
                # Method 2: If frame is the index
                if frame in tracking_df.index:
                    frame_data = tracking_df.loc[[frame]]
                    logger.debug(f"Using frame as index, found {len(frame_data)} rows")
                else:
                    # Method 3: If frame is a row number
                    if frame < len(tracking_df):
                        frame_data = tracking_df.iloc[[frame]]
                        logger.debug(f"Using frame as row number, found {len(frame_data)} rows")
                    else:
                        logger.warning(f"Frame {frame} not found in tracking data (max frame: {len(tracking_df)-1})")
                        logger.warning(f"Available columns: {list(tracking_df.columns)}")
                        dummy_vector_returned = True
                        return [0.0] * 26
            
            if frame_data.empty:
                logger.warning(f"No data found for frame {frame} in video {video_id}")
                dummy_vector_returned = True
                return [0.0] * 26
                
        except Exception as e:
            logger.warning(f"Error accessing frame data for video {video_id}, frame {frame}: {e}")
            dummy_vector_returned = True
            return [0.0] * 26
        
        # Extract spatial features
        try:
            spatial_features = self.extract_spatial_features(frame_data, agent_id, target_id)
            features.extend(spatial_features)
            logger.debug(f"Extracted {len(spatial_features)} spatial features")
        except Exception as e:
            logger.warning(f"Error extracting spatial features: {e}")
            features.extend([0.0] * 12)  # Default spatial features
        
        # Extract temporal features
        try:
            temporal_features = self.extract_temporal_features(tracking_df, frame)
            features.extend(temporal_features)
            logger.debug(f"Extracted {len(temporal_features)} temporal features")
        except Exception as e:
            logger.warning(f"Error extracting temporal features: {e}")
            features.extend([0.0] * 10)  # Default temporal features
        
        # Extract interaction features
        try:
            interaction_features = self.extract_interaction_features(frame_data, agent_id, target_id)
            features.extend(interaction_features)
            logger.debug(f"Extracted {len(interaction_features)} interaction features")
        except Exception as e:
            logger.warning(f"Error extracting interaction features: {e}")
            features.extend([0.0] * 4)  # Default interaction features
        
        # Validate feature count and quality
        assert len(features) == 26, f"Expected 26 features, got {len(features)}"
        
        # Check for dummy vector
        if all(f == 0.0 for f in features):
            logger.warning(f"All-zero feature vector detected for video {video_id}, frame {frame}")
            dummy_vector_returned = True
        
        # Log feature statistics for debugging
        if not dummy_vector_returned:
            non_zero_count = sum(1 for f in features if f != 0.0)
            logger.debug(f"Feature vector: {non_zero_count}/26 non-zero values, range: [{min(features):.3f}, {max(features):.3f}]")
        
        return features
    
    def extract_spatial_features(self, frame_data, agent_id, target_id):
        """Extract spatial features from frame data - MATCHES train.py and inference.py"""
        features = []
        
        # Agent position (ensure agent_id is string)
        agent_id_str = str(agent_id)
        agent_x_col = f'{agent_id_str}_body_center_x'
        agent_y_col = f'{agent_id_str}_body_center_y'
        
        if agent_x_col in frame_data.columns and agent_y_col in frame_data.columns:
            agent_x = frame_data[agent_x_col].iloc[0] if not frame_data.empty else 0
            agent_y = frame_data[agent_y_col].iloc[0] if not frame_data.empty else 0
            features.extend([agent_x, agent_y])
        else:
            features.extend([0.0, 0.0])
        
        # Target position (ensure target_id is string)
        target_id_str = str(target_id)
        target_x_col = f'{target_id_str}_body_center_x'
        target_y_col = f'{target_id_str}_body_center_y'
        
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
        """Extract temporal features using sliding window - MATCHES train.py and inference.py"""
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
        """Extract interaction features between mice - MATCHES train.py and inference.py"""
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


class MABEDataPreprocessor:
    """Data preprocessing class for MABE behavior detection"""
    
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.dataset_path = Path(cfg['dataset']['path'])
        self.model_save_path = Path(cfg['paths']['models_dir'])
        self.tracking_data = {}
        self.class_weights = None
        
        # Behavior mapping
        self.behavior_map = {
            'approach': 0, 'attack': 1, 'avoid': 2, 'chase': 3,
            'chaseattack': 4, 'submit': 5, 'rear': 6, 'shepherd': 7
        }
        
    def load_data(self, frame_labels_path: str = None, max_videos: int = None):
        """Load frame labels and tracking data"""
        logger.info("Loading data...")
        
        # Use max_videos from config if not provided
        if max_videos is None:
            max_videos = self.cfg.get('training', {}).get('max_videos', 5)
        
        # Load frame labels
        if frame_labels_path and Path(frame_labels_path).exists():
            frame_labels_df = pd.read_csv(frame_labels_path)
            logger.info(f"Loaded frame labels from {frame_labels_path}: {frame_labels_df.shape}")
        else:
            # Try to find legacy frame labels files
            frame_labels_df = self._load_legacy_frame_labels()
            if frame_labels_df is None:
                # Create frame labels from annotations
                frame_labels_df = self._create_frame_labels_from_annotations(max_videos)
        
        # Load tracking data
        self._load_tracking_data(frame_labels_df, max_videos)
        
        return frame_labels_df
    
    def _load_legacy_frame_labels(self):
        """Load legacy frame labels files"""
        # Look for existing frame labels in the model directory
        model_dir = Path(self.cfg['paths']['models_dir']).parent
        frame_labels_files = list(model_dir.glob("mabe_frame_labels_*.csv"))
        
        if frame_labels_files:
            # Use the most recent frame labels file
            frame_labels_path = max(frame_labels_files, key=lambda x: x.stat().st_mtime)
            logger.info(f"Found legacy frame labels: {frame_labels_path}")
            
            try:
                df = read_legacy_frame_labels(str(frame_labels_path))
                return convert_to_standard_format(df)
            except Exception as e:
                logger.warning(f"Error loading legacy frame labels: {e}")
                return None
        
        return None
    
    def _load_tracking_data(self, frame_labels_df, max_videos=5):
        """Load tracking data for videos in frame labels"""
        logger.info("Loading tracking data...")
        
        # Get unique videos and normalize to strings
        unique_videos = frame_labels_df['video_id'].unique()[:max_videos]
        unique_videos = [str(vid) for vid in unique_videos]  # Normalize to strings
        logger.info(f"Requested videos: {unique_videos}")
        
        # Try to load converted tracking data first
        converted_tracking_dir = self.dataset_path / "converted_tracking"
        if converted_tracking_dir.exists():
            logger.info("Loading converted tracking data...")
            from .tracking_converter import TrackingDataConverter
            converter = TrackingDataConverter()
            self.tracking_data = converter.create_tracking_data_loader(str(converted_tracking_dir))
            
            # Filter to only include videos we need and normalize keys
            filtered_data = {}
            for k, v in self.tracking_data.items():
                str_key = str(k)
                if str_key in unique_videos:
                    filtered_data[str_key] = v
            self.tracking_data = filtered_data
            logger.info(f"Loaded converted tracking data for {len(self.tracking_data)} videos")
            return
        
        # If no converted data, try to convert on the fly
        keypoints_dir = self.dataset_path / "train_tracking" / "MABe22_keypoints"
        if keypoints_dir.exists():
            logger.info("Converting tracking data on the fly...")
            from .tracking_converter import TrackingDataConverter
            converter = TrackingDataConverter()
            
            for video_id in unique_videos:
                tracking_file = keypoints_dir / f"{video_id}.parquet"
                
                if tracking_file.exists():
                    try:
                        # Convert the file
                        converted_df = converter.convert_tracking_file(str(tracking_file))
                        self.tracking_data[str(video_id)] = converted_df  # Store with string key
                        logger.info(f"Converted and loaded tracking data for video {video_id}: {converted_df.shape}")
                    except Exception as e:
                        logger.warning(f"Error converting tracking data for video {video_id}: {e}")
                else:
                    logger.warning(f"Tracking file not found for video {video_id}: {tracking_file}")
        else:
            # Fallback: create sample data for testing
            logger.warning("No tracking data found. Creating sample data for testing...")
            from .tracking_converter import create_sample_tracking_data
            sample_data = create_sample_tracking_data()
            # Normalize sample data keys to strings
            self.tracking_data = {str(k): v for k, v in sample_data.items() if str(k) in unique_videos}
            logger.info(f"Using sample tracking data for {len(self.tracking_data)} videos")
        
        logger.info(f"Loaded tracking data for {len(self.tracking_data)} videos")
        logger.info(f"Tracking data keys: {list(self.tracking_data.keys())}")
    
    def _create_frame_labels_from_annotations(self, max_videos=5):
        """Create frame labels from annotation files with behaviors_labeled filtering"""
        logger.info("Creating frame labels from annotations...")
        
        frame_labels = []
        
        # Load train.csv to get video metadata
        train_csv_path = self.dataset_path / "train.csv"
        if not train_csv_path.exists():
            logger.error(f"Train CSV not found: {train_csv_path}")
            return pd.DataFrame()
            
        train_df = pd.read_csv(train_csv_path)
        unique_videos = train_df['video_id'].unique()[:max_videos]
        
        for video_id in unique_videos:
            video_metadata = train_df[train_df['video_id'] == video_id]
            if video_metadata.empty:
                continue
                
            lab_id = video_metadata.iloc[0]['lab_id']
            behaviors_labeled_str = video_metadata.iloc[0]['behaviors_labeled']
            
            # Parse behaviors_labeled to get active behavior keys
            try:
                import json
                behaviors_labeled = json.loads(behaviors_labeled_str)
                active_keys = set(behaviors_labeled)
                logger.info(f"Video {video_id}: Active behaviors = {len(active_keys)} behaviors")
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Could not parse behaviors_labeled for video {video_id}: {e}")
                active_keys = set()  # Skip this video if we can't parse behaviors
            
            # Load annotation data
            annotation_path = self.dataset_path / "train_annotation" / lab_id / f"{video_id}.parquet"
            
            if annotation_path.exists():
                try:
                    annotation_df = pd.read_parquet(annotation_path)
                    
                    # Create frame labels for each annotation
                    for _, annotation in annotation_df.iterrows():
                        agent_id = str(annotation['agent_id'])  # Normalize to string
                        target_id = str(annotation['target_id'])  # Normalize to string
                        action = annotation['action']
                        start_frame = annotation['start_frame']
                        stop_frame = annotation['stop_frame']
                        
                        # Check if this behavior was actively labeled for this video
                        # Convert numeric IDs to mouse format for comparison
                        mouse_agent = f"mouse{agent_id}"
                        mouse_target = f"mouse{target_id}"
                        behavior_key = f"{mouse_agent},{mouse_target},{action}"
                        if behavior_key not in active_keys:
                            continue  # Skip behaviors not in the labeled set
                        
                        # Create a label for each frame in the range
                        for frame in range(start_frame, stop_frame + 1):
                            frame_labels.append({
                                'video_id': video_id,
                                'frame': frame,
                                'agent_id': agent_id,
                                'target_id': target_id,
                                'behavior': action,
                                'interval_start': start_frame,  # Preserve interval context
                                'interval_stop': stop_frame,   # Preserve interval context
                                'frame_position': (frame - start_frame) / max(1, stop_frame - start_frame)  # Relative position in interval
                            })
                    
                    logger.info(f"Created {len(annotation_df)} annotations for {video_id} (filtered by behaviors_labeled)")
                except Exception as e:
                    logger.warning(f"Error loading annotations for {video_id}: {e}")
        
        frame_labels_df = pd.DataFrame(frame_labels)
        logger.info(f"Created {len(frame_labels_df)} frame labels (filtered by behaviors_labeled)")
        
        return frame_labels_df
    
    def extract_features(self, frame_labels_df):
        """Extract features and detect feature dimension"""
        logger.info("Extracting features...")
        
        # Create a sample to detect feature size
        sample_dataset = MouseBehaviorDataset(frame_labels_df.head(1), self.tracking_data)
        sample_features, _ = sample_dataset[0]
        feature_dim = len(sample_features)
        
        logger.info(f"Detected feature dimension: {feature_dim}")
        
        # Calculate class weights
        self._calculate_class_weights(frame_labels_df)
        
        return feature_dim
    
    def _calculate_class_weights(self, frame_labels_df):
        """Calculate class weights for balanced training"""
        logger.info("Calculating class weights for balanced training...")
        
        # Map behaviors to numeric labels
        labels = frame_labels_df['behavior'].map(self.behavior_map).fillna(0)
        
        # Calculate class distribution
        class_counts = labels.value_counts().sort_index()
        logger.info("Class distribution:")
        for behavior, class_id in self.behavior_map.items():
            count = class_counts.get(class_id, 0)
            percentage = (count / len(labels)) * 100
            logger.info(f"  {behavior} (class {class_id}): {count} samples ({percentage:.1f}%)")
        
        # Calculate class weights using sklearn
        unique_classes = np.unique(labels)
        class_weights_array = compute_class_weight(
            class_weight='balanced',
            classes=unique_classes,
            y=labels
        )
        
        # Create full weight array for all 8 classes
        full_weights = np.ones(8)  # Default weight of 1.0 for all classes
        for i, weight in enumerate(class_weights_array):
            class_id = unique_classes[i]
            full_weights[class_id] = weight
        
        # Handle missing classes (set to 1.0 or a small value)
        for class_id in range(8):
            if class_id not in unique_classes:
                full_weights[class_id] = 1.0  # Default weight for missing classes
                logger.info(f"  Missing class {class_id} - using default weight 1.0")
        
        # Convert to tensor
        class_weights_tensor = torch.FloatTensor(full_weights)
        
        logger.info("Calculated class weights:")
        for i, weight in enumerate(full_weights):
            behavior_name = [k for k, v in self.behavior_map.items() if v == i][0] if i in self.behavior_map.values() else f"class_{i}"
            logger.info(f"  {behavior_name} (class {i}): {weight:.3f}")
        
        self.class_weights = class_weights_tensor
        return class_weights_tensor
    
    def create_dataset(self, frame_labels_df, batch_size=None, train_split=None, use_augmentation=None):
        """Create datasets and data loaders"""
        logger.info("Creating datasets and data loaders...")
        
        # Get parameters from config if not provided
        if batch_size is None:
            batch_size = self.cfg.get('training', {}).get('batch_size', 32)
        if train_split is None:
            train_split = self.cfg.get('training', {}).get('val_fraction', 0.2)
        if use_augmentation is None:
            use_augmentation = self.cfg.get('training', {}).get('use_augmentation', True)
        
        # Split data
        train_size = int(len(frame_labels_df) * (1 - train_split))
        train_df = frame_labels_df[:train_size]
        val_df = frame_labels_df[train_size:]
        
        # Get augmentation factor from config
        augment_factor = self.cfg.get('training', {}).get('augment_factor', 2.0)
        
        # Create datasets with augmentation for training
        train_dataset = MouseBehaviorDataset(train_df, self.tracking_data, 
                                           augment=use_augmentation, augment_factor=augment_factor)
        val_dataset = MouseBehaviorDataset(val_df, self.tracking_data, augment=False)
        
        # Create stratified sampler for balanced training
        stratified_sampling = self.cfg.get('training', {}).get('stratified_sampling', True)
        
        if stratified_sampling:
            # Create stratified sampler for balanced batches
            from torch.utils.data import WeightedRandomSampler
            
            train_labels = train_df['behavior'].map(self.behavior_map).values
            
            # Calculate sampling weights to balance classes in each batch
            class_counts = np.bincount(train_labels)
            class_weights = 1.0 / (class_counts + 1)  # +1 to avoid division by zero
            sample_weights = class_weights[train_labels]
            
            # Create weighted sampler
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(train_labels),
                replacement=True
            )
            logger.info("Using stratified sampling for balanced batches")
        else:
            # Use existing weighted sampler
            sampler = self._create_weighted_sampler(train_df)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                sampler=sampler, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                              shuffle=False, num_workers=0)
        
        logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
        logger.info(f"Using data augmentation: {use_augmentation}")
        
        return train_loader, val_loader, train_dataset, val_dataset
    
    def _create_weighted_sampler(self, frame_labels_df):
        """Create weighted sampler for balanced training"""
        # Map behaviors to numeric labels
        labels = frame_labels_df['behavior'].map(self.behavior_map).fillna(0)
        
        # Calculate sample weights
        class_counts = labels.value_counts().sort_index()
        total_samples = len(labels)
        
        # Calculate weight for each sample
        sample_weights = []
        for label in labels:
            class_count = class_counts.get(label, 1)
            weight = total_samples / (len(class_counts) * class_count)
            sample_weights.append(weight)
        
        return WeightedRandomSampler(sample_weights, len(sample_weights))
    
    def save_preprocessing_info(self, feature_dim, frame_labels_df):
        """Save preprocessing information for training"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        preprocessing_info = {
            'feature_dimension': feature_dim,
            'num_samples': len(frame_labels_df),
            'behavior_map': self.behavior_map,
            'class_weights': self.class_weights.tolist() if self.class_weights is not None else None,
            'num_videos_loaded': len(self.tracking_data),
            'preprocessing_timestamp': timestamp
        }
        
        # Save preprocessing info
        info_path = self.model_save_path / f"preprocessing_info_{timestamp}.json"
        with open(info_path, 'w') as f:
            json.dump(preprocessing_info, f, indent=2)
        
        logger.info(f"Saved preprocessing info to {info_path}")
        return info_path, preprocessing_info
    
    def get_class_weights(self):
        """Get calculated class weights"""
        return self.class_weights
    
    def get_behavior_map(self):
        """Get behavior mapping"""
        return self.behavior_map
    
    def get_tracking_data(self):
        """Get loaded tracking data"""
        return self.tracking_data


# Public API functions for the pipeline
def preprocess_data(cfg: Dict, max_videos: int = None) -> pd.DataFrame:
    """
    Main preprocessing function for the pipeline
    
    Args:
        cfg: Configuration dictionary
        max_videos: Maximum number of videos to process
        
    Returns:
        DataFrame with frame labels
    """
    logger.info("Starting data preprocessing...")
    
    # Initialize preprocessor
    preprocessor = MABEDataPreprocessor(cfg)
    
    # Load data
    frame_labels_df = preprocessor.load_data(max_videos=max_videos)
    
    # Extract features
    feature_dim = preprocessor.extract_features(frame_labels_df)
    
    # Save preprocessing info
    info_path, preprocessing_info = preprocessor.save_preprocessing_info(feature_dim, frame_labels_df)
    
    logger.info(f"Preprocessing completed: {len(frame_labels_df)} frame labels, {feature_dim} features")
    
    return frame_labels_df


def extract_features(frame_labels_df: pd.DataFrame, cfg: Dict) -> int:
    """
    Extract features from frame labels
    
    Args:
        frame_labels_df: DataFrame with frame labels
        cfg: Configuration dictionary
        
    Returns:
        Feature dimension
    """
    preprocessor = MABEDataPreprocessor(cfg)
    preprocessor.tracking_data = preprocessor._load_tracking_data(frame_labels_df)
    return preprocessor.extract_features(frame_labels_df)


def save_features(df: pd.DataFrame, path: str) -> None:
    """
    Save features DataFrame to file
    
    Args:
        df: DataFrame to save
        path: Output file path
    """
    df.to_parquet(path, index=False)
    logger.info(f"Features saved to {path}")


def create_dataset(frame_labels_df: pd.DataFrame, cfg: Dict, **kwargs):
    """
    Create PyTorch dataset from frame labels
    
    Args:
        frame_labels_df: DataFrame with frame labels
        cfg: Configuration dictionary
        **kwargs: Additional arguments for dataset creation
        
    Returns:
        Tuple of (train_loader, val_loader, train_dataset, val_dataset)
    """
    preprocessor = MABEDataPreprocessor(cfg)
    preprocessor.tracking_data = preprocessor._load_tracking_data(frame_labels_df)
    return preprocessor.create_dataset(frame_labels_df, **kwargs)
