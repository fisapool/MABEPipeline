"""
SMOTE Augmentation for Behavioral Data

Implements SMOTE (Synthetic Minority Over-sampling Technique) specifically
designed for mouse behavioral data with temporal and spatial features.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class BehavioralSMOTE:
    """
    SMOTE implementation for mouse behavioral data
    
    Handles temporal sequences, spatial features, and class imbalance
    in mouse behavior detection tasks.
    """
    
    def __init__(self, k_neighbors: int = 5, target_ratio: float = 0.5, random_state: int = 42):
        """
        Initialize BehavioralSMOTE
        
        Args:
            k_neighbors: Number of nearest neighbors for interpolation
            target_ratio: Target ratio of minority to majority class samples
            random_state: Random seed for reproducibility
        """
        self.k_neighbors = k_neighbors
        self.target_ratio = target_ratio
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Behavior mapping
        self.behavior_map = {
            'approach': 0, 'attack': 1, 'avoid': 2, 'chase': 3,
            'chaseattack': 4, 'submit': 5, 'rear': 6, 'shepherd': 7
        }
        self.reverse_behavior_map = {v: k for k, v in self.behavior_map.items()}
    
    def augment_dataset(self, frame_labels_df: pd.DataFrame, tracking_data: Dict) -> pd.DataFrame:
        """
        Apply SMOTE augmentation to frame labels dataset
        
        Args:
            frame_labels_df: Original frame labels DataFrame
            tracking_data: Tracking data dictionary
            
        Returns:
            Augmented DataFrame with synthetic samples
        """
        logger.info("Starting SMOTE augmentation for behavioral data...")
        
        # Analyze class distribution
        class_distribution = self._analyze_class_distribution(frame_labels_df)
        logger.info(f"Original class distribution: {class_distribution}")
        
        # Identify classes that need augmentation
        classes_to_augment = self._identify_classes_to_augment(class_distribution)
        logger.info(f"Classes to augment: {classes_to_augment}")
        
        # Extract features for each class
        class_features = {}
        for behavior, class_id in classes_to_augment.items():
            features = self._extract_class_features(frame_labels_df, tracking_data, behavior)
            if len(features) > 0:
                class_features[class_id] = features
                logger.info(f"Extracted {len(features)} features for class {behavior} (id: {class_id})")
        
        # Generate synthetic samples
        synthetic_samples = []
        for behavior, class_id in classes_to_augment.items():
            if class_id in class_features:
                n_samples = self._calculate_samples_needed(class_distribution, behavior)
                if n_samples > 0:
                    synthetic = self._generate_synthetic_samples(
                        class_features[class_id], behavior, n_samples
                    )
                    synthetic_samples.extend(synthetic)
                    logger.info(f"Generated {len(synthetic)} synthetic samples for {behavior}")
        
        # Combine original and synthetic data
        if synthetic_samples:
            synthetic_df = pd.DataFrame(synthetic_samples)
            augmented_df = pd.concat([frame_labels_df, synthetic_df], ignore_index=True)
            logger.info(f"Augmented dataset: {len(frame_labels_df)} -> {len(augmented_df)} samples")
        else:
            logger.warning("No synthetic samples generated")
            augmented_df = frame_labels_df.copy()
        
        return augmented_df
    
    def _analyze_class_distribution(self, frame_labels_df: pd.DataFrame) -> Dict[str, int]:
        """Analyze class distribution in the dataset"""
        behavior_counts = frame_labels_df['behavior'].value_counts()
        distribution = {}
        
        for behavior in self.behavior_map.keys():
            count = behavior_counts.get(behavior, 0)
            distribution[behavior] = count
            logger.info(f"Class {behavior}: {count} samples")
        
        return distribution
    
    def _identify_classes_to_augment(self, class_distribution: Dict[str, int]) -> Dict[str, int]:
        """Identify which classes need augmentation"""
        classes_to_augment = {}
        
        # Find majority class count
        max_count = max(class_distribution.values()) if class_distribution.values() else 0
        minority_threshold = max_count * 0.1  # Less than 10% of majority
        
        for behavior, count in class_distribution.items():
            class_id = self.behavior_map[behavior]
            
            # Augment if:
            # 1. Missing class (count = 0)
            # 2. Very small class (< 1000 samples)
            # 3. Minority class (< 10% of majority)
            if count == 0 or count < 1000 or count < minority_threshold:
                classes_to_augment[behavior] = class_id
                logger.info(f"Will augment {behavior}: {count} samples")
        
        return classes_to_augment
    
    def _extract_class_features(self, frame_labels_df: pd.DataFrame, tracking_data: Dict, 
                              behavior: str) -> List[np.ndarray]:
        """Extract features for a specific behavior class"""
        # Filter samples for this behavior
        behavior_samples = frame_labels_df[frame_labels_df['behavior'] == behavior]
        
        if len(behavior_samples) == 0:
            logger.warning(f"No samples found for behavior: {behavior}")
            return []
        
        features = []
        for _, row in behavior_samples.iterrows():
            video_id = row['video_id']
            frame = row['frame']
            agent_id = row['agent_id']
            target_id = row['target_id']
            
            # Extract features from tracking data
            feature_vector = self._extract_tracking_features(
                tracking_data, video_id, frame, agent_id, target_id
            )
            
            if feature_vector is not None:
                features.append(feature_vector)
        
        return features
    
    def _extract_tracking_features(self, tracking_data: Dict, video_id: str, 
                                  frame: int, agent_id: str, target_id: str) -> Optional[np.ndarray]:
        """Extract 26-dimensional feature vector from tracking data"""
        if video_id not in tracking_data:
            logger.warning(f"SMOTE: Tracking data not available for video {video_id}")
            # Return dummy vector to match preprocessing/inference behavior
            return np.zeros(26, dtype=np.float32)
        
        tracking_df = tracking_data[video_id]
        
        # Log tracking data info for debugging
        logger.debug(f"SMOTE: Extracting features for video {video_id}, frame {frame}, agent {agent_id}, target {target_id}")
        logger.debug(f"SMOTE: Tracking data shape: {tracking_df.shape}, columns: {list(tracking_df.columns)}")
        
        # Get frame data - handle different column names (case-insensitive)
        try:
            # Check for 'frame' or 'Frame' column (case-insensitive)
            frame_col = None
            for col in tracking_df.columns:
                if col.lower() == 'frame':
                    frame_col = col
                    break
            
            if frame_col is not None:
                # Handle both int and float frame values
                frame_data = tracking_df[tracking_df[frame_col] == frame]
                logger.debug(f"SMOTE: Using '{frame_col}' column, found {len(frame_data)} rows")
            else:
                if frame < len(tracking_df):
                    frame_data = tracking_df.iloc[frame:frame+1]
                    logger.debug(f"SMOTE: Using frame as row number, found {len(frame_data)} rows")
                else:
                    logger.warning(f"SMOTE: Frame {frame} not found in tracking data (max frame: {len(tracking_df)-1})")
                    logger.warning(f"SMOTE: Available columns: {list(tracking_df.columns)}")
                    # Return dummy vector to match preprocessing/inference behavior
                    return np.zeros(26, dtype=np.float32)
        except Exception as e:
            logger.warning(f"SMOTE: Error accessing frame data for video {video_id}, frame {frame}: {e}")
            # Return dummy vector to match preprocessing/inference behavior
            return np.zeros(26, dtype=np.float32)
        
        if frame_data.empty:
            logger.warning(f"SMOTE: No data found for frame {frame} in video {video_id}")
            # Return dummy vector to match preprocessing/inference behavior
            return np.zeros(26, dtype=np.float32)
        
        # Extract spatial features (12 features)
        try:
            spatial_features = self._extract_spatial_features(frame_data, agent_id, target_id)
            logger.debug(f"SMOTE: Extracted {len(spatial_features)} spatial features")
        except Exception as e:
            logger.warning(f"SMOTE: Error extracting spatial features: {e}")
            spatial_features = [0.0] * 12  # Default spatial features
        
        # Extract temporal features (10 features)
        try:
            temporal_features = self._extract_temporal_features(tracking_df, frame)
            logger.debug(f"SMOTE: Extracted {len(temporal_features)} temporal features")
        except Exception as e:
            logger.warning(f"SMOTE: Error extracting temporal features: {e}")
            temporal_features = [0.0] * 10  # Default temporal features
        
        # Extract interaction features (4 features)
        try:
            interaction_features = self._extract_interaction_features(frame_data, agent_id, target_id)
            logger.debug(f"SMOTE: Extracted {len(interaction_features)} interaction features")
        except Exception as e:
            logger.warning(f"SMOTE: Error extracting interaction features: {e}")
            interaction_features = [0.0] * 4  # Default interaction features
        
        # Combine all features
        all_features = spatial_features + temporal_features + interaction_features
        
        # Ensure we have exactly 26 features
        if len(all_features) < 26:
            all_features.extend([0.0] * (26 - len(all_features)))
            logger.warning(f"SMOTE: Feature vector too short ({len(all_features)}), padding with zeros")
        elif len(all_features) > 26:
            all_features = all_features[:26]
            logger.warning(f"SMOTE: Feature vector too long ({len(all_features)}), truncating")
        
        # Validate feature quality
        if all(f == 0.0 for f in all_features):
            logger.warning(f"SMOTE: All-zero feature vector detected for video {video_id}, frame {frame}")
            return None
        
        # Log feature statistics for debugging
        non_zero_count = sum(1 for f in all_features if f != 0.0)
        logger.debug(f"SMOTE: Feature vector: {non_zero_count}/26 non-zero values, range: [{min(all_features):.3f}, {max(all_features):.3f}]")
        
        return np.array(all_features, dtype=np.float32)
    
    def _extract_spatial_features(self, frame_data: pd.DataFrame, agent_id: str, target_id: str) -> List[float]:
        """Extract spatial features (12 features)"""
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
    
    def _extract_temporal_features(self, tracking_df: pd.DataFrame, frame: int, window_size: int = 5) -> List[float]:
        """Extract temporal features (10 features)"""
        features = []
        
        # Get window around current frame
        start_frame = max(0, frame - window_size)
        end_frame = min(len(tracking_df), frame + window_size + 1)
        window_data = tracking_df.iloc[start_frame:end_frame]
        
        if len(window_data) < 2:
            return [0.0] * 10
        
        # Find the first available mouse column
        mouse_columns = [col for col in window_data.columns if col.endswith('_body_center_x')]
        
        if mouse_columns:
            mouse_id = mouse_columns[0].split('_')[0]
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
    
    def _extract_interaction_features(self, frame_data: pd.DataFrame, agent_id: str, target_id: str) -> List[float]:
        """Extract interaction features (4 features)"""
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
    
    def _calculate_samples_needed(self, class_distribution: Dict[str, int], behavior: str) -> int:
        """Calculate how many synthetic samples are needed for a class"""
        current_count = class_distribution[behavior]
        max_count = max(class_distribution.values())
        
        if current_count == 0:
            # Missing class: generate 500 samples
            return 500
        elif current_count < 1000:
            # Very small class: generate to reach 50% of majority
            target_count = int(max_count * self.target_ratio)
            return max(0, target_count - current_count)
        else:
            # Small class: generate to reach 30% of majority
            target_count = int(max_count * 0.3)
            return max(0, target_count - current_count)
    
    def _generate_synthetic_samples(self, features: List[np.ndarray], behavior: str, n_samples: int) -> List[Dict]:
        """Generate synthetic samples using SMOTE"""
        if len(features) < 2:
            logger.warning(f"Not enough samples for SMOTE on {behavior}: {len(features)}")
            return []
        
        features_array = np.array(features)
        synthetic_samples = []
        
        # Find k-nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=min(self.k_neighbors, len(features)), algorithm='auto')
        nbrs.fit(features_array)
        
        # Generate synthetic samples
        for _ in range(n_samples):
            # Randomly select a sample
            sample_idx = np.random.randint(0, len(features_array))
            sample = features_array[sample_idx]
            
            # Find k-nearest neighbors
            distances, indices = nbrs.kneighbors([sample])
            
            # Randomly select a neighbor (excluding the sample itself)
            neighbor_idx = np.random.choice(indices[0][1:])  # Skip first (itself)
            neighbor = features_array[neighbor_idx]
            
            # Generate synthetic sample by interpolation
            alpha = np.random.random()  # Random interpolation weight
            synthetic_feature = sample + alpha * (neighbor - sample)
            
            # Validate synthetic feature quality
            if all(f == 0.0 for f in synthetic_feature):
                logger.warning(f"SMOTE: Generated all-zero synthetic feature for {behavior}, skipping")
                continue
            
            # Check for reasonable feature values (not too extreme)
            if np.any(np.abs(synthetic_feature) > 1000):
                logger.warning(f"SMOTE: Generated extreme synthetic feature for {behavior}, clipping values")
                synthetic_feature = np.clip(synthetic_feature, -1000, 1000)
            
            # Create synthetic sample metadata
            synthetic_sample = {
                'video_id': f'synthetic_{behavior}_{len(synthetic_samples)}',
                'frame': np.random.randint(0, 1000),  # Random frame
                'agent_id': 'synthetic_agent',
                'target_id': 'synthetic_target',
                'behavior': behavior,
                'synthetic': True,  # Mark as synthetic
                'feature_vector': synthetic_feature.tolist()  # Store feature vector for validation
            }
            
            synthetic_samples.append(synthetic_sample)
        
        logger.info(f"Generated {len(synthetic_samples)} synthetic samples for {behavior}")
        
        # Validate synthetic samples
        self._validate_synthetic_samples(synthetic_samples, behavior)
        
        return synthetic_samples
    
    def _validate_synthetic_samples(self, synthetic_samples: List[Dict], behavior: str) -> None:
        """Validate quality of synthetic samples"""
        if not synthetic_samples:
            return
        
        logger.info(f"Validating {len(synthetic_samples)} synthetic samples for {behavior}")
        
        # Check feature vector quality
        valid_samples = 0
        all_zero_samples = 0
        extreme_samples = 0
        
        for sample in synthetic_samples:
            if 'feature_vector' in sample:
                feature_vector = np.array(sample['feature_vector'])
                
                # Check for all-zero vectors
                if all(f == 0.0 for f in feature_vector):
                    all_zero_samples += 1
                    continue
                
                # Check for extreme values
                if np.any(np.abs(feature_vector) > 1000):
                    extreme_samples += 1
                    continue
                
                valid_samples += 1
        
        logger.info(f"Synthetic sample validation for {behavior}:")
        logger.info(f"  Valid samples: {valid_samples}/{len(synthetic_samples)}")
        logger.info(f"  All-zero samples: {all_zero_samples}")
        logger.info(f"  Extreme value samples: {extreme_samples}")
        
        if valid_samples < len(synthetic_samples) * 0.5:
            logger.warning(f"Low quality synthetic samples for {behavior}: only {valid_samples}/{len(synthetic_samples)} are valid")


def apply_smote_augmentation(frame_labels_df: pd.DataFrame, tracking_data: Dict, 
                           cfg: Dict) -> pd.DataFrame:
    """
    Apply SMOTE augmentation to frame labels dataset
    
    Args:
        frame_labels_df: Original frame labels DataFrame
        tracking_data: Tracking data dictionary
        cfg: Configuration dictionary
        
    Returns:
        Augmented DataFrame with synthetic samples
    """
    training_cfg = cfg.get('training', {})
    
    if not training_cfg.get('use_smote', False):
        logger.info("SMOTE augmentation disabled")
        return frame_labels_df
    
    # Initialize SMOTE
    smote = BehavioralSMOTE(
        k_neighbors=training_cfg.get('smote_k_neighbors', 5),
        target_ratio=training_cfg.get('smote_target_ratio', 0.5),
        random_state=cfg.get('seed', 42)
    )
    
    # Apply augmentation
    augmented_df = smote.augment_dataset(frame_labels_df, tracking_data)
    
    return augmented_df
