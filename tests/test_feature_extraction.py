"""
Tests for feature extraction functionality
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mabe.preprocessing import MouseBehaviorDataset


class TestFeatureExtraction:
    """Test feature extraction functionality"""
    
    def test_feature_extraction_returns_correct_shape(self):
        """Test that feature extraction returns correct shape (26 features)"""
        # Create dummy tracking data
        tracking_data = {
            'video_001': pd.DataFrame({
                'frame': [0, 1, 2, 3, 4],
                'mouse1_body_center_x': [100, 101, 102, 103, 104],
                'mouse1_body_center_y': [200, 201, 202, 203, 204],
                'mouse2_body_center_x': [300, 301, 302, 303, 304],
                'mouse2_body_center_y': [400, 401, 402, 403, 404]
            })
        }
        
        # Create dummy frame labels
        frame_labels = pd.DataFrame({
            'video_id': ['video_001'],
            'frame': [2],
            'agent_id': ['mouse1'],
            'target_id': ['mouse2'],
            'behavior': ['approach']
        })
        
        # Create dataset
        dataset = MouseBehaviorDataset(frame_labels, tracking_data, augment=False)
        
        # Test feature extraction
        features, label = dataset[0]
        
        # Check feature shape
        assert len(features) == 26, f"Expected 26 features, got {len(features)}"
        assert isinstance(features, list), "Features should be a list"
        assert all(isinstance(f, (int, float)) for f in features), "All features should be numeric"
    
    def test_feature_extraction_non_zero_features(self):
        """Test that feature extraction produces non-zero features on sample data"""
        # Create dummy tracking data with non-zero values
        tracking_data = {
            'video_001': pd.DataFrame({
                'frame': [0, 1, 2, 3, 4],
                'mouse1_body_center_x': [100, 101, 102, 103, 104],
                'mouse1_body_center_y': [200, 201, 202, 203, 204],
                'mouse2_body_center_x': [300, 301, 302, 303, 304],
                'mouse2_body_center_y': [400, 401, 402, 403, 404]
            })
        }
        
        # Create dummy frame labels
        frame_labels = pd.DataFrame({
            'video_id': ['video_001'],
            'frame': [2],
            'agent_id': ['mouse1'],
            'target_id': ['mouse2'],
            'behavior': ['approach']
        })
        
        # Create dataset
        dataset = MouseBehaviorDataset(frame_labels, tracking_data, augment=False)
        
        # Test feature extraction
        features, label = dataset[0]
        
        # Check that features are not all zero
        non_zero_features = [f for f in features if f != 0.0]
        assert len(non_zero_features) > 0, "Should have some non-zero features"
        
        # Check that spatial features are reasonable
        assert features[0] != 0.0, "Agent X position should not be zero"
        assert features[1] != 0.0, "Agent Y position should not be zero"
        assert features[2] != 0.0, "Target X position should not be zero"
        assert features[3] != 0.0, "Target Y position should not be zero"
    
    def test_spatial_feature_extraction(self):
        """Test spatial feature extraction functions"""
        # Create dummy frame data
        frame_data = pd.DataFrame({
            'mouse1_body_center_x': [100],
            'mouse1_body_center_y': [200],
            'mouse2_body_center_x': [300],
            'mouse2_body_center_y': [400]
        })
        
        # Create dataset
        dataset = MouseBehaviorDataset(pd.DataFrame(), {}, augment=False)
        
        # Test spatial feature extraction
        spatial_features = dataset.extract_spatial_features(frame_data, 'mouse1', 'mouse2')
        
        # Check spatial features
        assert len(spatial_features) == 12, f"Expected 12 spatial features, got {len(spatial_features)}"
        assert spatial_features[0] == 100, "Agent X position should be 100"
        assert spatial_features[1] == 200, "Agent Y position should be 200"
        assert spatial_features[2] == 300, "Target X position should be 300"
        assert spatial_features[3] == 400, "Target Y position should be 400"
        
        # Check distance calculation
        expected_distance = np.sqrt((300 - 100)**2 + (400 - 200)**2)
        assert abs(spatial_features[4] - expected_distance) < 1e-6, "Distance calculation incorrect"
    
    def test_temporal_feature_extraction(self):
        """Test temporal feature extraction functions"""
        # Create dummy tracking data
        tracking_df = pd.DataFrame({
            'frame': [0, 1, 2, 3, 4],
            'mouse1_body_center_x': [100, 101, 102, 103, 104],
            'mouse1_body_center_y': [200, 201, 202, 203, 204]
        })
        
        # Create dataset
        dataset = MouseBehaviorDataset(pd.DataFrame(), {}, augment=False)
        
        # Test temporal feature extraction
        temporal_features = dataset.extract_temporal_features(tracking_df, 2)
        
        # Check temporal features
        assert len(temporal_features) == 10, f"Expected 10 temporal features, got {len(temporal_features)}"
        assert all(isinstance(f, (int, float)) for f in temporal_features), "All temporal features should be numeric"
    
    def test_interaction_feature_extraction(self):
        """Test interaction feature extraction functions"""
        # Create dummy frame data
        frame_data = pd.DataFrame({
            'mouse1_body_center_x': [100],
            'mouse1_body_center_y': [200],
            'mouse2_body_center_x': [300],
            'mouse2_body_center_y': [400]
        })
        
        # Create dataset
        dataset = MouseBehaviorDataset(pd.DataFrame(), {}, augment=False)
        
        # Test interaction feature extraction
        interaction_features = dataset.extract_interaction_features(frame_data, 'mouse1', 'mouse2')
        
        # Check interaction features
        assert len(interaction_features) == 4, f"Expected 4 interaction features, got {len(interaction_features)}"
        assert all(isinstance(f, (int, float)) for f in interaction_features), "All interaction features should be numeric"
        
        # Check distance calculation
        expected_distance = np.sqrt((300 - 100)**2 + (400 - 200)**2)
        assert abs(interaction_features[0] - expected_distance) < 1e-6, "Distance calculation incorrect"
    
    def test_feature_extraction_with_missing_data(self):
        """Test feature extraction handles missing data gracefully"""
        # Create dummy tracking data with missing columns
        tracking_data = {
            'video_001': pd.DataFrame({
                'frame': [0, 1, 2, 3, 4],
                'mouse1_body_center_x': [100, 101, 102, 103, 104],
                'mouse1_body_center_y': [200, 201, 202, 203, 204]
                # Missing mouse2 columns
            })
        }
        
        # Create dummy frame labels
        frame_labels = pd.DataFrame({
            'video_id': ['video_001'],
            'frame': [2],
            'agent_id': ['mouse1'],
            'target_id': ['mouse2'],
            'behavior': ['approach']
        })
        
        # Create dataset
        dataset = MouseBehaviorDataset(frame_labels, tracking_data, augment=False)
        
        # Test feature extraction
        features, label = dataset[0]
        
        # Should still return 26 features
        assert len(features) == 26, f"Expected 26 features, got {len(features)}"
        
        # Missing features should be zero
        assert features[2] == 0.0, "Target X position should be zero for missing data"
        assert features[3] == 0.0, "Target Y position should be zero for missing data"
    
    def test_feature_extraction_with_empty_tracking_data(self):
        """Test feature extraction handles empty tracking data gracefully"""
        # Create empty tracking data
        tracking_data = {}
        
        # Create dummy frame labels
        frame_labels = pd.DataFrame({
            'video_id': ['video_001'],
            'frame': [2],
            'agent_id': ['mouse1'],
            'target_id': ['mouse2'],
            'behavior': ['approach']
        })
        
        # Create dataset
        dataset = MouseBehaviorDataset(frame_labels, tracking_data, augment=False)
        
        # Test feature extraction
        features, label = dataset[0]
        
        # Should still return 26 features (all zeros)
        assert len(features) == 26, f"Expected 26 features, got {len(features)}"
        assert all(f == 0.0 for f in features), "All features should be zero for empty tracking data"
    
    def test_behavior_mapping(self):
        """Test behavior mapping in dataset"""
        # Create dummy tracking data
        tracking_data = {
            'video_001': pd.DataFrame({
                'frame': [0, 1, 2, 3, 4],
                'mouse1_body_center_x': [100, 101, 102, 103, 104],
                'mouse1_body_center_y': [200, 201, 202, 203, 204],
                'mouse2_body_center_x': [300, 301, 302, 303, 304],
                'mouse2_body_center_y': [400, 401, 402, 403, 404]
            })
        }
        
        # Test different behaviors
        behaviors = ['approach', 'attack', 'avoid', 'chase', 'chaseattack', 'submit', 'rear', 'shepherd']
        
        for behavior in behaviors:
            frame_labels = pd.DataFrame({
                'video_id': ['video_001'],
                'frame': [2],
                'agent_id': ['mouse1'],
                'target_id': ['mouse2'],
                'behavior': [behavior]
            })
            
            dataset = MouseBehaviorDataset(frame_labels, tracking_data, augment=False)
            features, label = dataset[0]
            
            # Check that label is correctly mapped
            expected_label = dataset.behavior_map[behavior]
            assert label == expected_label, f"Behavior {behavior} should map to {expected_label}, got {label}"


if __name__ == "__main__":
    pytest.main([__file__])
