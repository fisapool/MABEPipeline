"""
Tests for preprocessing functionality
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mabe.preprocessing import MABEDataPreprocessor, preprocess_data, extract_features


class TestPreprocessing:
    """Test preprocessing functionality"""
    
    def test_data_loading_with_valid_paths(self):
        """Test data loading with valid paths"""
        # Create temporary directory structure
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create dummy train.csv
            train_csv = temp_path / "train.csv"
            train_df = pd.DataFrame({
                'video_id': ['video_001', 'video_002'],
                'lab_id': ['lab1', 'lab2'],
                'frames_per_second': [30, 30],
                'video_duration_sec': [100, 100]
            })
            train_df.to_csv(train_csv, index=False)
            
            # Create dummy annotation data
            annotation_dir = temp_path / "train_annotation" / "lab1"
            annotation_dir.mkdir(parents=True, exist_ok=True)
            
            annotation_df = pd.DataFrame({
                'agent_id': ['mouse1'],
                'target_id': ['mouse2'],
                'action': ['approach'],
                'start_frame': [10],
                'stop_frame': [20]
            })
            annotation_df.to_parquet(annotation_dir / "video_001.parquet", index=False)
            
            # Create dummy tracking data
            tracking_dir = temp_path / "train_tracking" / "lab1"
            tracking_dir.mkdir(parents=True, exist_ok=True)
            
            tracking_df = pd.DataFrame({
                'frame': [0, 1, 2, 3, 4],
                'mouse1_body_center_x': [100, 101, 102, 103, 104],
                'mouse1_body_center_y': [200, 201, 202, 203, 204],
                'mouse2_body_center_x': [300, 301, 302, 303, 304],
                'mouse2_body_center_y': [400, 401, 402, 403, 404]
            })
            tracking_df.to_parquet(tracking_dir / "video_001.parquet", index=False)
            
            # Test configuration
            cfg = {
                'dataset': {
                    'path': str(temp_path),
                    'train_csv': 'train.csv'
                },
                'paths': {
                    'models_dir': str(temp_path / 'models')
                },
                'training': {
                    'max_videos': 1
                }
            }
            
            # Test data loading
            preprocessor = MABEDataPreprocessor(cfg)
            frame_labels_df = preprocessor.load_data()
            
            # Check results
            assert not frame_labels_df.empty, "Frame labels should not be empty"
            assert 'video_id' in frame_labels_df.columns, "Should have video_id column"
            assert 'frame' in frame_labels_df.columns, "Should have frame column"
            assert 'agent_id' in frame_labels_df.columns, "Should have agent_id column"
            assert 'target_id' in frame_labels_df.columns, "Should have target_id column"
            assert 'behavior' in frame_labels_df.columns, "Should have behavior column"
    
    def test_data_loading_with_invalid_paths(self):
        """Test data loading with invalid paths"""
        # Test configuration with invalid paths
        cfg = {
            'dataset': {
                'path': '/nonexistent/path',
                'train_csv': 'train.csv'
            },
            'paths': {
                'models_dir': '/nonexistent/models'
            },
            'training': {
                'max_videos': 1
            }
        }
        
        # Test data loading
        preprocessor = MABEDataPreprocessor(cfg)
        frame_labels_df = preprocessor.load_data()
        
        # Should return empty DataFrame for invalid paths
        assert frame_labels_df.empty, "Should return empty DataFrame for invalid paths"
    
    def test_tracking_data_loading(self):
        """Test tracking data loading"""
        # Create temporary directory structure
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create dummy train.csv
            train_csv = temp_path / "train.csv"
            train_df = pd.DataFrame({
                'video_id': ['video_001'],
                'lab_id': ['lab1'],
                'frames_per_second': [30],
                'video_duration_sec': [100]
            })
            train_df.to_csv(train_csv, index=False)
            
            # Create dummy tracking data
            tracking_dir = temp_path / "train_tracking" / "lab1"
            tracking_dir.mkdir(parents=True, exist_ok=True)
            
            tracking_df = pd.DataFrame({
                'frame': [0, 1, 2, 3, 4],
                'mouse1_body_center_x': [100, 101, 102, 103, 104],
                'mouse1_body_center_y': [200, 201, 202, 203, 204],
                'mouse2_body_center_x': [300, 301, 302, 303, 304],
                'mouse2_body_center_y': [400, 401, 402, 403, 404]
            })
            tracking_df.to_parquet(tracking_dir / "video_001.parquet", index=False)
            
            # Test configuration
            cfg = {
                'dataset': {
                    'path': str(temp_path),
                    'train_csv': 'train.csv'
                },
                'paths': {
                    'models_dir': str(temp_path / 'models')
                },
                'training': {
                    'max_videos': 1
                }
            }
            
            # Test tracking data loading
            preprocessor = MABEDataPreprocessor(cfg)
            frame_labels_df = pd.DataFrame({
                'video_id': ['video_001'],
                'frame': [2],
                'agent_id': ['mouse1'],
                'target_id': ['mouse2'],
                'behavior': ['approach']
            })
            
            preprocessor._load_tracking_data(frame_labels_df)
            
            # Check that tracking data was loaded
            assert 'video_001' in preprocessor.tracking_data, "Should load tracking data for video_001"
            assert len(preprocessor.tracking_data['video_001']) == 5, "Should load 5 frames of tracking data"
    
    def test_dataset_creation(self):
        """Test dataset creation"""
        # Create dummy frame labels
        frame_labels_df = pd.DataFrame({
            'video_id': ['video_001', 'video_001'],
            'frame': [10, 20],
            'agent_id': ['mouse1', 'mouse1'],
            'target_id': ['mouse2', 'mouse2'],
            'behavior': ['approach', 'attack']
        })
        
        # Create dummy tracking data
        tracking_data = {
            'video_001': pd.DataFrame({
                'frame': [0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24],
                'mouse1_body_center_x': [100, 101, 102, 103, 104, 110, 111, 112, 113, 114, 120, 121, 122, 123, 124],
                'mouse1_body_center_y': [200, 201, 202, 203, 204, 210, 211, 212, 213, 214, 220, 221, 222, 223, 224],
                'mouse2_body_center_x': [300, 301, 302, 303, 304, 310, 311, 312, 313, 314, 320, 321, 322, 323, 324],
                'mouse2_body_center_y': [400, 401, 402, 403, 404, 410, 411, 412, 413, 414, 420, 421, 422, 423, 424]
            })
        }
        
        # Test configuration
        cfg = {
            'training': {
                'batch_size': 2,
                'val_fraction': 0.5,
                'use_augmentation': False
            },
            'paths': {
                'models_dir': 'outputs/models'
            }
        }
        
        # Test dataset creation
        preprocessor = MABEDataPreprocessor(cfg)
        preprocessor.tracking_data = tracking_data
        
        train_loader, val_loader, train_dataset, val_dataset = preprocessor.create_dataset(frame_labels_df)
        
        # Check dataset creation
        assert train_loader is not None, "Should create train loader"
        assert val_loader is not None, "Should create validation loader"
        assert train_dataset is not None, "Should create train dataset"
        assert val_dataset is not None, "Should create validation dataset"
        
        # Check dataset sizes
        assert len(train_dataset) > 0, "Train dataset should not be empty"
        assert len(val_dataset) > 0, "Validation dataset should not be empty"
    
    def test_class_weight_calculation(self):
        """Test class weight calculation"""
        # Create dummy frame labels with imbalanced classes
        frame_labels_df = pd.DataFrame({
            'video_id': ['video_001'] * 10,
            'frame': list(range(10)),
            'agent_id': ['mouse1'] * 10,
            'target_id': ['mouse2'] * 10,
            'behavior': ['approach'] * 8 + ['attack'] * 2  # Imbalanced classes
        })
        
        # Test configuration
        cfg = {
            'paths': {
                'models_dir': 'outputs/models'
            }
        }
        
        # Test class weight calculation
        preprocessor = MABEDataPreprocessor(cfg)
        preprocessor._calculate_class_weights(frame_labels_df)
        
        # Check that class weights were calculated
        assert preprocessor.class_weights is not None, "Should calculate class weights"
        assert len(preprocessor.class_weights) == 8, "Should have weights for 8 behavior classes"
        assert all(w > 0 for w in preprocessor.class_weights), "All class weights should be positive"
    
    def test_preprocess_data_function(self):
        """Test preprocess_data function"""
        # Create temporary directory structure
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create dummy train.csv
            train_csv = temp_path / "train.csv"
            train_df = pd.DataFrame({
                'video_id': ['video_001'],
                'lab_id': ['lab1'],
                'frames_per_second': [30],
                'video_duration_sec': [100]
            })
            train_df.to_csv(train_csv, index=False)
            
            # Create dummy annotation data
            annotation_dir = temp_path / "train_annotation" / "lab1"
            annotation_dir.mkdir(parents=True, exist_ok=True)
            
            annotation_df = pd.DataFrame({
                'agent_id': ['mouse1'],
                'target_id': ['mouse2'],
                'action': ['approach'],
                'start_frame': [10],
                'stop_frame': [20]
            })
            annotation_df.to_parquet(annotation_dir / "video_001.parquet", index=False)
            
            # Create dummy tracking data
            tracking_dir = temp_path / "train_tracking" / "lab1"
            tracking_dir.mkdir(parents=True, exist_ok=True)
            
            tracking_df = pd.DataFrame({
                'frame': [0, 1, 2, 3, 4],
                'mouse1_body_center_x': [100, 101, 102, 103, 104],
                'mouse1_body_center_y': [200, 201, 202, 203, 204],
                'mouse2_body_center_x': [300, 301, 302, 303, 304],
                'mouse2_body_center_y': [400, 401, 402, 403, 404]
            })
            tracking_df.to_parquet(tracking_dir / "video_001.parquet", index=False)
            
            # Test configuration
            cfg = {
                'dataset': {
                    'path': str(temp_path),
                    'train_csv': 'train.csv'
                },
                'paths': {
                    'models_dir': str(temp_path / 'models')
                },
                'training': {
                    'max_videos': 1
                }
            }
            
            # Test preprocess_data function
            frame_labels_df = preprocess_data(cfg)
            
            # Check results
            assert not frame_labels_df.empty, "Should return non-empty DataFrame"
            assert 'video_id' in frame_labels_df.columns, "Should have video_id column"
            assert 'frame' in frame_labels_df.columns, "Should have frame column"
            assert 'agent_id' in frame_labels_df.columns, "Should have agent_id column"
            assert 'target_id' in frame_labels_df.columns, "Should have target_id column"
            assert 'behavior' in frame_labels_df.columns, "Should have behavior column"
    
    def test_extract_features_function(self):
        """Test extract_features function"""
        # Create dummy frame labels
        frame_labels_df = pd.DataFrame({
            'video_id': ['video_001'],
            'frame': [2],
            'agent_id': ['mouse1'],
            'target_id': ['mouse2'],
            'behavior': ['approach']
        })
        
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
        
        # Test configuration
        cfg = {
            'dataset': {
                'path': 'dummy_path'
            },
            'paths': {
                'models_dir': 'outputs/models'
            }
        }
        
        # Test extract_features function
        feature_dim = extract_features(frame_labels_df, cfg)
        
        # Check results
        assert feature_dim == 26, f"Should return feature dimension 26, got {feature_dim}"


if __name__ == "__main__":
    pytest.main([__file__])
