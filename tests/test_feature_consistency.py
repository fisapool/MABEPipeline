#!/usr/bin/env python3
"""
Feature Consistency Tests

Tests to ensure feature extraction consistency across training, inference, and SMOTE modules.
"""

import sys
import os
import unittest
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mabe.preprocessing import MouseBehaviorDataset
from mabe.inference import TestMouseBehaviorDataset
from mabe.smote_augmentation import BehavioralSMOTE


class TestFeatureConsistency(unittest.TestCase):
    """Test feature extraction consistency across modules"""
    
    def setUp(self):
        """Set up test data"""
        # Create sample tracking data
        self.tracking_data = {
            'test_video': pd.DataFrame({
                'frame': [0, 1, 2, 3, 4, 5],
                'mouse1_body_center_x': [100, 101, 102, 103, 104, 105],
                'mouse1_body_center_y': [200, 201, 202, 203, 204, 205],
                'mouse2_body_center_x': [300, 301, 302, 303, 304, 305],
                'mouse2_body_center_y': [400, 401, 402, 403, 404, 405]
            })
        }
        
        # Create sample frame labels
        self.frame_labels = pd.DataFrame({
            'video_id': ['test_video', 'test_video', 'test_video'],
            'frame': [1, 2, 3],
            'agent_id': ['mouse1', 'mouse1', 'mouse1'],
            'target_id': ['mouse2', 'mouse2', 'mouse2'],
            'behavior': ['approach', 'attack', 'avoid']
        })
        
        # Create test predictions for inference (as DataFrame)
        self.test_predictions = pd.DataFrame({
            'video_id': ['test_video', 'test_video'],
            'frame': [1, 2],
            'agent_id': ['mouse1', 'mouse1'],
            'target_id': ['mouse2', 'mouse2'],
            'frames_per_second': [30, 30]  # Add required column
        })
    
    def test_feature_dimensions(self):
        """Test that all modules produce 26-dimensional features"""
        # Test preprocessing
        train_dataset = MouseBehaviorDataset(self.frame_labels, self.tracking_data, augment=False)
        train_features, _ = train_dataset[0]
        self.assertEqual(len(train_features), 26, "Preprocessing should produce 26 features")
        
        # Test inference
        inference_dataset = TestMouseBehaviorDataset(self.test_predictions, self.tracking_data)
        inference_features = inference_dataset.extract_tracking_features('test_video', 1, 'mouse1', 'mouse2')
        self.assertEqual(len(inference_features), 26, "Inference should produce 26 features")
        
        # Test SMOTE
        smote = BehavioralSMOTE()
        smote_features = smote._extract_tracking_features(self.tracking_data, 'test_video', 1, 'mouse1', 'mouse2')
        self.assertEqual(len(smote_features), 26, "SMOTE should produce 26 features")
    
    def test_feature_consistency(self):
        """Test that all modules produce identical features for same input"""
        # Extract features from all modules
        train_dataset = MouseBehaviorDataset(self.frame_labels, self.tracking_data, augment=False)
        train_features, _ = train_dataset[0]
        
        inference_dataset = TestMouseBehaviorDataset(self.test_predictions, self.tracking_data)
        inference_features = inference_dataset.extract_tracking_features('test_video', 1, 'mouse1', 'mouse2')
        
        smote = BehavioralSMOTE()
        smote_features = smote._extract_tracking_features(self.tracking_data, 'test_video', 1, 'mouse1', 'mouse2')
        
        # Convert to numpy arrays for comparison
        train_array = np.array(train_features)
        inference_array = np.array(inference_features)
        smote_array = np.array(smote_features)
        
        # Check if features are identical
        preprocessing_inference_match = np.allclose(train_array, inference_array, rtol=1e-5)
        preprocessing_smote_match = np.allclose(train_array, smote_array, rtol=1e-5)
        inference_smote_match = np.allclose(inference_array, smote_array, rtol=1e-5)
        
        self.assertTrue(preprocessing_inference_match, 
                       "Preprocessing and inference features should be identical")
        self.assertTrue(preprocessing_smote_match, 
                       "Preprocessing and SMOTE features should be identical")
        self.assertTrue(inference_smote_match, 
                       "Inference and SMOTE features should be identical")
    
    def test_dummy_vector_detection(self):
        """Test detection of dummy vectors"""
        # Create tracking data with missing video
        empty_tracking_data = {}
        
        # Test preprocessing with missing data
        train_dataset = MouseBehaviorDataset(self.frame_labels, empty_tracking_data, augment=False)
        train_features, _ = train_dataset[0]
        self.assertTrue(all(f == 0.0 for f in train_features), 
                       "Should return dummy vector for missing tracking data")
        
        # Test inference with missing data
        inference_dataset = TestMouseBehaviorDataset(self.test_predictions, empty_tracking_data)
        inference_features = inference_dataset.extract_tracking_features('missing_video', 1, 'mouse1', 'mouse2')
        self.assertTrue(all(f == 0.0 for f in inference_features), 
                       "Should return dummy vector for missing tracking data")
        
        # Test SMOTE with missing data
        smote = BehavioralSMOTE()
        smote_features = smote._extract_tracking_features(empty_tracking_data, 'missing_video', 1, 'mouse1', 'mouse2')
        self.assertIsNone(smote_features, "SMOTE should return None for missing tracking data")
    
    def test_feature_quality(self):
        """Test feature quality and non-zero values"""
        # Test with valid tracking data
        train_dataset = MouseBehaviorDataset(self.frame_labels, self.tracking_data, augment=False)
        train_features, _ = train_dataset[0]
        
        # Should have non-zero features
        non_zero_count = sum(1 for f in train_features if f != 0.0)
        self.assertGreater(non_zero_count, 0, "Should have non-zero features with valid tracking data")
        
        # Should not have extreme values
        max_abs_value = max(abs(f) for f in train_features)
        self.assertLess(max_abs_value, 1000, "Should not have extreme feature values")
    
    def test_column_name_handling(self):
        """Test handling of different column name formats"""
        # Create tracking data with different column names
        alt_tracking_data = {
            'test_video': pd.DataFrame({
                'Frame': [0, 1, 2, 3, 4, 5],  # Capital F
                'mouse1_body_center_x': [100, 101, 102, 103, 104, 105],
                'mouse1_body_center_y': [200, 201, 202, 203, 204, 205],
                'mouse2_body_center_x': [300, 301, 302, 303, 304, 305],
                'mouse2_body_center_y': [400, 401, 402, 403, 404, 405]
            })
        }
        
        # Test that modules handle different column names
        train_dataset = MouseBehaviorDataset(self.frame_labels, alt_tracking_data, augment=False)
        train_features, _ = train_dataset[0]
        self.assertEqual(len(train_features), 26, "Should handle different column names")
        
        inference_dataset = TestMouseBehaviorDataset(self.test_predictions, alt_tracking_data)
        inference_features = inference_dataset.extract_tracking_features('test_video', 1, 'mouse1', 'mouse2')
        self.assertEqual(len(inference_features), 26, "Should handle different column names")
    
    def test_synthetic_sample_quality(self):
        """Test quality of synthetic samples from SMOTE"""
        smote = BehavioralSMOTE()
        
        # Extract features for a behavior
        features = []
        for frame in [1, 2, 3]:
            feature_vector = smote._extract_tracking_features(self.tracking_data, 'test_video', frame, 'mouse1', 'mouse2')
            if feature_vector is not None:
                features.append(feature_vector)
        
        # Generate synthetic samples
        synthetic_samples = smote._generate_synthetic_samples(features, 'approach', 10)
        
        # Validate synthetic samples
        self.assertGreater(len(synthetic_samples), 0, "Should generate synthetic samples")
        
        for sample in synthetic_samples:
            self.assertIn('feature_vector', sample, "Synthetic sample should have feature vector")
            self.assertIn('synthetic', sample, "Synthetic sample should be marked as synthetic")
            self.assertTrue(sample['synthetic'], "Synthetic flag should be True")
            
            # Check feature vector quality
            feature_vector = np.array(sample['feature_vector'])
            self.assertEqual(len(feature_vector), 26, "Synthetic feature vector should be 26-dimensional")
            
            # Should not be all zeros
            non_zero_count = np.sum(feature_vector != 0.0)
            self.assertGreater(non_zero_count, 0, "Synthetic features should not be all zeros")


class TestEvaluationPipeline(unittest.TestCase):
    """Test evaluation pipeline diagnostics"""
    
    def setUp(self):
        """Set up test data for evaluation"""
        # Create sample ground truth
        self.ground_truth = pd.DataFrame({
            'video_id': ['test_video', 'test_video', 'test_video'],
            'start_frame': [10, 20, 30],
            'stop_frame': [15, 25, 35],
            'agent_id': ['mouse1', 'mouse1', 'mouse1'],
            'target_id': ['mouse2', 'mouse2', 'mouse2'],
            'action': ['approach', 'attack', 'avoid']
        })
        
        # Create sample predictions
        self.predictions = pd.DataFrame({
            'video_id': ['test_video', 'test_video'],
            'start_frame': [12, 22],
            'stop_frame': [17, 27],
            'agent_id': ['mouse1', 'mouse1'],
            'target_id': ['mouse2', 'mouse2'],
            'action': ['approach', 'attack']
        })
    
    def test_frame_overlap_detection(self):
        """Test frame overlap detection"""
        from mabe.evaluate_local import frames_overlap
        
        # Test overlapping frames
        pred1 = pd.Series({'start_frame': 12, 'stop_frame': 17})
        gt1 = pd.Series({'start_frame': 10, 'stop_frame': 15})
        self.assertTrue(frames_overlap(pred1, gt1), "Should detect frame overlap")
        
        # Test non-overlapping frames
        pred2 = pd.Series({'start_frame': 20, 'stop_frame': 25})
        gt2 = pd.Series({'start_frame': 10, 'stop_frame': 15})
        self.assertFalse(frames_overlap(pred2, gt2), "Should detect no frame overlap")
    
    def test_precision_calculation(self):
        """Test precision calculation with diagnostics"""
        from mabe.evaluate_local import calculate_precision
        
        precision = calculate_precision(self.ground_truth, self.predictions)
        self.assertIsInstance(precision, float, "Precision should be a float")
        self.assertGreaterEqual(precision, 0.0, "Precision should be non-negative")
        self.assertLessEqual(precision, 1.0, "Precision should be at most 1.0")
    
    def test_recall_calculation(self):
        """Test recall calculation with diagnostics"""
        from mabe.evaluate_local import calculate_recall
        
        recall = calculate_recall(self.ground_truth, self.predictions)
        self.assertIsInstance(recall, float, "Recall should be a float")
        self.assertGreaterEqual(recall, 0.0, "Recall should be non-negative")
        self.assertLessEqual(recall, 1.0, "Recall should be at most 1.0")
    
    def test_f_score_calculation(self):
        """Test F-score calculation"""
        from mabe.evaluate_local import calculate_overall_f_score
        
        f_score = calculate_overall_f_score(self.ground_truth, self.predictions)
        self.assertIsInstance(f_score, float, "F-score should be a float")
        self.assertGreaterEqual(f_score, 0.0, "F-score should be non-negative")
        self.assertLessEqual(f_score, 1.0, "F-score should be at most 1.0")


if __name__ == '__main__':
    unittest.main()
