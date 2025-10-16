#!/usr/bin/env python3
"""
Feature Extraction Diagnostic Script

Validates feature extraction consistency across training, inference, and SMOTE modules.
Identifies dummy vectors, column name mismatches, and feature quality issues.
"""

import sys
import os
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mabe.preprocessing import MouseBehaviorDataset
from mabe.inference import TestMouseBehaviorDataset
from mabe.smote_augmentation import BehavioralSMOTE
from mabe.utils import load_config, get_logger

logger = get_logger(__name__)


class FeatureDiagnostic:
    """Diagnostic tool for feature extraction consistency"""
    
    def __init__(self, cfg_path: str = "configs/default.yaml"):
        """Initialize diagnostic with configuration"""
        self.cfg = load_config(cfg_path)
        self.results = {
            'feature_consistency': {},
            'dummy_vector_stats': {},
            'column_name_mapping': {},
            'feature_quality': {},
            'tracking_data_info': {}
        }
        
    def run_full_diagnostic(self) -> Dict[str, Any]:
        """Run complete feature extraction diagnostic"""
        logger.info("Starting feature extraction diagnostic...")
        
        # Load sample data
        self._load_sample_data()
        
        # Test feature extraction consistency
        self._test_feature_consistency()
        
        # Analyze dummy vectors
        self._analyze_dummy_vectors()
        
        # Check column name mappings
        self._check_column_mappings()
        
        # Analyze feature quality
        self._analyze_feature_quality()
        
        # Inspect tracking data
        self._inspect_tracking_data()
        
        # Generate report
        self._generate_report()
        
        return self.results
    
    def _load_sample_data(self):
        """Load sample tracking data and frame labels for testing"""
        logger.info("Loading sample data...")
        
        # Create minimal test data
        self.tracking_data = {
            'test_video': pd.DataFrame({
                'frame': [0, 1, 2, 3, 4, 5],
                'mouse1_body_center_x': [100, 101, 102, 103, 104, 105],
                'mouse1_body_center_y': [200, 201, 202, 203, 204, 205],
                'mouse2_body_center_x': [300, 301, 302, 303, 304, 305],
                'mouse2_body_center_y': [400, 401, 402, 403, 404, 405]
            })
        }
        
        self.frame_labels = pd.DataFrame({
            'video_id': ['test_video', 'test_video', 'test_video'],
            'frame': [1, 2, 3],
            'agent_id': ['mouse1', 'mouse1', 'mouse1'],
            'target_id': ['mouse2', 'mouse2', 'mouse2'],
            'behavior': ['approach', 'attack', 'avoid']
        })
        
        logger.info(f"Loaded {len(self.frame_labels)} frame labels")
    
    def _test_feature_consistency(self):
        """Test feature extraction consistency across modules"""
        logger.info("Testing feature extraction consistency...")
        
        consistency_results = {}
        
        # Test preprocessing module
        try:
            train_dataset = MouseBehaviorDataset(self.frame_labels, self.tracking_data, augment=False)
            train_features, _ = train_dataset[0]
            consistency_results['preprocessing'] = {
                'success': True,
                'feature_count': len(train_features),
                'is_dummy': all(f == 0.0 for f in train_features),
                'sample_features': train_features[:5]  # First 5 features
            }
        except Exception as e:
            consistency_results['preprocessing'] = {
                'success': False,
                'error': str(e)
            }
        
        # Test inference module
        try:
            # Create test dataset for inference
            test_predictions = [
                {'video_id': 'test_video', 'frame': 1, 'agent_id': 'mouse1', 'target_id': 'mouse2'}
            ]
            inference_dataset = TestMouseBehaviorDataset(test_predictions, self.tracking_data)
            inference_features = inference_dataset.extract_tracking_features('test_video', 1, 'mouse1', 'mouse2')
            consistency_results['inference'] = {
                'success': True,
                'feature_count': len(inference_features),
                'is_dummy': all(f == 0.0 for f in inference_features),
                'sample_features': inference_features[:5]
            }
        except Exception as e:
            consistency_results['inference'] = {
                'success': False,
                'error': str(e)
            }
        
        # Test SMOTE module
        try:
            smote = BehavioralSMOTE()
            smote_features = smote._extract_tracking_features(self.tracking_data, 'test_video', 1, 'mouse1', 'mouse2')
            consistency_results['smote'] = {
                'success': True,
                'feature_count': len(smote_features) if smote_features is not None else 0,
                'is_dummy': all(f == 0.0 for f in smote_features) if smote_features is not None else True,
                'sample_features': smote_features[:5] if smote_features is not None else None
            }
        except Exception as e:
            consistency_results['smote'] = {
                'success': False,
                'error': str(e)
            }
        
        # Compare features if all successful
        if all(result['success'] for result in consistency_results.values()):
            train_feat = consistency_results['preprocessing']['sample_features']
            inf_feat = consistency_results['inference']['sample_features']
            smote_feat = consistency_results['smote']['sample_features']
            
            preprocessing_inference_match = train_feat == inf_feat
            preprocessing_smote_match = train_feat == smote_feat
            inference_smote_match = inf_feat == smote_feat
            
            consistency_results['comparisons'] = {
                'preprocessing_inference_match': preprocessing_inference_match,
                'preprocessing_smote_match': preprocessing_smote_match,
                'inference_smote_match': inference_smote_match,
                'all_identical': preprocessing_inference_match and preprocessing_smote_match
            }
        
        self.results['feature_consistency'] = consistency_results
        logger.info("Feature consistency test completed")
    
    def _analyze_dummy_vectors(self):
        """Analyze proportion of dummy vectors in feature extraction"""
        logger.info("Analyzing dummy vectors...")
        
        dummy_stats = {
            'preprocessing_dummy_count': 0,
            'inference_dummy_count': 0,
            'smote_dummy_count': 0,
            'total_samples_tested': len(self.frame_labels)
        }
        
        # Test preprocessing
        try:
            train_dataset = MouseBehaviorDataset(self.frame_labels, self.tracking_data, augment=False)
            for i in range(len(train_dataset)):
                features, _ = train_dataset[i]
                if all(f == 0.0 for f in features):
                    dummy_stats['preprocessing_dummy_count'] += 1
        except Exception as e:
            logger.warning(f"Error testing preprocessing dummy vectors: {e}")
        
        # Test inference
        try:
            test_predictions = [
                {'video_id': 'test_video', 'frame': frame, 'agent_id': 'mouse1', 'target_id': 'mouse2'}
                for frame in [1, 2, 3]
            ]
            inference_dataset = TestMouseBehaviorDataset(test_predictions, self.tracking_data)
            for i in range(len(inference_dataset)):
                features, _, _, _, _ = inference_dataset[i]
                if all(f == 0.0 for f in features):
                    dummy_stats['inference_dummy_count'] += 1
        except Exception as e:
            logger.warning(f"Error testing inference dummy vectors: {e}")
        
        # Test SMOTE
        try:
            smote = BehavioralSMOTE()
            for frame in [1, 2, 3]:
                features = smote._extract_tracking_features(self.tracking_data, 'test_video', frame, 'mouse1', 'mouse2')
                if features is None or all(f == 0.0 for f in features):
                    dummy_stats['smote_dummy_count'] += 1
        except Exception as e:
            logger.warning(f"Error testing SMOTE dummy vectors: {e}")
        
        # Calculate percentages
        total = dummy_stats['total_samples_tested']
        dummy_stats['preprocessing_dummy_pct'] = (dummy_stats['preprocessing_dummy_count'] / total) * 100
        dummy_stats['inference_dummy_pct'] = (dummy_stats['inference_dummy_count'] / total) * 100
        dummy_stats['smote_dummy_pct'] = (dummy_stats['smote_dummy_count'] / total) * 100
        
        self.results['dummy_vector_stats'] = dummy_stats
        logger.info(f"Dummy vector analysis completed: {dummy_stats}")
    
    def _check_column_mappings(self):
        """Check column name mappings in tracking data"""
        logger.info("Checking column name mappings...")
        
        mapping_info = {
            'frame_column_variants': [],
            'mouse_id_formats': [],
            'coordinate_patterns': [],
            'detected_columns': []
        }
        
        if self.tracking_data:
            sample_df = list(self.tracking_data.values())[0]
            mapping_info['detected_columns'] = list(sample_df.columns)
            
            # Check frame column variants
            frame_cols = [col for col in sample_df.columns if 'frame' in col.lower()]
            mapping_info['frame_column_variants'] = frame_cols
            
            # Check mouse ID formats
            mouse_cols = [col for col in sample_df.columns if 'mouse' in col.lower()]
            mapping_info['mouse_id_formats'] = mouse_cols
            
            # Check coordinate patterns
            coord_cols = [col for col in sample_df.columns if any(x in col.lower() for x in ['x', 'y', 'center'])]
            mapping_info['coordinate_patterns'] = coord_cols
        
        self.results['column_name_mapping'] = mapping_info
        logger.info(f"Column mapping analysis completed: {mapping_info}")
    
    def _analyze_feature_quality(self):
        """Analyze feature quality and statistics"""
        logger.info("Analyzing feature quality...")
        
        quality_stats = {
            'feature_dimensions': [],
            'feature_statistics': {},
            'zero_feature_counts': [],
            'constant_feature_counts': []
        }
        
        try:
            # Test with preprocessing
            train_dataset = MouseBehaviorDataset(self.frame_labels, self.tracking_data, augment=False)
            all_features = []
            
            for i in range(len(train_dataset)):
                features, _ = train_dataset[i]
                all_features.append(features)
                quality_stats['feature_dimensions'].append(len(features))
            
            if all_features:
                # Convert to numpy array for statistics
                feature_matrix = np.array(all_features)
                
                # Calculate statistics per dimension
                for dim in range(feature_matrix.shape[1]):
                    dim_features = feature_matrix[:, dim]
                    quality_stats['feature_statistics'][f'dim_{dim}'] = {
                        'mean': float(np.mean(dim_features)),
                        'std': float(np.std(dim_features)),
                        'min': float(np.min(dim_features)),
                        'max': float(np.max(dim_features)),
                        'zero_count': int(np.sum(dim_features == 0.0)),
                        'constant': bool(np.std(dim_features) == 0.0)
                    }
                
                # Count zero and constant features
                quality_stats['zero_feature_counts'] = [
                    int(np.sum(feature_matrix[:, dim] == 0.0)) 
                    for dim in range(feature_matrix.shape[1])
                ]
                quality_stats['constant_feature_counts'] = [
                    int(np.std(feature_matrix[:, dim]) == 0.0)
                    for dim in range(feature_matrix.shape[1])
                ]
        
        except Exception as e:
            logger.warning(f"Error analyzing feature quality: {e}")
            quality_stats['error'] = str(e)
        
        self.results['feature_quality'] = quality_stats
        logger.info("Feature quality analysis completed")
    
    def _inspect_tracking_data(self):
        """Inspect raw tracking data format"""
        logger.info("Inspecting tracking data format...")
        
        tracking_info = {
            'available_videos': list(self.tracking_data.keys()),
            'sample_data_shape': {},
            'column_info': {},
            'frame_range': {},
            'coordinate_ranges': {}
        }
        
        for video_id, df in self.tracking_data.items():
            tracking_info['sample_data_shape'][video_id] = df.shape
            tracking_info['column_info'][video_id] = {
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict()
            }
            
            if 'frame' in df.columns:
                tracking_info['frame_range'][video_id] = {
                    'min': int(df['frame'].min()),
                    'max': int(df['frame'].max()),
                    'count': len(df)
                }
            
            # Check coordinate ranges
            coord_cols = [col for col in df.columns if any(x in col.lower() for x in ['x', 'y'])]
            if coord_cols:
                coord_ranges = {}
                for col in coord_cols:
                    coord_ranges[col] = {
                        'min': float(df[col].min()),
                        'max': float(df[col].max()),
                        'mean': float(df[col].mean())
                    }
                tracking_info['coordinate_ranges'][video_id] = coord_ranges
        
        self.results['tracking_data_info'] = tracking_info
        logger.info("Tracking data inspection completed")
    
    def _generate_report(self):
        """Generate comprehensive diagnostic report"""
        logger.info("Generating diagnostic report...")
        
        # Save results to JSON
        output_path = Path("outputs/feature_diagnostic_report.json")
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"Diagnostic report saved to {output_path}")
        
        # Print summary
        self._print_summary()
    
    def _print_summary(self):
        """Print diagnostic summary"""
        print("\n" + "="*60)
        print("FEATURE EXTRACTION DIAGNOSTIC SUMMARY")
        print("="*60)
        
        # Feature consistency
        consistency = self.results['feature_consistency']
        print(f"\n1. FEATURE CONSISTENCY:")
        for module, result in consistency.items():
            if module != 'comparisons':
                status = "[SUCCESS]" if result.get('success', False) else "[FAILED]"
                print(f"   {module.capitalize()}: {status}")
                if not result.get('success', False):
                    print(f"     Error: {result.get('error', 'Unknown')}")
        
        if 'comparisons' in consistency:
            comp = consistency['comparisons']
            print(f"   All modules identical: {comp.get('all_identical', False)}")
        
        # Dummy vectors
        dummy_stats = self.results['dummy_vector_stats']
        print(f"\n2. DUMMY VECTOR ANALYSIS:")
        print(f"   Preprocessing: {dummy_stats.get('preprocessing_dummy_pct', 0):.1f}% dummy vectors")
        print(f"   Inference: {dummy_stats.get('inference_dummy_pct', 0):.1f}% dummy vectors")
        print(f"   SMOTE: {dummy_stats.get('smote_dummy_pct', 0):.1f}% dummy vectors")
        
        # Column mappings
        mapping = self.results['column_name_mapping']
        print(f"\n3. COLUMN MAPPING:")
        print(f"   Frame columns: {mapping.get('frame_column_variants', [])}")
        print(f"   Mouse ID columns: {mapping.get('mouse_id_formats', [])}")
        print(f"   Coordinate columns: {mapping.get('coordinate_patterns', [])}")
        
        # Feature quality
        quality = self.results['feature_quality']
        if 'feature_statistics' in quality:
            print(f"\n4. FEATURE QUALITY:")
            print(f"   Feature dimensions tested: {len(quality.get('feature_dimensions', []))}")
            print(f"   Zero features: {sum(quality.get('zero_feature_counts', []))}")
            print(f"   Constant features: {sum(quality.get('constant_feature_counts', []))}")
        
        print("\n" + "="*60)


def main():
    """Main diagnostic function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Feature extraction diagnostic")
    parser.add_argument('--config', default='configs/default.yaml', help='Configuration file')
    parser.add_argument('--output', default='outputs/feature_diagnostic_report.json', help='Output report path')
    
    args = parser.parse_args()
    
    # Run diagnostic
    diagnostic = FeatureDiagnostic(args.config)
    results = diagnostic.run_full_diagnostic()
    
    print(f"\nDiagnostic completed. Results saved to {args.output}")


if __name__ == "__main__":
    main()
