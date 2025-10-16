#!/usr/bin/env python3
"""
Tracking Data Inspector

Inspects raw parquet files to verify column names, formats, and data quality.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mabe.utils import load_config, get_logger

logger = get_logger(__name__)


class TrackingDataInspector:
    """Inspector for raw tracking data format and quality"""
    
    def __init__(self, cfg_path: str = "configs/default.yaml"):
        """Initialize inspector with configuration"""
        self.cfg = load_config(cfg_path)
        self.results = {
            'tracking_files_found': [],
            'column_analysis': {},
            'data_quality': {},
            'format_issues': [],
            'recommendations': []
        }
        
    def inspect_all_tracking_data(self) -> Dict[str, Any]:
        """Inspect all available tracking data"""
        logger.info("Starting tracking data inspection...")
        
        # Find tracking data directory
        dataset_path = self.cfg['dataset']['path']
        train_tracking_dir = Path(dataset_path) / self.cfg['dataset']['train_tracking_dir']
        
        if not train_tracking_dir.exists():
            logger.error(f"Tracking directory not found: {train_tracking_dir}")
            return self.results
        
        logger.info(f"Scanning tracking directory: {train_tracking_dir}")
        
        # Find all parquet files
        parquet_files = list(train_tracking_dir.glob("**/*.parquet"))
        logger.info(f"Found {len(parquet_files)} parquet files")
        
        if not parquet_files:
            logger.warning("No parquet files found in tracking directory")
            return self.results
        
        # Inspect sample files
        sample_files = parquet_files[:5]  # Inspect first 5 files
        for file_path in sample_files:
            self._inspect_single_file(file_path)
        
        # Generate recommendations
        self._generate_recommendations()
        
        # Save results
        self._save_results()
        
        return self.results
    
    def _inspect_single_file(self, file_path: Path):
        """Inspect a single tracking file"""
        logger.info(f"Inspecting file: {file_path.name}")
        
        try:
            # Load parquet file
            df = pd.read_parquet(file_path)
            
            file_info = {
                'file_path': str(file_path),
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'sample_data': df.head(3).to_dict(),
                'frame_range': None,
                'coordinate_columns': [],
                'mouse_id_columns': [],
                'issues': []
            }
            
            # Analyze frame column
            if 'frame' in df.columns:
                file_info['frame_range'] = {
                    'min': int(df['frame'].min()),
                    'max': int(df['frame'].max()),
                    'count': len(df)
                }
            elif 'Frame' in df.columns:
                file_info['frame_range'] = {
                    'min': int(df['Frame'].min()),
                    'max': int(df['Frame'].max()),
                    'count': len(df)
                }
                file_info['issues'].append("Uses 'Frame' instead of 'frame'")
            else:
                file_info['issues'].append("No frame column found")
            
            # Analyze coordinate columns
            coord_patterns = ['_body_center_x', '_body_center_y', '_x', '_y']
            for col in df.columns:
                if any(pattern in col.lower() for pattern in coord_patterns):
                    file_info['coordinate_columns'].append(col)
            
            # Analyze mouse ID patterns
            mouse_patterns = ['mouse', 'agent', 'id']
            for col in df.columns:
                if any(pattern in col.lower() for pattern in mouse_patterns):
                    file_info['mouse_id_columns'].append(col)
            
            # Check for expected columns
            expected_columns = [
                'frame', 'mouse1_body_center_x', 'mouse1_body_center_y',
                'mouse2_body_center_x', 'mouse2_body_center_y'
            ]
            missing_columns = [col for col in expected_columns if col not in df.columns]
            if missing_columns:
                file_info['issues'].append(f"Missing expected columns: {missing_columns}")
            
            # Check data quality
            quality_issues = []
            
            # Check for NaN values
            nan_counts = df.isnull().sum()
            if nan_counts.any():
                quality_issues.append(f"NaN values found: {nan_counts[nan_counts > 0].to_dict()}")
            
            # Check for extreme values
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col != 'frame':  # Skip frame column
                    col_data = df[col].dropna()
                    if len(col_data) > 0:
                        if col_data.abs().max() > 10000:
                            quality_issues.append(f"Extreme values in {col}: max={col_data.abs().max()}")
            
            file_info['quality_issues'] = quality_issues
            
            self.results['tracking_files_found'].append(file_info)
            
        except Exception as e:
            logger.error(f"Error inspecting {file_path}: {e}")
            self.results['format_issues'].append(f"Error reading {file_path.name}: {e}")
    
    def _generate_recommendations(self):
        """Generate recommendations based on inspection results"""
        if not self.results['tracking_files_found']:
            return
        
        # Analyze common patterns
        all_columns = set()
        frame_column_variants = set()
        coordinate_patterns = set()
        
        for file_info in self.results['tracking_files_found']:
            all_columns.update(file_info['columns'])
            
            if 'frame' in file_info['columns']:
                frame_column_variants.add('frame')
            if 'Frame' in file_info['columns']:
                frame_column_variants.add('Frame')
            
            coordinate_patterns.update(file_info['coordinate_columns'])
        
        # Generate recommendations
        recommendations = []
        
        # Frame column recommendations
        if 'Frame' in frame_column_variants and 'frame' not in frame_column_variants:
            recommendations.append("Standardize frame column name: use 'frame' instead of 'Frame'")
        
        # Coordinate column recommendations
        expected_coords = [
            'mouse1_body_center_x', 'mouse1_body_center_y',
            'mouse2_body_center_x', 'mouse2_body_center_y'
        ]
        missing_coords = [coord for coord in expected_coords if coord not in all_columns]
        if missing_coords:
            recommendations.append(f"Missing expected coordinate columns: {missing_coords}")
        
        # Data quality recommendations
        quality_issues = []
        for file_info in self.results['tracking_files_found']:
            quality_issues.extend(file_info.get('quality_issues', []))
        
        if quality_issues:
            recommendations.append("Address data quality issues: " + "; ".join(quality_issues[:3]))
        
        # Feature extraction recommendations
        if any('issues' in file_info and file_info['issues'] for file_info in self.results['tracking_files_found']):
            recommendations.append("Update feature extraction code to handle column name variations")
        
        self.results['recommendations'] = recommendations
    
    def _save_results(self):
        """Save inspection results"""
        import json
        from datetime import datetime
        
        output_path = Path("outputs/tracking_data_inspection.json")
        output_path.parent.mkdir(exist_ok=True)
        
        # Add timestamp
        self.results['inspection_timestamp'] = datetime.now().isoformat()
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"Inspection results saved to {output_path}")
        
        # Print summary
        self._print_summary()
    
    def _print_summary(self):
        """Print inspection summary"""
        print("\n" + "="*60)
        print("TRACKING DATA INSPECTION SUMMARY")
        print("="*60)
        
        files_found = len(self.results['tracking_files_found'])
        print(f"\n1. FILES INSPECTED: {files_found}")
        
        if files_found > 0:
            # Column analysis
            all_columns = set()
            for file_info in self.results['tracking_files_found']:
                all_columns.update(file_info['columns'])
            
            print(f"\n2. COLUMN ANALYSIS:")
            print(f"   Total unique columns found: {len(all_columns)}")
            print(f"   Common columns: {sorted(list(all_columns))[:10]}")
            
            # Frame column analysis
            frame_variants = set()
            for file_info in self.results['tracking_files_found']:
                if 'frame' in file_info['columns']:
                    frame_variants.add('frame')
                if 'Frame' in file_info['columns']:
                    frame_variants.add('Frame')
            
            print(f"\n3. FRAME COLUMN VARIATIONS:")
            print(f"   Found: {list(frame_variants)}")
            
            # Issues found
            total_issues = sum(len(file_info.get('issues', [])) for file_info in self.results['tracking_files_found'])
            print(f"\n4. ISSUES FOUND: {total_issues}")
            
            if total_issues > 0:
                for file_info in self.results['tracking_files_found']:
                    if file_info.get('issues'):
                        print(f"   {Path(file_info['file_path']).name}: {file_info['issues']}")
        
        # Recommendations
        recommendations = self.results.get('recommendations', [])
        print(f"\n5. RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
        
        print("\n" + "="*60)


def main():
    """Main inspection function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Inspect tracking data format")
    parser.add_argument('--config', default='configs/default.yaml', help='Configuration file')
    parser.add_argument('--output', default='outputs/tracking_data_inspection.json', help='Output report path')
    
    args = parser.parse_args()
    
    # Run inspection
    inspector = TrackingDataInspector(args.config)
    results = inspector.inspect_all_tracking_data()
    
    print(f"\nInspection completed. Results saved to {args.output}")


if __name__ == "__main__":
    main()
