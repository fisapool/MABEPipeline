#!/usr/bin/env python3
"""
MABE Frame Labels Migration Script

One-time migration script to convert legacy mabe_frame_labels_*.csv files
to standardized format for the new pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import logging
import argparse
import sys
from typing import List, Dict, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mabe.utils.logger import get_logger
from mabe.utils.io_compat import find_latest_legacy_frame_labels, load_legacy_frame_labels, convert_to_standard_format

logger = get_logger(__name__)


def find_legacy_files(search_dir: Path) -> List[Path]:
    """
    Find all legacy mabe_frame_labels_*.csv files in directory
    
    Args:
        search_dir: Directory to search
        
    Returns:
        List of legacy file paths
    """
    legacy_files = list(search_dir.glob("mabe_frame_labels_*.csv"))
    return sorted(legacy_files, key=lambda x: x.stat().st_mtime, reverse=True)


def validate_legacy_file(file_path: Path) -> bool:
    """
    Validate legacy frame labels file
    
    Args:
        file_path: Path to legacy file
        
    Returns:
        True if file is valid, False otherwise
    """
    try:
        df = pd.read_csv(file_path)
        
        # Check required columns
        required_columns = ['video_id', 'frame', 'agent_id', 'target_id', 'behavior']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.warning(f"Missing columns in {file_path}: {missing_columns}")
            return False
        
        # Check data types
        if not pd.api.types.is_numeric_dtype(df['video_id']):
            logger.warning(f"video_id should be numeric in {file_path}")
            return False
        
        if not pd.api.types.is_numeric_dtype(df['frame']):
            logger.warning(f"frame should be numeric in {file_path}")
            return False
        
        # Check for empty file
        if len(df) == 0:
            logger.warning(f"Empty file: {file_path}")
            return False
        
        logger.info(f"Valid legacy file: {file_path} ({len(df)} rows)")
        return True
        
    except Exception as e:
        logger.error(f"Error validating {file_path}: {e}")
        return False


def convert_legacy_file(file_path: Path, output_dir: Path) -> Optional[Path]:
    """
    Convert legacy frame labels file to standardized format
    
    Args:
        file_path: Path to legacy file
        output_dir: Output directory for converted file
        
    Returns:
        Path to converted file, or None if conversion failed
    """
    try:
        # Load legacy file
        legacy_df = load_legacy_frame_labels(file_path)
        
        # Convert to standard format
        standard_df = convert_to_standard_format(legacy_df)
        
        # Create output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"frame_labels_standardized_{timestamp}.csv"
        output_path = output_dir / output_filename
        
        # Save standardized file
        standard_df.to_csv(output_path, index=False)
        
        logger.info(f"Converted {file_path} -> {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error converting {file_path}: {e}")
        return None


def create_migration_report(legacy_files: List[Path], converted_files: List[Path], 
                          output_dir: Path) -> Path:
    """
    Create migration report
    
    Args:
        legacy_files: List of legacy files found
        converted_files: List of successfully converted files
        output_dir: Output directory for report
        
    Returns:
        Path to migration report
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"migration_report_{timestamp}.json"
    
    report = {
        'migration_timestamp': timestamp,
        'legacy_files_found': len(legacy_files),
        'files_successfully_converted': len(converted_files),
        'conversion_success_rate': len(converted_files) / len(legacy_files) if legacy_files else 0,
        'legacy_files': [str(f) for f in legacy_files],
        'converted_files': [str(f) for f in converted_files],
        'failed_conversions': len(legacy_files) - len(converted_files)
    }
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Migration report saved to {report_path}")
    return report_path


def migrate_frame_labels(source_dir: Path, output_dir: Path, 
                       validate_only: bool = False) -> Dict:
    """
    Migrate legacy frame labels files to standardized format
    
    Args:
        source_dir: Directory containing legacy files
        output_dir: Directory to save converted files
        validate_only: If True, only validate files without converting
        
    Returns:
        Dictionary with migration results
    """
    logger.info(f"Starting frame labels migration from {source_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find legacy files
    legacy_files = find_legacy_files(source_dir)
    
    if not legacy_files:
        logger.warning(f"No legacy files found in {source_dir}")
        return {
            'legacy_files_found': 0,
            'files_converted': 0,
            'conversion_success_rate': 0.0
        }
    
    logger.info(f"Found {len(legacy_files)} legacy files")
    
    # Validate files
    valid_files = []
    for file_path in legacy_files:
        if validate_legacy_file(file_path):
            valid_files.append(file_path)
    
    logger.info(f"Validated {len(valid_files)} files")
    
    if validate_only:
        logger.info("Validation only mode - no files converted")
        return {
            'legacy_files_found': len(legacy_files),
            'valid_files': len(valid_files),
            'files_converted': 0,
            'conversion_success_rate': 0.0
        }
    
    # Convert files
    converted_files = []
    for file_path in valid_files:
        converted_path = convert_legacy_file(file_path, output_dir)
        if converted_path:
            converted_files.append(converted_path)
    
    # Create migration report
    report_path = create_migration_report(legacy_files, converted_files, output_dir)
    
    # Summary
    success_rate = len(converted_files) / len(legacy_files) if legacy_files else 0.0
    
    logger.info(f"Migration completed:")
    logger.info(f"  Legacy files found: {len(legacy_files)}")
    logger.info(f"  Files converted: {len(converted_files)}")
    logger.info(f"  Success rate: {success_rate:.2%}")
    logger.info(f"  Report saved to: {report_path}")
    
    return {
        'legacy_files_found': len(legacy_files),
        'valid_files': len(valid_files),
        'files_converted': len(converted_files),
        'conversion_success_rate': success_rate,
        'report_path': str(report_path)
    }


def main():
    """Main entry point for migration script"""
    parser = argparse.ArgumentParser(
        description="Migrate legacy MABE frame labels files to standardized format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Migrate files from MABEModel directory
  python scripts/migrate_frame_labels.py --source C:/Users/MYPC/Documents/MABEModel --output data/

  # Validate files without converting
  python scripts/migrate_frame_labels.py --source C:/Users/MYPC/Documents/MABEModel --validate-only

  # Migrate with custom output directory
  python scripts/migrate_frame_labels.py --source C:/Users/MYPC/Documents/MABEModel --output outputs/migrated_data/
        """
    )
    
    parser.add_argument('--source', type=str, required=True,
                       help='Source directory containing legacy files')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for converted files')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate files without converting')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Convert paths
    source_dir = Path(args.source)
    output_dir = Path(args.output)
    
    # Validate source directory
    if not source_dir.exists():
        logger.error(f"Source directory does not exist: {source_dir}")
        sys.exit(1)
    
    if not source_dir.is_dir():
        logger.error(f"Source path is not a directory: {source_dir}")
        sys.exit(1)
    
    # Run migration
    try:
        results = migrate_frame_labels(source_dir, output_dir, args.validate_only)
        
        # Print summary
        print(f"\nMigration Summary:")
        print(f"  Legacy files found: {results['legacy_files_found']}")
        print(f"  Files converted: {results['files_converted']}")
        print(f"  Success rate: {results['conversion_success_rate']:.2%}")
        
        if results['conversion_success_rate'] < 1.0:
            print(f"  Warning: {results['legacy_files_found'] - results['files_converted']} files failed to convert")
            sys.exit(1)
        else:
            print(f"  Success: All files converted successfully!")
            
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
