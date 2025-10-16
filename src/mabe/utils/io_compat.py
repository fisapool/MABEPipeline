"""
Legacy file compatibility for MABE Pipeline

Handles reading and converting legacy file formats to standardized formats.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import warnings

logger = logging.getLogger(__name__)


def read_legacy_frame_labels(path: str) -> pd.DataFrame:
    """
    Read legacy mabe_frame_labels_*.csv files
    
    Args:
        path: Path to legacy CSV file
        
    Returns:
        DataFrame with standardized column names
    """
    try:
        df = pd.read_csv(path)
        logger.info(f"Loaded legacy frame labels from {path}: {df.shape}")
        
        # Check for required columns
        required_columns = ['video_id', 'frame', 'agent_id', 'target_id', 'behavior']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.warning(f"Missing columns in legacy file: {missing_columns}")
            # Try to infer from available columns
            df = _infer_missing_columns(df, missing_columns)
        
        # Standardize column names
        df = _standardize_column_names(df)
        
        # Validate data types
        df = _validate_data_types(df)
        
        logger.info(f"Successfully loaded and standardized {len(df)} frame labels")
        return df
        
    except Exception as e:
        logger.error(f"Error reading legacy frame labels from {path}: {e}")
        raise


def convert_to_standard_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert DataFrame to standardized format
    
    Args:
        df: Input DataFrame
        
    Returns:
        Standardized DataFrame
    """
    # Make a copy to avoid modifying original
    df_std = df.copy()
    
    # Standardize column names
    df_std = _standardize_column_names(df_std)
    
    # Ensure required columns exist
    df_std = _ensure_required_columns(df_std)
    
    # Validate and convert data types
    df_std = _validate_data_types(df_std)
    
    # Sort by video_id, frame for consistency
    if 'video_id' in df_std.columns and 'frame' in df_std.columns:
        df_std = df_std.sort_values(['video_id', 'frame']).reset_index(drop=True)
    
    logger.info(f"Converted to standard format: {df_std.shape}")
    return df_std


def _infer_missing_columns(df: pd.DataFrame, missing_columns: List[str]) -> pd.DataFrame:
    """
    Try to infer missing columns from available data
    
    Args:
        df: Input DataFrame
        missing_columns: List of missing column names
        
    Returns:
        DataFrame with inferred columns
    """
    for col in missing_columns:
        if col == 'video_id':
            # Look for video-related columns
            video_cols = [c for c in df.columns if 'video' in c.lower()]
            if video_cols:
                df[col] = df[video_cols[0]]
                logger.info(f"Inferred {col} from {video_cols[0]}")
            else:
                df[col] = 'unknown_video'
                logger.warning(f"Could not infer {col}, using default value")
        
        elif col == 'frame':
            # Look for frame-related columns
            frame_cols = [c for c in df.columns if 'frame' in c.lower()]
            if frame_cols:
                df[col] = df[frame_cols[0]]
                logger.info(f"Inferred {col} from {frame_cols[0]}")
            else:
                df[col] = 0
                logger.warning(f"Could not infer {col}, using default value")
        
        elif col == 'agent_id':
            # Look for agent-related columns
            agent_cols = [c for c in df.columns if 'agent' in c.lower()]
            if agent_cols:
                df[col] = df[agent_cols[0]]
                logger.info(f"Inferred {col} from {agent_cols[0]}")
            else:
                df[col] = 'agent1'
                logger.warning(f"Could not infer {col}, using default value")
        
        elif col == 'target_id':
            # Look for target-related columns
            target_cols = [c for c in df.columns if 'target' in c.lower()]
            if target_cols:
                df[col] = df[target_cols[0]]
                logger.info(f"Inferred {col} from {target_cols[0]}")
            else:
                df[col] = 'target1'
                logger.warning(f"Could not infer {col}, using default value")
        
        elif col == 'behavior':
            # Look for behavior/action-related columns
            behavior_cols = [c for c in df.columns if any(x in c.lower() for x in ['behavior', 'action', 'label'])]
            if behavior_cols:
                df[col] = df[behavior_cols[0]]
                logger.info(f"Inferred {col} from {behavior_cols[0]}")
            else:
                df[col] = 'unknown'
                logger.warning(f"Could not infer {col}, using default value")
    
    return df


def _standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names to expected format
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with standardized column names
    """
    # Column name mappings
    column_mappings = {
        'video_id': ['video_id', 'video', 'Video', 'VIDEO_ID'],
        'frame': ['frame', 'Frame', 'FRAME', 'frame_number'],
        'agent_id': ['agent_id', 'agent', 'Agent', 'AGENT_ID', 'mouse_id'],
        'target_id': ['target_id', 'target', 'Target', 'TARGET_ID'],
        'behavior': ['behavior', 'Behaviour', 'action', 'Action', 'label', 'Label']
    }
    
    df_std = df.copy()
    
    for standard_name, possible_names in column_mappings.items():
        for possible_name in possible_names:
            if possible_name in df_std.columns and standard_name not in df_std.columns:
                df_std = df_std.rename(columns={possible_name: standard_name})
                logger.debug(f"Renamed column {possible_name} to {standard_name}")
                break
    
    return df_std


def _ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure all required columns exist with default values if missing
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with all required columns
    """
    required_columns = {
        'video_id': 'unknown_video',
        'frame': 0,
        'agent_id': 'agent1',
        'target_id': 'target1',
        'behavior': 'unknown'
    }
    
    df_std = df.copy()
    
    for col, default_value in required_columns.items():
        if col not in df_std.columns:
            df_std[col] = default_value
            logger.warning(f"Added missing column {col} with default value {default_value}")
    
    return df_std


def _validate_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and convert data types to expected formats
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with validated data types
    """
    df_std = df.copy()
    
    # Convert video_id to string
    if 'video_id' in df_std.columns:
        df_std['video_id'] = df_std['video_id'].astype(str)
    
    # Convert frame to integer
    if 'frame' in df_std.columns:
        df_std['frame'] = pd.to_numeric(df_std['frame'], errors='coerce').fillna(0).astype(int)
    
    # Convert agent_id and target_id to string
    for col in ['agent_id', 'target_id']:
        if col in df_std.columns:
            df_std[col] = df_std[col].astype(str)
    
    # Convert behavior to string
    if 'behavior' in df_std.columns:
        df_std['behavior'] = df_std['behavior'].astype(str)
    
    return df_std


def detect_legacy_format(file_path: str) -> bool:
    """
    Detect if a file is in legacy format
    
    Args:
        file_path: Path to file to check
        
    Returns:
        True if file appears to be in legacy format
    """
    try:
        # Check if it's a CSV file
        if not file_path.lower().endswith('.csv'):
            return False
        
        # Check filename pattern
        filename = Path(file_path).name
        if 'mabe_frame_labels' in filename:
            return True
        
        # Check file contents
        df = pd.read_csv(file_path, nrows=5)  # Read first 5 rows
        
        # Check for legacy column patterns
        legacy_indicators = [
            'video_id' in df.columns,
            'frame' in df.columns,
            'agent_id' in df.columns,
            'target_id' in df.columns,
            'behavior' in df.columns
        ]
        
        # If most indicators are present, likely legacy format
        return sum(legacy_indicators) >= 3
        
    except Exception:
        return False


def migrate_legacy_files(source_dir: str, target_dir: str, 
                        pattern: str = "mabe_frame_labels_*.csv") -> List[str]:
    """
    Migrate all legacy files from source to target directory
    
    Args:
        source_dir: Source directory containing legacy files
        target_dir: Target directory for migrated files
        pattern: File pattern to match
        
    Returns:
        List of migrated file paths
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    migrated_files = []
    
    # Find all legacy files
    legacy_files = list(source_path.glob(pattern))
    logger.info(f"Found {len(legacy_files)} legacy files to migrate")
    
    for legacy_file in legacy_files:
        try:
            # Read and convert legacy file
            df = read_legacy_frame_labels(str(legacy_file))
            df_std = convert_to_standard_format(df)
            
            # Save as parquet for better performance
            target_file = target_path / f"{legacy_file.stem}.parquet"
            df_std.to_parquet(target_file, index=False)
            
            migrated_files.append(str(target_file))
            logger.info(f"Migrated {legacy_file.name} to {target_file.name}")
            
        except Exception as e:
            logger.error(f"Error migrating {legacy_file}: {e}")
            continue
    
    logger.info(f"Successfully migrated {len(migrated_files)} files")
    return migrated_files


def create_migration_report(source_dir: str, target_dir: str) -> Dict[str, Any]:
    """
    Create a migration report
    
    Args:
        source_dir: Source directory
        target_dir: Target directory
        
    Returns:
        Migration report dictionary
    """
    import json
    from datetime import datetime
    
    report = {
        'migration_timestamp': datetime.now().isoformat(),
        'source_directory': source_dir,
        'target_directory': target_dir,
        'migrated_files': [],
        'errors': []
    }
    
    try:
        migrated_files = migrate_legacy_files(source_dir, target_dir)
        report['migrated_files'] = migrated_files
        report['total_migrated'] = len(migrated_files)
        
    except Exception as e:
        report['errors'].append(str(e))
    
    return report
