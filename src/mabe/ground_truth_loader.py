"""
Ground Truth Loader for MABE Dataset

Loads ground truth annotations from parquet files in the MABE dataset structure.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging

from .utils.logger import get_logger

logger = get_logger(__name__)


def load_ground_truth_from_parquet(dataset_path: str, video_ids: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Load ground truth data from parquet files in train_annotation directory
    
    Args:
        dataset_path: Path to MABE dataset root
        video_ids: Optional list of specific video IDs to load
        
    Returns:
        DataFrame with ground truth annotations including lab_id and behaviors_labeled
    """
    dataset_path = Path(dataset_path)
    annotation_path = dataset_path / "train_annotation"
    
    if not annotation_path.exists():
        logger.error(f"Annotation path not found: {annotation_path}")
        return pd.DataFrame()
    
    # Load train.csv metadata for lab_id and behaviors_labeled
    train_csv_path = dataset_path / "train.csv"
    if not train_csv_path.exists():
        logger.error(f"Train CSV not found: {train_csv_path}")
        return pd.DataFrame()
    
    logger.info("Loading train.csv metadata...")
    train_metadata = pd.read_csv(train_csv_path)
    logger.info(f"Loaded metadata for {len(train_metadata)} videos")
    
    all_annotations = []
    
    # Get all lab directories
    for lab_dir in annotation_path.iterdir():
        if lab_dir.is_dir():
            logger.info(f"Loading annotations from {lab_dir.name}")
            
            # Get all parquet files in this lab
            for parquet_file in lab_dir.glob("*.parquet"):
                video_id = parquet_file.stem
                
                # Skip if specific video IDs requested and this one not included
                if video_ids and video_id not in video_ids:
                    continue
                
                try:
                    df = pd.read_parquet(parquet_file)
                    
                    # Add video_id column if not present
                    if 'video_id' not in df.columns:
                        df['video_id'] = video_id
                    
                    # Get metadata for this video
                    video_metadata = train_metadata[train_metadata['video_id'] == int(video_id)]
                    if len(video_metadata) == 0:
                        logger.warning(f"No metadata found for video {video_id}")
                        continue
                    
                    # Add lab_id and behaviors_labeled from train.csv
                    df['lab_id'] = video_metadata.iloc[0]['lab_id']
                    df['behaviors_labeled'] = video_metadata.iloc[0]['behaviors_labeled']
                    
                    all_annotations.append(df)
                    logger.info(f"Loaded {len(df)} annotations from {video_id} (lab: {df['lab_id'].iloc[0]})")
                    
                except Exception as e:
                    logger.warning(f"Failed to load {parquet_file}: {e}")
    
    if not all_annotations:
        logger.warning("No ground truth annotations found")
        return pd.DataFrame()
    
    # Combine all annotations
    ground_truth = pd.concat(all_annotations, ignore_index=True)
    
    # Standardize column names to match evaluation format
    ground_truth = standardize_ground_truth_format(ground_truth)
    
    logger.info(f"Loaded {len(ground_truth)} total ground truth annotations")
    return ground_truth


def standardize_ground_truth_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize ground truth DataFrame to match evaluation format
    
    Args:
        df: Raw ground truth DataFrame
        
    Returns:
        Standardized DataFrame
    """
    # Create a copy to avoid modifying original
    df = df.copy()
    
    # Ensure required columns exist with proper names
    required_columns = ['video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame']
    kaggle_columns = ['lab_id', 'behaviors_labeled']  # Additional columns for Kaggle metric
    
    # Map common column name variations
    column_mapping = {}
    for col in required_columns:
        if col not in df.columns:
            # Try alternative names
            if col == 'agent_id' and 'agent' in df.columns:
                column_mapping[col] = 'agent'
            elif col == 'target_id' and 'target' in df.columns:
                column_mapping[col] = 'target'
            elif col == 'action' and 'behavior' in df.columns:
                column_mapping[col] = 'behavior'
            elif col == 'start_frame' and 'start' in df.columns:
                column_mapping[col] = 'start'
            elif col == 'stop_frame' and 'stop' in df.columns:
                column_mapping[col] = 'stop'
    
    # Apply column mapping
    if column_mapping:
        df = df.rename(columns=column_mapping)
    
    # Ensure all required columns exist
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        logger.info(f"Available columns: {df.columns.tolist()}")
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check for Kaggle-specific columns
    missing_kaggle_columns = [col for col in kaggle_columns if col not in df.columns]
    if missing_kaggle_columns:
        logger.warning(f"Missing Kaggle metric columns: {missing_kaggle_columns}")
        logger.info("Kaggle metric evaluation will not be available")
    
    # Convert to appropriate types
    df['video_id'] = df['video_id'].astype(str)
    df['agent_id'] = df['agent_id'].astype(str)
    df['target_id'] = df['target_id'].astype(str)
    df['action'] = df['action'].astype(str)
    df['start_frame'] = df['start_frame'].astype(int)
    df['stop_frame'] = df['stop_frame'].astype(int)
    
    # Convert Kaggle columns if present
    if 'lab_id' in df.columns:
        df['lab_id'] = df['lab_id'].astype(str)
    if 'behaviors_labeled' in df.columns:
        df['behaviors_labeled'] = df['behaviors_labeled'].astype(str)
    
    return df


def load_ground_truth_for_evaluation(cfg: Dict) -> pd.DataFrame:
    """
    Load ground truth data using configuration settings
    
    Args:
        cfg: Configuration dictionary
        
    Returns:
        Ground truth DataFrame
    """
    # Handle different config structures
    if 'dataset' in cfg and 'path' in cfg['dataset']:
        dataset_path = cfg['dataset']['path']
    elif 'paths' in cfg and 'dataset_path' in cfg['paths']:
        dataset_path = cfg['paths']['dataset_path']
    else:
        raise ValueError("Dataset path not found in configuration")
    
    max_videos = cfg.get('data', {}).get('max_videos', None)
    if max_videos is None:
        max_videos = cfg.get('training', {}).get('max_videos', None)
    
    # Get video IDs if max_videos is specified
    video_ids = None
    if max_videos:
        # Load train.csv to get video IDs
        train_csv_path = Path(dataset_path) / "train.csv"
        if train_csv_path.exists():
            train_df = pd.read_csv(train_csv_path)
            video_ids = train_df['video_id'].head(max_videos).astype(str).tolist()
            logger.info(f"Limiting to {max_videos} videos: {video_ids}")
    
    return load_ground_truth_from_parquet(dataset_path, video_ids)


def get_available_video_ids(dataset_path: str) -> List[str]:
    """
    Get list of available video IDs from the dataset
    
    Args:
        dataset_path: Path to MABE dataset root
        
    Returns:
        List of video IDs
    """
    dataset_path = Path(dataset_path)
    annotation_path = dataset_path / "train_annotation"
    
    video_ids = []
    for lab_dir in annotation_path.iterdir():
        if lab_dir.is_dir():
            for parquet_file in lab_dir.glob("*.parquet"):
                video_ids.append(parquet_file.stem)
    
    return sorted(video_ids)
