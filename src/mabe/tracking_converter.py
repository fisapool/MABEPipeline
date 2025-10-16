"""
Tracking Data Converter

Converts the actual MABE tracking data format to the expected format for feature extraction.
The actual format has columns: ['video_frame', 'mouse_id', 'bodypart', 'x', 'y']
The expected format has columns: ['frame', 'mouse1_body_center_x', 'mouse1_body_center_y', ...]
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TrackingDataConverter:
    """Converts MABE tracking data from actual format to expected format"""
    
    def __init__(self):
        """Initialize converter"""
        self.bodypart_mapping = {
            'body_center': 'body_center',
            'nose': 'nose',
            'neck': 'neck',
            'ear_left': 'ear_left',
            'ear_right': 'ear_right',
            'forepaw_left': 'forepaw_left',
            'forepaw_right': 'forepaw_right',
            'hindpaw_left': 'hindpaw_left',
            'hindpaw_right': 'hindpaw_right',
            'tail_base': 'tail_base',
            'tail_midpoint': 'tail_midpoint',
            'tail_tip': 'tail_tip'
        }
        
        # Priority bodyparts for feature extraction
        self.priority_bodyparts = ['body_center', 'nose', 'neck']
    
    def convert_tracking_file(self, file_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Convert a single tracking file from actual format to expected format
        
        Args:
            file_path: Path to input parquet file
            output_path: Path to save converted file (optional)
            
        Returns:
            Converted DataFrame
        """
        logger.info(f"Converting tracking file: {file_path}")
        
        try:
            # Load the actual format
            df = pd.read_parquet(file_path)
            logger.debug(f"Loaded file with shape: {df.shape}, columns: {list(df.columns)}")
            
            # Convert to expected format
            converted_df = self._convert_dataframe(df)
            
            # Save if output path provided
            if output_path:
                converted_df.to_parquet(output_path, index=False)
                logger.info(f"Saved converted file to: {output_path}")
            
            return converted_df
            
        except Exception as e:
            logger.error(f"Error converting file {file_path}: {e}")
            raise
    
    def _convert_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert DataFrame from actual format to expected format
        
        Args:
            df: DataFrame in actual format ['video_frame', 'mouse_id', 'bodypart', 'x', 'y']
            
        Returns:
            DataFrame in expected format ['frame', 'mouse1_body_center_x', ...]
        """
        logger.debug(f"Converting DataFrame with shape: {df.shape}")
        
        # Rename video_frame to frame
        if 'video_frame' in df.columns:
            df = df.rename(columns={'video_frame': 'frame'})
        
        # Get unique mice and frames
        unique_mice = sorted(df['mouse_id'].unique())
        unique_frames = sorted(df['frame'].unique())
        
        logger.debug(f"Found {len(unique_mice)} mice: {unique_mice}")
        logger.debug(f"Found {len(unique_frames)} frames: {unique_frames[0]} to {unique_frames[-1]}")
        
        # Create the expected format
        converted_data = []
        
        for frame in unique_frames:
            frame_data = {'frame': frame}
            
            # For each mouse, extract bodypart coordinates
            for mouse_id in unique_mice:
                mouse_data = df[(df['frame'] == frame) & (df['mouse_id'] == mouse_id)]
                
                if len(mouse_data) > 0:
                    # Extract bodypart coordinates
                    for _, row in mouse_data.iterrows():
                        bodypart = row['bodypart']
                        x, y = row['x'], row['y']
                        
                        # Create column names in expected format
                        col_x = f'mouse{mouse_id}_{bodypart}_x'
                        col_y = f'mouse{mouse_id}_{bodypart}_y'
                        
                        frame_data[col_x] = x
                        frame_data[col_y] = y
                
                # Ensure we have at least body_center coordinates
                body_center_data = mouse_data[mouse_data['bodypart'] == 'body_center']
                if len(body_center_data) > 0:
                    body_center = body_center_data.iloc[0]
                    frame_data[f'mouse{mouse_id}_body_center_x'] = body_center['x']
                    frame_data[f'mouse{mouse_id}_body_center_y'] = body_center['y']
                else:
                    # Use nose or neck as fallback
                    fallback_data = mouse_data[mouse_data['bodypart'].isin(['nose', 'neck'])]
                    if len(fallback_data) > 0:
                        fallback = fallback_data.iloc[0]
                        frame_data[f'mouse{mouse_id}_body_center_x'] = fallback['x']
                        frame_data[f'mouse{mouse_id}_body_center_y'] = fallback['y']
                    else:
                        # Use any available coordinate
                        if len(mouse_data) > 0:
                            any_coord = mouse_data.iloc[0]
                            frame_data[f'mouse{mouse_id}_body_center_x'] = any_coord['x']
                            frame_data[f'mouse{mouse_id}_body_center_y'] = any_coord['y']
                        else:
                            # No data for this mouse at this frame
                            frame_data[f'mouse{mouse_id}_body_center_x'] = np.nan
                            frame_data[f'mouse{mouse_id}_body_center_y'] = np.nan
            
            converted_data.append(frame_data)
        
        # Create DataFrame
        converted_df = pd.DataFrame(converted_data)
        
        # Fill missing values with forward fill, then backward fill
        converted_df = converted_df.fillna(method='ffill').fillna(method='bfill')
        
        # If still missing values, fill with 0
        converted_df = converted_df.fillna(0)
        
        logger.debug(f"Converted DataFrame shape: {converted_df.shape}")
        logger.debug(f"Converted columns: {list(converted_df.columns)}")
        
        return converted_df
    
    def convert_tracking_directory(self, input_dir: str, output_dir: str, 
                                 pattern: str = "*.parquet") -> List[str]:
        """
        Convert all tracking files in a directory
        
        Args:
            input_dir: Input directory with actual format files
            output_dir: Output directory for converted files
            pattern: File pattern to match
            
        Returns:
            List of converted file paths
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        converted_files = []
        
        # Find all parquet files
        parquet_files = list(input_path.glob(pattern))
        logger.info(f"Found {len(parquet_files)} files to convert")
        
        for file_path in parquet_files:
            try:
                output_file = output_path / f"converted_{file_path.name}"
                converted_df = self.convert_tracking_file(str(file_path), str(output_file))
                converted_files.append(str(output_file))
                logger.info(f"Converted {file_path.name} -> {output_file.name}")
                
            except Exception as e:
                logger.error(f"Error converting {file_path.name}: {e}")
                continue
        
        logger.info(f"Successfully converted {len(converted_files)} files")
        return converted_files
    
    def create_tracking_data_loader(self, tracking_dir: str) -> Dict[str, pd.DataFrame]:
        """
        Create a tracking data loader that returns data in expected format
        
        Args:
            tracking_dir: Directory containing converted tracking files
            
        Returns:
            Dictionary mapping video_id to DataFrame
        """
        tracking_path = Path(tracking_dir)
        tracking_data = {}
        
        if not tracking_path.exists():
            logger.warning(f"Tracking directory not found: {tracking_dir}")
            return tracking_data
        
        # Find all converted files
        converted_files = list(tracking_path.glob("converted_*.parquet"))
        logger.info(f"Loading {len(converted_files)} converted tracking files")
        
        for file_path in converted_files:
            try:
                # Extract video_id from filename
                video_id = file_path.stem.replace('converted_', '')
                
                # Load the converted file
                df = pd.read_parquet(file_path)
                tracking_data[video_id] = df
                
                logger.debug(f"Loaded tracking data for video {video_id}: {df.shape}")
                
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                continue
        
        logger.info(f"Loaded tracking data for {len(tracking_data)} videos")
        return tracking_data


def convert_mabe_tracking_data(dataset_path: str, output_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Convert all MABE tracking data to expected format
    
    Args:
        dataset_path: Path to MABE dataset
        output_dir: Output directory for converted files
        
    Returns:
        Dictionary mapping video_id to converted DataFrame
    """
    logger.info("Starting MABE tracking data conversion...")
    
    converter = TrackingDataConverter()
    
    # Find the keypoints directory
    keypoints_dir = Path(dataset_path) / "train_tracking" / "MABe22_keypoints"
    
    if not keypoints_dir.exists():
        logger.error(f"Keypoints directory not found: {keypoints_dir}")
        return {}
    
    # Convert all files
    converted_files = converter.convert_tracking_directory(
        str(keypoints_dir), 
        output_dir
    )
    
    # Load converted data
    tracking_data = converter.create_tracking_data_loader(output_dir)
    
    logger.info(f"Conversion completed. Loaded {len(tracking_data)} videos")
    return tracking_data


def create_sample_tracking_data() -> Dict[str, pd.DataFrame]:
    """
    Create sample tracking data for testing when real data is not available
    
    Returns:
        Dictionary with sample tracking data
    """
    logger.info("Creating sample tracking data for testing...")
    
    # Create sample data for a few videos
    sample_data = {}
    
    for video_id in ['143861384', '44566106', '209576908']:
        # Create 100 frames of sample data
        frames = list(range(100))
        data = []
        
        for frame in frames:
            frame_data = {'frame': frame}
            
            # Add coordinates for 2 mice
            for mouse_id in [1, 2]:
                # Add some movement and noise
                base_x = 100 + mouse_id * 200 + np.sin(frame * 0.1) * 50
                base_y = 100 + mouse_id * 100 + np.cos(frame * 0.1) * 30
                
                frame_data[f'mouse{mouse_id}_body_center_x'] = base_x + np.random.normal(0, 5)
                frame_data[f'mouse{mouse_id}_body_center_y'] = base_y + np.random.normal(0, 5)
            
            data.append(frame_data)
        
        sample_data[video_id] = pd.DataFrame(data)
        logger.debug(f"Created sample data for video {video_id}: {sample_data[video_id].shape}")
    
    logger.info(f"Created sample data for {len(sample_data)} videos")
    return sample_data


if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Add src to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    # Test the converter
    converter = TrackingDataConverter()
    
    # Test with a sample file
    test_file = "test_tracking.parquet"
    if Path(test_file).exists():
        converted_df = converter.convert_tracking_file(test_file)
        print(f"Converted DataFrame shape: {converted_df.shape}")
        print(f"Columns: {list(converted_df.columns)}")
        print("Sample data:")
        print(converted_df.head())
    else:
        print("No test file found. Creating sample data...")
        sample_data = create_sample_tracking_data()
        print(f"Created sample data for {len(sample_data)} videos")
