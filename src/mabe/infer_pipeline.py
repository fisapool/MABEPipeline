"""
MABE Inference Pipeline Orchestrator

High-level inference orchestrator that coordinates model loading, data preparation,
prediction generation, and submission creation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging
import torch

# Import utilities
from .utils.logger import get_logger
from .utils.seed import set_seed
from .utils.config import load_config

# Import core modules
from .inference import (
    TestMouseBehaviorDataset, load_model, predict_single, 
    ensemble_predictions, create_kaggle_submission
)
from .preprocessing import MABEDataPreprocessor

logger = get_logger(__name__)


def run_inference(cfg: Dict, test_csv: Optional[str] = None, output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Run the complete inference pipeline
    
    Args:
        cfg: Configuration dictionary
        test_csv: Optional path to test CSV file
        output_path: Optional path for output submission
        
    Returns:
        DataFrame with submission predictions
    """
    logger.info("Starting MABE inference pipeline...")
    
    # Set random seed for reproducibility
    set_seed(cfg.get('seed', 42))
    
    # Step 1: Load test data
    logger.info("Step 1/5: Loading test data")
    test_df = load_test_data(cfg, test_csv)
    
    if test_df.empty:
        logger.error("No test data loaded. Check data paths and configuration.")
        return pd.DataFrame()
    
    # Step 2: Load tracking data
    logger.info("Step 2/5: Loading tracking data")
    tracking_data = load_tracking_data(cfg, test_df)
    
    # Step 3: Load models
    logger.info("Step 3/5: Loading trained models")
    models = load_models(cfg)
    
    if not models:
        logger.error("No models loaded. Check model paths and configuration.")
        return pd.DataFrame()
    
    # Step 4: Generate predictions
    logger.info("Step 4/5: Generating predictions")
    predictions = generate_predictions(cfg, test_df, tracking_data, models)
    
    if not predictions:
        logger.error("No predictions generated.")
        return pd.DataFrame()
    
    # Step 5: Create submission
    logger.info("Step 5/5: Creating submission file")
    submission_path, submission_df = create_kaggle_submission(predictions, cfg)
    
    if output_path:
        # Copy to specified output path
        import shutil
        shutil.copy2(submission_path, output_path)
        logger.info(f"Submission copied to {output_path}")
    
    logger.info("Inference pipeline completed successfully!")
    logger.info(f"Submission saved to: {submission_path}")
    logger.info(f"Total predictions: {len(submission_df)}")
    
    return submission_df


def load_test_data(cfg: Dict, test_csv: Optional[str] = None) -> pd.DataFrame:
    """
    Load test data from configuration or specified path
    
    Args:
        cfg: Configuration dictionary
        test_csv: Optional path to test CSV file
        
    Returns:
        Test data DataFrame
    """
    if test_csv:
        test_path = Path(test_csv)
    else:
        dataset_path = Path(cfg.get('dataset', {}).get('path', ''))
        test_csv_name = cfg.get('dataset', {}).get('test_csv', 'test.csv')
        test_path = dataset_path / test_csv_name
    
    if not test_path.exists():
        logger.error(f"Test CSV not found: {test_path}")
        return pd.DataFrame()
    
    logger.info(f"Loading test data from {test_path}")
    test_df = pd.read_csv(test_path)
    
    logger.info(f"Loaded {len(test_df)} test videos")
    return test_df


def load_tracking_data(cfg: Dict, test_df: pd.DataFrame) -> Dict:
    """
    Load tracking data for test videos
    
    Args:
        cfg: Configuration dictionary
        test_df: Test data DataFrame
        
    Returns:
        Dictionary with tracking data
    """
    dataset_path = Path(cfg.get('dataset', {}).get('path', ''))
    
    tracking_data = {}
    
    # Try to load converted tracking data first
    converted_tracking_dir = dataset_path / "converted_tracking"
    if converted_tracking_dir.exists():
        logger.info("Loading converted tracking data...")
        from .tracking_converter import TrackingDataConverter
        converter = TrackingDataConverter()
        tracking_data = converter.create_tracking_data_loader(str(converted_tracking_dir))
        
        # Filter to only include videos we need
        video_ids = test_df['video_id'].unique()
        tracking_data = {k: v for k, v in tracking_data.items() if k in video_ids}
        logger.info(f"Loaded converted tracking data for {len(tracking_data)} videos")
        return tracking_data
    
    # If no converted data, try to convert on the fly
    keypoints_dir = dataset_path / "train_tracking" / "MABe22_keypoints"
    if keypoints_dir.exists():
        logger.info("Converting tracking data on the fly...")
        from .tracking_converter import TrackingDataConverter
        converter = TrackingDataConverter()
        
        for _, video_row in test_df.iterrows():
            video_id = video_row['video_id']
            tracking_file = keypoints_dir / f"{video_id}.parquet"
            
            if tracking_file.exists():
                try:
                    # Convert the file
                    converted_df = converter.convert_tracking_file(str(tracking_file))
                    tracking_data[video_id] = converted_df
                    logger.info(f"Converted and loaded tracking data for video {video_id}: {converted_df.shape}")
                except Exception as e:
                    logger.warning(f"Error converting tracking data for video {video_id}: {e}")
            else:
                logger.warning(f"Tracking file not found for video {video_id}: {tracking_file}")
    else:
        # Fallback: create sample data for testing
        logger.warning("No tracking data found. Creating sample data for testing...")
        from .tracking_converter import create_sample_tracking_data
        sample_data = create_sample_tracking_data()
        video_ids = test_df['video_id'].unique()
        tracking_data = {k: v for k, v in sample_data.items() if k in video_ids}
        logger.info(f"Using sample tracking data for {len(tracking_data)} videos")
    
    logger.info(f"Loaded tracking data for {len(tracking_data)} videos")
    return tracking_data


def load_models(cfg: Dict) -> Dict:
    """
    Load trained models from configuration
    
    Args:
        cfg: Configuration dictionary
        
    Returns:
        Dictionary with loaded models
    """
    models = {}
    models_dir = Path(cfg.get('paths', {}).get('models_dir', 'outputs/models'))
    device = torch.device(cfg.get('device', {}).get('device_str', 'cuda:0'))
    
    # Load CNN model if available
    cnn_model_path = models_dir / cfg.get('paths', {}).get('cnn_model_filename', 'cnn_enhanced_model.pth')
    if cnn_model_path.exists():
        try:
            cnn_model = load_model(str(cnn_model_path), 'cnn', input_dim=26)
            cnn_model = cnn_model.to(device)
            cnn_model.eval()
            models['cnn'] = cnn_model
            logger.info(f"Loaded CNN model from {cnn_model_path}")
        except Exception as e:
            logger.warning(f"Error loading CNN model: {e}")
    
    # Load LSTM model if available
    lstm_model_path = models_dir / cfg.get('paths', {}).get('lstm_model_filename', 'lstm_enhanced_model.pth')
    if lstm_model_path.exists():
        try:
            lstm_model = load_model(str(lstm_model_path), 'lstm', input_dim=26)
            lstm_model = lstm_model.to(device)
            lstm_model.eval()
            models['lstm'] = lstm_model
            logger.info(f"Loaded LSTM model from {lstm_model_path}")
        except Exception as e:
            logger.warning(f"Error loading LSTM model: {e}")
    
    if not models:
        logger.error("No models could be loaded. Check model paths and files.")
    
    return models


def generate_predictions(cfg: Dict, test_df: pd.DataFrame, tracking_data: Dict, models: Dict) -> List[Dict]:
    """
    Generate predictions using loaded models
    
    Args:
        cfg: Configuration dictionary
        test_df: Test data DataFrame
        tracking_data: Tracking data dictionary
        models: Loaded models dictionary
        
    Returns:
        List of prediction dictionaries
    """
    device = torch.device(cfg.get('device', {}).get('device_str', 'cuda:0'))
    inference_cfg = cfg.get('inference', {})
    confidence_threshold = inference_cfg.get('confidence_threshold', 0.05)
    ensemble_method = inference_cfg.get('ensemble_method', 'max')
    
    # Create test dataset
    test_dataset = TestMouseBehaviorDataset(test_df, tracking_data)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Generate predictions with each model
    model_predictions = {}
    
    for model_name, model in models.items():
        logger.info(f"Generating predictions with {model_name.upper()} model...")
        
        # Get predictions
        predictions = predict_single(model, test_loader, device, return_proba=True)
        model_predictions[model_name] = predictions
        
        logger.info(f"{model_name.upper()} predictions generated: {predictions.shape}")
    
    # Ensemble predictions if multiple models
    if len(model_predictions) > 1:
        logger.info("Ensembling predictions from multiple models...")
        proba_list = list(model_predictions.values())
        ensemble_proba = ensemble_predictions(proba_list, ensemble_method)
        model_predictions['ensemble'] = ensemble_proba
    
    # Convert probabilities to predictions
    all_predictions = []
    
    for model_name, predictions in model_predictions.items():
        logger.info(f"Converting {model_name} probabilities to predictions...")
        logger.info(f"Predictions shape: {predictions.shape}")
        logger.info(f"Number of prediction data points: {len(test_dataset.predictions_data)}")
        
        # Get predicted classes
        predicted_classes = np.argmax(predictions, axis=1)
        predicted_proba = np.max(predictions, axis=1)
        
        # Debug: Log some statistics
        logger.info(f"Max probability: {np.max(predicted_proba):.4f}")
        logger.info(f"Min probability: {np.min(predicted_proba):.4f}")
        logger.info(f"Mean probability: {np.mean(predicted_proba):.4f}")
        logger.info(f"Confidence threshold: {confidence_threshold}")
        
        # Filter by confidence threshold
        confident_mask = predicted_proba >= confidence_threshold
        confident_count = np.sum(confident_mask)
        logger.info(f"Confident predictions: {confident_count}/{len(predicted_proba)}")
        
        # Create prediction dictionaries
        for i, pred_data in enumerate(test_dataset.predictions_data):
            if i < len(predicted_classes) and confident_mask[i]:
                behavior_map = {
                    0: 'approach', 1: 'attack', 2: 'avoid', 3: 'chase',
                    4: 'chaseattack', 5: 'submit', 6: 'rear', 7: 'shepherd'
                }
                
                behavior_name = behavior_map.get(predicted_classes[i], 'unknown')
                confidence = predicted_proba[i]
                
                all_predictions.append({
                    'video_id': pred_data['video_id'],
                    'frame': pred_data['frame'],
                    'agent_id': pred_data['agent_id'],
                    'target_id': pred_data['target_id'],
                    'behavior_name': behavior_name,
                    'confidence': confidence,
                    'model': model_name
                })
    
    logger.info(f"Generated {len(all_predictions)} confident predictions")
    return all_predictions


def run_inference_with_models(cfg: Dict, model_paths: Dict, test_csv: Optional[str] = None) -> pd.DataFrame:
    """
    Run inference with specific model paths
    
    Args:
        cfg: Configuration dictionary
        model_paths: Dictionary mapping model names to paths
        test_csv: Optional path to test CSV file
        
    Returns:
        DataFrame with submission predictions
    """
    logger.info("Starting inference with specific model paths...")
    
    # Set random seed
    set_seed(cfg.get('seed', 42))
    
    # Load test data
    test_df = load_test_data(cfg, test_csv)
    if test_df.empty:
        return pd.DataFrame()
    
    # Load tracking data
    tracking_data = load_tracking_data(cfg, test_df)
    
    # Load specified models
    models = {}
    device = torch.device(cfg.get('device', {}).get('device_str', 'cuda:0'))
    
    for model_name, model_path in model_paths.items():
        if Path(model_path).exists():
            try:
                model = load_model(model_path, model_name, input_dim=26)
                model = model.to(device)
                model.eval()
                models[model_name] = model
                logger.info(f"Loaded {model_name} model from {model_path}")
            except Exception as e:
                logger.warning(f"Error loading {model_name} model: {e}")
    
    if not models:
        logger.error("No models could be loaded.")
        return pd.DataFrame()
    
    # Generate predictions
    predictions = generate_predictions(cfg, test_df, tracking_data, models)
    
    if not predictions:
        return pd.DataFrame()
    
    # Create submission
    submission_path, submission_df = create_kaggle_submission(predictions, cfg)
    
    logger.info(f"Inference completed: {len(submission_df)} predictions")
    return submission_df


def validate_inference_setup(cfg: Dict) -> bool:
    """
    Validate inference configuration and model availability
    
    Args:
        cfg: Configuration dictionary
        
    Returns:
        True if setup is valid, False otherwise
    """
    logger.info("Validating inference setup...")
    
    # Check required paths
    dataset_path = Path(cfg.get('dataset', {}).get('path', ''))
    if not dataset_path.exists():
        logger.error(f"Dataset path does not exist: {dataset_path}")
        return False
    
    # Check test data
    test_csv = dataset_path / cfg.get('dataset', {}).get('test_csv', 'test.csv')
    if not test_csv.exists():
        logger.error(f"Test CSV not found: {test_csv}")
        return False
    
    # Check models
    models_dir = Path(cfg.get('paths', {}).get('models_dir', 'outputs/models'))
    cnn_model = models_dir / cfg.get('paths', {}).get('cnn_model_filename', 'cnn_enhanced_model.pth')
    lstm_model = models_dir / cfg.get('paths', {}).get('lstm_model_filename', 'lstm_enhanced_model.pth')
    
    if not cnn_model.exists() and not lstm_model.exists():
        logger.error("No trained models found. Run training first.")
        return False
    
    # Check device availability
    device_str = cfg.get('device', {}).get('device_str', 'cuda:0')
    if 'cuda' in device_str:
        import torch
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            cfg['device']['device_str'] = 'cpu'
            cfg['device']['use_cuda'] = False
    
    logger.info("Inference setup validation completed successfully!")
    return True


def run_inference_on_training(cfg: Dict, val_video_ids: List[str], output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Run inference on training videos for validation purposes.
    
    This allows evaluating model performance on held-out training videos
    where ground truth is available for proper evaluation.
    
    Args:
        cfg: Configuration dictionary
        val_video_ids: List of training video IDs to use for validation
        output_path: Optional output path for submission file
        
    Returns:
        DataFrame with predictions
    """
    logger.info(f"Running inference on {len(val_video_ids)} training videos for validation...")
    logger.info(f"Validation videos: {val_video_ids}")
    
    # Create a modified config for training data inference
    val_cfg = cfg.copy()
    
    # Override paths to use training data instead of test data
    dataset_path = Path(cfg['dataset']['path'])
    val_cfg['paths']['test_csv_path'] = str(dataset_path / "train.csv")
    val_cfg['paths']['test_tracking_dir'] = str(dataset_path / "train_tracking")
    
    # Load training metadata
    train_csv_path = dataset_path / "train.csv"
    if not train_csv_path.exists():
        raise FileNotFoundError(f"Training CSV not found: {train_csv_path}")
    
    train_df = pd.read_csv(train_csv_path)
    
    # Filter to only validation videos
    val_df = train_df[train_df['video_id'].astype(str).isin(val_video_ids)]
    if len(val_df) == 0:
        raise ValueError(f"No training videos found for IDs: {val_video_ids}")
    
    logger.info(f"Found {len(val_df)} training videos for validation")
    
    # Save filtered training data as temporary test CSV
    temp_test_csv = Path(cfg['paths']['outputs_dir']) / "temp_val_test.csv"
    temp_test_csv.parent.mkdir(parents=True, exist_ok=True)
    val_df.to_csv(temp_test_csv, index=False)
    
    # Update config to use temporary test CSV
    val_cfg['paths']['test_csv_path'] = str(temp_test_csv)
    
    try:
        # Run inference using the modified config
        submission_df = run_inference(val_cfg, test_csv=str(temp_test_csv), output_path=output_path)
        
        logger.info(f"Validation inference completed: {len(submission_df)} predictions generated")
        return submission_df
        
    finally:
        # Clean up temporary file
        if temp_test_csv.exists():
            temp_test_csv.unlink()
            logger.info("Cleaned up temporary validation test CSV")


def get_inference_summary(cfg: Dict) -> Dict:
    """
    Get summary of inference configuration and available models
    
    Args:
        cfg: Configuration dictionary
        
    Returns:
        Dictionary with inference summary
    """
    models_dir = Path(cfg.get('paths', {}).get('models_dir', 'outputs/models'))
    
    summary = {
        'configuration': {
            'device': cfg.get('device', {}).get('device_str', 'cuda:0'),
            'confidence_threshold': cfg.get('inference', {}).get('confidence_threshold', 0.05),
            'ensemble_method': cfg.get('inference', {}).get('ensemble_method', 'max')
        },
        'data': {
            'dataset_path': cfg.get('dataset', {}).get('path', ''),
            'test_csv': cfg.get('dataset', {}).get('test_csv', 'test.csv')
        },
        'models': {
            'cnn_available': (models_dir / cfg.get('paths', {}).get('cnn_model_filename', 'cnn_enhanced_model.pth')).exists(),
            'lstm_available': (models_dir / cfg.get('paths', {}).get('lstm_model_filename', 'lstm_enhanced_model.pth')).exists()
        },
        'outputs': {
            'submissions_dir': cfg.get('paths', {}).get('submissions_dir', 'outputs/submissions')
        }
    }
    
    return summary
