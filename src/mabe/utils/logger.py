"""
Centralized logging setup for MABE Pipeline

Provides consistent logging configuration across all pipeline components.
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional, Dict, Any
import sys


def get_logger(name: str, log_level: str = 'INFO', log_file: Optional[str] = None) -> logging.Logger:
    """
    Get a logger with consistent formatting
    
    Args:
        name: Logger name (usually __name__)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid adding multiple handlers
    if logger.handlers:
        return logger
    
    # Set level
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.TimedRotatingFileHandler(
            log_file,
            when='midnight',
            interval=1,
            backupCount=7,
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def configure_logging_from_config(cfg: Dict[str, Any]) -> None:
    """
    Configure logging from configuration dictionary
    
    Args:
        cfg: Configuration dictionary with logging settings
    """
    logging_config = cfg.get('logging', {})
    
    # Set root logger level
    log_level = logging_config.get('level', 'INFO')
    logging.getLogger().setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Configure file logging if specified
    log_file = logging_config.get('file')
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create file handler for root logger
        file_handler = logging.handlers.TimedRotatingFileHandler(
            log_file,
            when='midnight',
            interval=1,
            backupCount=7,
            encoding='utf-8'
        )
        
        formatter = logging.Formatter(
            logging_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        file_handler.setFormatter(formatter)
        
        # Add to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)


def setup_pipeline_logging(cfg: Dict[str, Any]) -> logging.Logger:
    """
    Setup logging for the entire pipeline
    
    Args:
        cfg: Configuration dictionary
        
    Returns:
        Main pipeline logger
    """
    # Configure from config
    configure_logging_from_config(cfg)
    
    # Get main pipeline logger
    logger = get_logger('mabe_pipeline')
    
    # Log configuration summary
    logger.info("MABE Pipeline Logging Initialized")
    logger.info(f"Dataset path: {cfg.get('dataset', {}).get('path', 'Not specified')}")
    logger.info(f"Device: {cfg.get('device', {}).get('device_str', 'Not specified')}")
    logger.info(f"Seed: {cfg.get('seed', 'Not specified')}")
    
    return logger


def log_config_summary(cfg: Dict[str, Any], logger: logging.Logger) -> None:
    """
    Log a summary of the current configuration
    
    Args:
        cfg: Configuration dictionary
        logger: Logger instance
    """
    logger.info("=== Configuration Summary ===")
    
    # Dataset settings
    dataset_cfg = cfg.get('dataset', {})
    logger.info(f"Dataset path: {dataset_cfg.get('path', 'Not specified')}")
    logger.info(f"Max videos: {cfg.get('training', {}).get('max_videos', 'Not specified')}")
    
    # Training settings
    training_cfg = cfg.get('training', {})
    logger.info(f"Model type: {training_cfg.get('model_type', 'Not specified')}")
    logger.info(f"Batch size: {training_cfg.get('batch_size', 'Not specified')}")
    logger.info(f"Epochs: {training_cfg.get('epochs', 'Not specified')}")
    logger.info(f"Learning rate: {training_cfg.get('learning_rate', 'Not specified')}")
    
    # Device settings
    device_cfg = cfg.get('device', {})
    logger.info(f"Device: {device_cfg.get('device_str', 'Not specified')}")
    logger.info(f"CUDA enabled: {device_cfg.get('use_cuda', 'Not specified')}")
    
    logger.info("=== End Configuration Summary ===")


def log_training_progress(epoch: int, total_epochs: int, train_loss: float, 
                         val_loss: float, val_acc: float, logger: logging.Logger) -> None:
    """
    Log training progress in a consistent format
    
    Args:
        epoch: Current epoch number
        total_epochs: Total number of epochs
        train_loss: Training loss
        val_loss: Validation loss
        val_acc: Validation accuracy
        logger: Logger instance
    """
    logger.info(f"Epoch {epoch}/{total_epochs}: "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val Acc: {val_acc:.2f}%")


def log_inference_results(num_predictions: int, submission_path: str, 
                         logger: logging.Logger) -> None:
    """
    Log inference results
    
    Args:
        num_predictions: Number of predictions generated
        submission_path: Path to submission file
        logger: Logger instance
    """
    logger.info(f"Inference completed: {num_predictions} predictions generated")
    logger.info(f"Submission saved to: {submission_path}")


def log_error_with_context(error: Exception, context: str, logger: logging.Logger) -> None:
    """
    Log error with additional context
    
    Args:
        error: Exception that occurred
        context: Additional context about where the error occurred
        logger: Logger instance
    """
    logger.error(f"Error in {context}: {str(error)}")
    logger.error(f"Error type: {type(error).__name__}")
    
    # Log full traceback at DEBUG level
    import traceback
    logger.debug(f"Full traceback:\n{traceback.format_exc()}")
