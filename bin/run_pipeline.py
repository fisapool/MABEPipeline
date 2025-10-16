#!/usr/bin/env python3
"""
MABE Pipeline CLI Entry Point

Central command-line interface for the MABE machine learning pipeline.
Provides subcommands for all pipeline stages: preprocess, train, tune, infer, evaluate, all.
"""

import argparse
import sys
import os
from pathlib import Path
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mabe.utils import load_config, get_logger, set_seed
from mabe import (
    run_training,
    run_inference,
    run_optuna,
    run_local_evaluation
)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser with all subcommands"""
    parser = argparse.ArgumentParser(
        description="MABE Pipeline - Machine Learning Pipeline for Mouse Behavior Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python bin/run_pipeline.py all --config configs/default.yaml

  # Train models only
  python bin/run_pipeline.py train --config configs/default.yaml --override training.epochs=10

  # Run inference
  python bin/run_pipeline.py infer --config configs/default.yaml --device cuda

  # Hyperparameter tuning
  python bin/run_pipeline.py tune --config configs/default.yaml --override optuna.n_trials=20
        """
    )
    
    # Global arguments
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Configuration file path')
    parser.add_argument('--override', action='append', default=[],
                       help='Override configuration values (format: key=value)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default=None,
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Pipeline commands')
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser('preprocess', help='Run data preprocessing')
    preprocess_parser.add_argument('--max-videos', type=int, default=None,
                                 help='Maximum number of videos to process')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument('--resume', action='store_true',
                             help='Resume training from checkpoint')
    train_parser.add_argument('--checkpoint', type=str, default=None,
                             help='Checkpoint file path')
    
    # Tune command
    tune_parser = subparsers.add_parser('tune', help='Run hyperparameter tuning')
    tune_parser.add_argument('--n-trials', type=int, default=None,
                            help='Number of Optuna trials')
    tune_parser.add_argument('--timeout', type=int, default=None,
                            help='Timeout in seconds')
    tune_parser.add_argument('--model-type', choices=['cnn', 'lstm', 'both'], default='both',
                            help='Model type to tune')
    
    # Infer command
    infer_parser = subparsers.add_parser('infer', help='Run inference')
    infer_parser.add_argument('--test-csv', type=str, default=None,
                             help='Test CSV file path')
    infer_parser.add_argument('--output', type=str, default=None,
                             help='Output submission file path')
    infer_parser.add_argument('--confidence', type=float, default=None,
                             help='Confidence threshold')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate predictions')
    eval_parser.add_argument('--predictions', type=str, required=True,
                            help='Predictions CSV file path')
    
    # All command
    all_parser = subparsers.add_parser('all', help='Run full pipeline')
    
    return parser


def setup_logging(cfg: dict, verbose: bool = False) -> logging.Logger:
    """Setup logging from configuration"""
    log_level = 'DEBUG' if verbose else cfg.get('logging', {}).get('level', 'INFO')
    logger = get_logger('mabe_pipeline', log_level)
    
    # Configure file logging if specified
    log_file = cfg.get('logging', {}).get('file')
    if log_file:
        from mabe.utils.logger import configure_logging_from_config
        configure_logging_from_config(cfg)
    
    return logger


def run_preprocess_command(args, cfg: dict) -> int:
    """Run preprocessing command"""
    logger = logging.getLogger('mabe_pipeline')
    logger.info("Starting data preprocessing...")
    
    try:
        from mabe.preprocessing import preprocess_data
        
        # Handle max_videos attribute - check if it exists in args
        max_videos = getattr(args, 'max_videos', None) or cfg.get('training', {}).get('max_videos', 5)
        logger.info(f"Processing up to {max_videos} videos")
        
        # Run preprocessing
        frame_labels_df = preprocess_data(cfg, max_videos=max_videos)
        
        logger.info(f"Preprocessing completed: {len(frame_labels_df)} frame labels generated")
        return 0
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        return 1


def run_train_command(args, cfg: dict) -> int:
    """Run training command"""
    logger = logging.getLogger('mabe_pipeline')
    logger.info("Starting model training...")
    
    try:
        # Handle resume and checkpoint attributes - check if they exist in args
        resume = getattr(args, 'resume', False)
        checkpoint_path = getattr(args, 'checkpoint', None)
        
        # Run training
        results = run_training(
            cfg=cfg,
            resume=resume,
            checkpoint_path=checkpoint_path
        )
        
        logger.info("Training completed successfully")
        logger.info(f"Models saved to: {cfg['paths']['models_dir']}")
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1


def run_tune_command(args, cfg: dict) -> int:
    """Run hyperparameter tuning command"""
    logger = logging.getLogger('mabe_pipeline')
    logger.info("Starting hyperparameter tuning...")
    
    try:
        # Override config with command line arguments
        if args.n_trials:
            cfg['optuna']['n_trials'] = args.n_trials
        if args.timeout:
            cfg['optuna']['timeout'] = args.timeout
        
        # Run tuning
        results = run_optuna(cfg)
        
        logger.info("Hyperparameter tuning completed successfully")
        logger.info(f"Best parameters: {results.get('best_params', {})}")
        return 0
        
    except Exception as e:
        logger.error(f"Hyperparameter tuning failed: {e}")
        return 1


def run_infer_command(args, cfg: dict) -> int:
    """Run inference command"""
    logger = logging.getLogger('mabe_pipeline')
    logger.info("Starting inference...")
    
    try:
        # Handle inference attributes - check if they exist in args
        confidence = getattr(args, 'confidence', None)
        test_csv = getattr(args, 'test_csv', None)
        output_path = getattr(args, 'output', None)
        
        # Override config with command line arguments
        if confidence:
            cfg['inference']['confidence_threshold'] = confidence
        
        # Run inference
        submission_df = run_inference(
            cfg=cfg,
            test_csv=test_csv,
            output_path=output_path
        )
        
        logger.info(f"Inference completed: {len(submission_df)} predictions generated")
        return 0
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return 1


def run_evaluate_command(args, cfg: dict) -> int:
    """Run evaluation command"""
    logger = logging.getLogger('mabe_pipeline')
    logger.info("Starting evaluation...")
    
    try:
        # Run evaluation
        metrics = run_local_evaluation(cfg, args.predictions)
        
        logger.info("Evaluation completed successfully")
        logger.info(f"Metrics: {metrics}")
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1


def run_all_command(args, cfg: dict) -> int:
    """Run full pipeline command"""
    logger = logging.getLogger('mabe_pipeline')
    logger.info("Starting full pipeline...")
    
    try:
        # Step 1: Preprocessing
        logger.info("Step 1/5: Data preprocessing")
        if run_preprocess_command(args, cfg) != 0:
            return 1
        
        # Step 2: Training
        logger.info("Step 2/5: Model training")
        if run_train_command(args, cfg) != 0:
            return 1
        
        # Step 3: Inference
        logger.info("Step 3/5: Inference")
        if run_infer_command(args, cfg) != 0:
            return 1
        
        # Step 4: Evaluation (if ground truth available)
        logger.info("Step 4/5: Evaluation")
        # Note: Evaluation requires ground truth, so this might be skipped
        # if run_evaluate_command(args, cfg) != 0:
        #     return 1
        
        logger.info("Full pipeline completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Full pipeline failed: {e}")
        return 1


def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        # Load configuration
        cfg = load_config(args.config, args.override)
        
        # Apply command line overrides
        if args.seed is not None:
            cfg['seed'] = args.seed
        if args.device is not None:
            cfg['device']['device_str'] = args.device
            cfg['device']['use_cuda'] = (args.device == 'cuda')
        
        # Setup logging
        logger = setup_logging(cfg, args.verbose)
        
        # Set random seed
        set_seed(cfg['seed'])
        
        # Log configuration summary
        from mabe.utils.logger import log_config_summary
        log_config_summary(cfg, logger)
        
        # Run appropriate command
        if args.command == 'preprocess':
            return run_preprocess_command(args, cfg)
        elif args.command == 'train':
            return run_train_command(args, cfg)
        elif args.command == 'tune':
            return run_tune_command(args, cfg)
        elif args.command == 'infer':
            return run_infer_command(args, cfg)
        elif args.command == 'evaluate':
            return run_evaluate_command(args, cfg)
        elif args.command == 'all':
            return run_all_command(args, cfg)
        else:
            logger.error(f"Unknown command: {args.command}")
            return 1
            
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
