"""
Random seed management for MABE Pipeline

Provides deterministic behavior across all pipeline components.
"""

import random
import numpy as np
import torch
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """
    Set random seed for all random number generators
    
    Args:
        seed: Random seed value
    """
    # Python random
    random.seed(seed)
    
    # NumPy random
    np.random.seed(seed)
    
    # PyTorch random
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # PyTorch deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seed set to {seed}")
    logger.info("Deterministic behavior enabled for PyTorch")


def get_rng_state() -> Dict[str, Any]:
    """
    Get current random number generator states for reproducibility logging
    
    Returns:
        Dictionary containing RNG states and settings
    """
    state = {
        'python_seed': random.getstate(),
        'numpy_seed': np.random.get_state(),
        'torch_seed': torch.initial_seed(),
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'cudnn_benchmark': torch.backends.cudnn.benchmark
    }
    
    if torch.cuda.is_available():
        state['cuda_seed'] = torch.cuda.initial_seed()
        state['cuda_device_count'] = torch.cuda.device_count()
    
    return state


def log_rng_state(logger: logging.Logger) -> None:
    """
    Log current RNG state for debugging
    
    Args:
        logger: Logger instance
    """
    state = get_rng_state()
    
    logger.debug("=== RNG State ===")
    logger.debug(f"PyTorch initial seed: {state['torch_seed']}")
    logger.debug(f"CuDNN deterministic: {state['cudnn_deterministic']}")
    logger.debug(f"CuDNN benchmark: {state['cudnn_benchmark']}")
    
    if 'cuda_seed' in state:
        logger.debug(f"CUDA initial seed: {state['cuda_seed']}")
        logger.debug(f"CUDA device count: {state['cuda_device_count']}")
    
    logger.debug("=== End RNG State ===")


def ensure_deterministic() -> None:
    """
    Ensure deterministic behavior across all components
    
    This function should be called at the start of each pipeline stage
    to ensure reproducible results.
    """
    # Set PyTorch to deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables for additional determinism
    import os
    os.environ['PYTHONHASHSEED'] = '0'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    logger.info("Deterministic behavior enforced")


def set_seed_from_config(cfg: Dict[str, Any]) -> None:
    """
    Set seed from configuration dictionary
    
    Args:
        cfg: Configuration dictionary
    """
    seed = cfg.get('seed', 42)
    set_seed(seed)
    
    # Log RNG state for debugging
    log_rng_state(logger)


def create_reproducible_loader(dataset, batch_size: int, shuffle: bool = True, 
                              num_workers: int = 0) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader with reproducible behavior
    
    Args:
        dataset: PyTorch dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        
    Returns:
        DataLoader with reproducible settings
    """
    # Use a generator for reproducible shuffling
    generator = torch.Generator()
    generator.manual_seed(42)
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
        pin_memory=torch.cuda.is_available()
    )


def save_seed_info(seed: int, output_path: str) -> None:
    """
    Save seed information to file for reproducibility
    
    Args:
        seed: Random seed used
        output_path: Path to save seed information
    """
    import json
    from datetime import datetime
    
    seed_info = {
        'seed': seed,
        'timestamp': datetime.now().isoformat(),
        'rng_state': get_rng_state(),
        'pytorch_version': torch.__version__,
        'numpy_version': np.__version__
    }
    
    with open(output_path, 'w') as f:
        json.dump(seed_info, f, indent=2, default=str)
    
    logger.info(f"Seed information saved to {output_path}")


def load_seed_info(seed_path: str) -> Dict[str, Any]:
    """
    Load seed information from file
    
    Args:
        seed_path: Path to seed information file
        
    Returns:
        Seed information dictionary
    """
    import json
    
    try:
        with open(seed_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading seed info from {seed_path}: {e}")
        return {}


def restore_rng_state(state: Dict[str, Any]) -> None:
    """
    Restore random number generator state
    
    Args:
        state: RNG state dictionary from get_rng_state()
    """
    try:
        # Restore Python random state
        if 'python_seed' in state:
            random.setstate(state['python_seed'])
        
        # Restore NumPy random state
        if 'numpy_seed' in state:
            np.random.set_state(state['numpy_seed'])
        
        # Restore PyTorch settings
        if 'cudnn_deterministic' in state:
            torch.backends.cudnn.deterministic = state['cudnn_deterministic']
        if 'cudnn_benchmark' in state:
            torch.backends.cudnn.benchmark = state['cudnn_benchmark']
        
        logger.info("RNG state restored")
        
    except Exception as e:
        logger.error(f"Error restoring RNG state: {e}")
