"""
Configuration loader for MABE Pipeline

Handles loading and merging configuration from multiple sources:
1. CLI arguments (highest priority)
2. Environment variables
3. User config files
4. Default configuration (lowest priority)
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: Optional[str] = None, overrides: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Load configuration with priority: CLI args > env vars > user config > defaults
    
    Args:
        config_path: Path to user config file (optional)
        overrides: List of override strings in format "key=value" or "nested.key=value"
        
    Returns:
        Merged configuration dictionary
    """
    # Start with default configuration
    default_config = _load_default_config()
    
    # Load user config if provided
    if config_path and Path(config_path).exists():
        user_config = _load_yaml_config(config_path)
        default_config = merge_configs(default_config, user_config)
        logger.info(f"Loaded user config from {config_path}")
    
    # Apply environment variable overrides
    env_config = _load_env_config()
    if env_config:
        default_config = merge_configs(default_config, env_config)
        logger.info("Applied environment variable overrides")
    
    # Apply CLI overrides
    if overrides:
        override_config = _parse_overrides(overrides)
        default_config = merge_configs(default_config, override_config)
        logger.info(f"Applied {len(overrides)} CLI overrides")
    
    return default_config


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two configuration dictionaries
    
    Args:
        base: Base configuration
        override: Override configuration
        
    Returns:
        Merged configuration
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def parse_override(override_str: str) -> tuple[str, Any]:
    """
    Parse a single override string into key-value pair
    
    Args:
        override_str: String in format "key=value" or "nested.key=value"
        
    Returns:
        Tuple of (key, value) where key is a list for nested keys
    """
    if '=' not in override_str:
        raise ValueError(f"Override must contain '=': {override_str}")
    
    key_str, value_str = override_str.split('=', 1)
    
    # Handle nested keys
    keys = key_str.split('.')
    
    # Convert value to appropriate type
    value = _convert_value(value_str)
    
    return keys, value


def _load_default_config() -> Dict[str, Any]:
    """Load default configuration from configs/default.yaml"""
    default_path = Path(__file__).parent.parent.parent.parent / "configs" / "default.yaml"
    return _load_yaml_config(str(default_path))


def _load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {e}")
        return {}


def _load_env_config() -> Dict[str, Any]:
    """Load configuration from environment variables"""
    env_config = {}
    
    # Common environment variables
    env_mappings = {
        'MABE_DATASET_PATH': ['dataset', 'path'],
        'MABE_MODELS_DIR': ['paths', 'models_dir'],
        'MABE_BATCH_SIZE': ['training', 'batch_size'],
        'MABE_EPOCHS': ['training', 'epochs'],
        'MABE_LEARNING_RATE': ['training', 'learning_rate'],
        'MABE_DEVICE': ['device', 'device_str'],
        'MABE_SEED': ['seed'],
    }
    
    for env_var, config_path in env_mappings.items():
        value = os.getenv(env_var)
        if value is not None:
            _set_nested_value(env_config, config_path, _convert_value(value))
    
    return env_config


def _parse_overrides(overrides: List[str]) -> Dict[str, Any]:
    """Parse CLI override strings into configuration dictionary"""
    config = {}
    
    for override_str in overrides:
        try:
            keys, value = parse_override(override_str)
            _set_nested_value(config, keys, value)
        except Exception as e:
            logger.error(f"Error parsing override '{override_str}': {e}")
            continue
    
    return config


def _set_nested_value(config: Dict[str, Any], keys: List[str], value: Any) -> None:
    """Set a nested value in configuration dictionary"""
    current = config
    
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    current[keys[-1]] = value


def _convert_value(value_str: str) -> Any:
    """Convert string value to appropriate Python type"""
    # Boolean values
    if value_str.lower() in ('true', 'false'):
        return value_str.lower() == 'true'
    
    # Numeric values
    try:
        if '.' in value_str:
            return float(value_str)
        else:
            return int(value_str)
    except ValueError:
        pass
    
    # String value
    return value_str


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Save configuration to YAML file
    
    Args:
        config: Configuration dictionary
        output_path: Output file path
    """
    try:
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        logger.info(f"Saved configuration to {output_path}")
    except Exception as e:
        logger.error(f"Error saving config to {output_path}: {e}")


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration has required keys
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid, False otherwise
    """
    required_keys = [
        'dataset',
        'paths', 
        'training',
        'device',
        'seed'
    ]
    
    for key in required_keys:
        if key not in config:
            logger.error(f"Missing required configuration key: {key}")
            return False
    
    return True
