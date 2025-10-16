"""
Tests for configuration management
"""

import pytest
import tempfile
import os
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mabe.utils.config import load_config, merge_configs, parse_override, validate_config


class TestConfigLoading:
    """Test configuration loading functionality"""
    
    def test_load_default_config(self):
        """Test loading default configuration"""
        cfg = load_config()
        
        # Check required keys exist
        assert 'dataset' in cfg
        assert 'paths' in cfg
        assert 'training' in cfg
        assert 'device' in cfg
        assert 'seed' in cfg
        
        # Check default values
        assert cfg['seed'] == 42
        assert cfg['device']['use_cuda'] == True
        assert cfg['training']['model_type'] == 'both'
    
    def test_merge_configs(self):
        """Test configuration merging"""
        base = {
            'training': {
                'epochs': 10,
                'batch_size': 32
            },
            'device': {
                'use_cuda': True
            }
        }
        
        override = {
            'training': {
                'epochs': 20
            },
            'seed': 123
        }
        
        merged = merge_configs(base, override)
        
        assert merged['training']['epochs'] == 20
        assert merged['training']['batch_size'] == 32  # Should be preserved
        assert merged['device']['use_cuda'] == True  # Should be preserved
        assert merged['seed'] == 123  # Should be added
    
    def test_parse_override(self):
        """Test parsing override strings"""
        # Simple override
        keys, value = parse_override("seed=42")
        assert keys == ['seed']
        assert value == 42
        
        # Nested override
        keys, value = parse_override("training.epochs=50")
        assert keys == ['training', 'epochs']
        assert value == 50
        
        # String value
        keys, value = parse_override("device.device_str=cuda:1")
        assert keys == ['device', 'device_str']
        assert value == "cuda:1"
        
        # Boolean value
        keys, value = parse_override("device.use_cuda=true")
        assert keys == ['device', 'use_cuda']
        assert value == True
    
    def test_override_parsing(self):
        """Test override parsing with various types"""
        overrides = [
            "seed=42",
            "training.epochs=50",
            "training.batch_size=16",
            "device.use_cuda=false",
            "device.device_str=cpu"
        ]
        
        cfg = load_config(overrides=overrides)
        
        assert cfg['seed'] == 42
        assert cfg['training']['epochs'] == 50
        assert cfg['training']['batch_size'] == 16
        assert cfg['device']['use_cuda'] == False
        assert cfg['device']['device_str'] == "cpu"
    
    def test_validate_config(self):
        """Test configuration validation"""
        # Valid config
        valid_config = {
            'dataset': {'path': '/data'},
            'paths': {'models_dir': '/models'},
            'training': {'epochs': 10},
            'device': {'use_cuda': True},
            'seed': 42
        }
        assert validate_config(valid_config) == True
        
        # Invalid config (missing required keys)
        invalid_config = {
            'dataset': {'path': '/data'},
            'training': {'epochs': 10}
        }
        assert validate_config(invalid_config) == False
    
    def test_environment_variables(self):
        """Test environment variable loading"""
        # Set environment variables
        os.environ['MABE_SEED'] = '123'
        os.environ['MABE_BATCH_SIZE'] = '64'
        os.environ['MABE_DEVICE'] = 'cpu'
        
        try:
            cfg = load_config()
            
            # Environment variables should override defaults
            assert cfg['seed'] == 123
            assert cfg['training']['batch_size'] == 64
            assert cfg['device']['device_str'] == 'cpu'
            
        finally:
            # Clean up environment variables
            for key in ['MABE_SEED', 'MABE_BATCH_SIZE', 'MABE_DEVICE']:
                if key in os.environ:
                    del os.environ[key]
    
    def test_user_config_file(self):
        """Test loading user configuration file"""
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
training:
  epochs: 100
  batch_size: 64
device:
  use_cuda: false
seed: 999
            """)
            config_path = f.name
        
        try:
            cfg = load_config(config_path=config_path)
            
            # User config should override defaults
            assert cfg['training']['epochs'] == 100
            assert cfg['training']['batch_size'] == 64
            assert cfg['device']['use_cuda'] == False
            assert cfg['seed'] == 999
            
        finally:
            # Clean up
            os.unlink(config_path)
    
    def test_override_precedence(self):
        """Test that CLI overrides take precedence over user config"""
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
training:
  epochs: 50
  batch_size: 32
seed: 100
            """)
            config_path = f.name
        
        try:
            # CLI overrides should take precedence
            overrides = ["training.epochs=200", "seed=42"]
            cfg = load_config(config_path=config_path, overrides=overrides)
            
            assert cfg['training']['epochs'] == 200  # CLI override
            assert cfg['training']['batch_size'] == 32  # From user config
            assert cfg['seed'] == 42  # CLI override
            
        finally:
            os.unlink(config_path)


if __name__ == "__main__":
    pytest.main([__file__])
