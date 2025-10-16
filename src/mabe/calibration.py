"""
Model Calibration using Temperature Scaling

Implements temperature scaling for post-hoc calibration of neural network
confidence scores to improve reliability of predictions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TemperatureScaling(nn.Module):
    """
    Post-hoc calibration using temperature scaling
    
    Temperature scaling is a simple but effective method for calibrating
    neural network confidence scores without retraining the model.
    
    Reference: "On Calibration of Modern Neural Networks"
    https://arxiv.org/abs/1706.04599
    """
    
    def __init__(self, temperature_init: float = 1.5):
        """
        Initialize temperature scaling
        
        Args:
            temperature_init: Initial temperature value
        """
        super(TemperatureScaling, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * temperature_init)
        logger.info(f"Initialized temperature scaling with T={temperature_init}")
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling to logits
        
        Args:
            logits: Raw model outputs (logits)
            
        Returns:
            Calibrated logits
        """
        return logits / self.temperature
    
    def calibrate(self, model: nn.Module, val_loader: DataLoader, device: torch.device, 
                  max_iterations: int = 1000, lr: float = 0.01) -> None:
        """
        Calibrate temperature parameter on validation set
        
        Args:
            model: Trained model to calibrate
            val_loader: Validation data loader
            device: Device to run calibration on
            max_iterations: Maximum optimization iterations
            lr: Learning rate for temperature optimization
        """
        logger.info("Starting temperature scaling calibration...")
        
        # Set model to evaluation mode
        model.eval()
        
        # Collect logits and labels from validation set
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                data, target = data.to(device), target.to(device)
                
                # Get model outputs
                logits = model(data)
                all_logits.append(logits.cpu())
                all_labels.append(target.cpu())
        
        # Combine all data
        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)
        
        logger.info(f"Calibrating on {len(logits)} validation samples")
        
        # Optimize temperature parameter
        self._optimize_temperature(logits, labels, max_iterations, lr)
        
        logger.info(f"Calibration complete. Final temperature: {self.temperature.item():.4f}")
    
    def _optimize_temperature(self, logits: torch.Tensor, labels: torch.Tensor, 
                           max_iterations: int, lr: float) -> None:
        """
        Optimize temperature parameter using NLL loss
        
        Args:
            logits: Model logits
            labels: True labels
            max_iterations: Maximum optimization iterations
            lr: Learning rate
        """
        # Set up optimizer for temperature parameter only
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iterations)
        
        def eval_loss():
            optimizer.zero_grad()
            loss = nn.CrossEntropyLoss()(self.forward(logits), labels)
            loss.backward()
            return loss
        
        # Optimize
        optimizer.step(eval_loss)
    
    def get_temperature(self) -> float:
        """Get current temperature value"""
        return self.temperature.item()
    
    def set_temperature(self, temperature: float) -> None:
        """Set temperature value"""
        with torch.no_grad():
            self.temperature.fill_(temperature)
        logger.info(f"Temperature set to {temperature}")


def calibrate_model(model: nn.Module, val_loader: DataLoader, device: torch.device, 
                   cfg: dict) -> TemperatureScaling:
    """
    Apply temperature scaling calibration to a trained model
    
    Args:
        model: Trained model to calibrate
        val_loader: Validation data loader
        device: Device to run calibration on
        cfg: Configuration dictionary
        
    Returns:
        Calibrated temperature scaler
    """
    training_cfg = cfg.get('training', {})
    
    if not training_cfg.get('use_calibration', True):
        logger.info("Model calibration disabled")
        return None
    
    # Initialize temperature scaler
    temp_init = training_cfg.get('calibration_temperature_init', 1.5)
    temp_scaler = TemperatureScaling(temperature_init=temp_init)
    
    # Calibrate on validation set
    temp_scaler.calibrate(model, val_loader, device)
    
    return temp_scaler


def save_calibration(temp_scaler: TemperatureScaling, model_path: str) -> None:
    """
    Save temperature scaler with model
    
    Args:
        temp_scaler: Calibrated temperature scaler
        model_path: Path to save the scaler
    """
    if temp_scaler is None:
        return
    
    scaler_path = model_path.replace('.pth', '_calibration.pth')
    torch.save({
        'temperature': temp_scaler.temperature.data,
        'temperature_value': temp_scaler.get_temperature()
    }, scaler_path)
    logger.info(f"Calibration saved to {scaler_path}")


def load_calibration(model_path: str) -> Optional[TemperatureScaling]:
    """
    Load temperature scaler from file
    
    Args:
        model_path: Path to model file
        
    Returns:
        Loaded temperature scaler or None
    """
    scaler_path = model_path.replace('.pth', '_calibration.pth')
    
    try:
        checkpoint = torch.load(scaler_path, map_location='cpu')
        temp_scaler = TemperatureScaling()
        temp_scaler.temperature.data = checkpoint['temperature']
        logger.info(f"Calibration loaded from {scaler_path}, T={temp_scaler.get_temperature():.4f}")
        return temp_scaler
    except FileNotFoundError:
        logger.warning(f"No calibration file found at {scaler_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading calibration: {e}")
        return None


def apply_calibration(logits: torch.Tensor, temp_scaler: Optional[TemperatureScaling]) -> torch.Tensor:
    """
    Apply temperature scaling to model outputs
    
    Args:
        logits: Raw model logits
        temp_scaler: Temperature scaler (None for no calibration)
        
    Returns:
        Calibrated logits
    """
    if temp_scaler is None:
        return logits
    
    return temp_scaler(logits)


def calculate_calibration_error(logits: torch.Tensor, labels: torch.Tensor, 
                               temp_scaler: Optional[TemperatureScaling] = None,
                               n_bins: int = 15) -> float:
    """
    Calculate Expected Calibration Error (ECE)
    
    Args:
        logits: Model logits
        labels: True labels
        temp_scaler: Temperature scaler (optional)
        n_bins: Number of bins for calibration error calculation
        
    Returns:
        Expected Calibration Error
    """
    # Apply calibration if provided
    if temp_scaler is not None:
        logits = temp_scaler(logits)
    
    # Get probabilities
    probs = torch.softmax(logits, dim=1)
    max_probs, predictions = torch.max(probs, dim=1)
    
    # Convert to numpy
    max_probs = max_probs.detach().cpu().numpy()
    predictions = predictions.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    
    # Calculate ECE
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (max_probs > bin_lower) & (max_probs <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = (predictions[in_bin] == labels[in_bin]).mean()
            avg_confidence_in_bin = max_probs[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece
