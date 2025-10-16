"""
MABE Model Training Module

Refactored from original train.py to work with configuration system.
Handles model training, validation, and saving.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import json
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional, Sequence
import warnings
warnings.filterwarnings('ignore')

# GPU/CUDA Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight

# Import utilities
from .utils.logger import get_logger
from .utils.seed import set_seed

logger = get_logger(__name__)


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        """Save model checkpoint"""
        self.best_weights = model.state_dict().copy()


class BehaviorCNN(nn.Module):
    """CNN model for behavior classification - MATCHES inference.py OLD architecture"""
    
    def __init__(self, input_dim, num_classes, dropout=0.3):
        super(BehaviorCNN, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Fully connected layers - OLD ARCHITECTURE (matches inference.py)
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class BehaviorLSTM(nn.Module):
    """LSTM model for temporal behavior classification - MATCHES inference.py OLD architecture"""
    
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=2, dropout=0.3):
        super(BehaviorLSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # LSTM layers - OLD ARCHITECTURE (matches inference.py)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Fully connected layers - OLD ARCHITECTURE
        self.fc1 = nn.Linear(hidden_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Ensure x has the right shape for LSTM
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last output
        output = lstm_out[:, -1, :]
        
        # Fully connected layers
        output = self.relu(self.fc1(output))
        output = self.dropout(output)
        output = self.relu(self.fc2(output))
        output = self.dropout(output)
        output = self.fc3(output)
        
        return output


def build_model(cfg: Dict, input_dim: int, model_type: str) -> nn.Module:
    """
    Build model based on configuration and model type
    
    Args:
        cfg: Configuration dictionary
        input_dim: Input feature dimension
        model_type: Model type ('cnn' or 'lstm')
        
    Returns:
        PyTorch model
    """
    training_cfg = cfg.get('training', {})
    dropout = training_cfg.get('dropout', 0.3)
    num_classes = 8  # Fixed for MABE behavior classes
    
    if model_type.lower() == 'cnn':
        model = BehaviorCNN(input_dim, num_classes, dropout)
        logger.info(f"Built CNN model: input_dim={input_dim}, num_classes={num_classes}, dropout={dropout}")
    elif model_type.lower() == 'lstm':
        hidden_dim = training_cfg.get('hidden_dim', 128)
        num_layers = training_cfg.get('num_layers', 2)
        model = BehaviorLSTM(input_dim, hidden_dim, num_classes, num_layers, dropout)
        logger.info(f"Built LSTM model: input_dim={input_dim}, hidden_dim={hidden_dim}, num_classes={num_classes}, num_layers={num_layers}, dropout={dropout}")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def compute_class_weights(labels: Sequence[int]) -> torch.Tensor:
    """
    Compute class weights for balanced training
    
    Args:
        labels: Sequence of class labels
        
    Returns:
        Tensor of class weights
    """
    # Calculate class weights using sklearn
    unique_classes = np.unique(labels)
    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=labels
    )
    
    # Create full weight array for all 8 classes
    full_weights = np.ones(8)  # Default weight of 1.0 for all classes
    for i, weight in enumerate(class_weights_array):
        class_id = unique_classes[i]
        full_weights[class_id] = weight
    
    # Handle missing classes (set to 1.0)
    for class_id in range(8):
        if class_id not in unique_classes:
            full_weights[class_id] = 1.0
    
    return torch.FloatTensor(full_weights)


def train(model, train_loader, val_loader, cfg: Dict, resume_checkpoint: Optional[str] = None) -> Dict:
    """
    Train a model with given configuration
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        cfg: Configuration dictionary
        resume_checkpoint: Optional checkpoint path to resume from
        
    Returns:
        Dictionary with training results
    """
    logger.info("Starting model training...")
    
    # Setup device
    device = torch.device(cfg.get('device', {}).get('device_str', 'cuda:0'))
    model = model.to(device)
    
    # Get training configuration
    training_cfg = cfg.get('training', {})
    epochs = training_cfg.get('epochs', 30)
    learning_rate = training_cfg.get('learning_rate', 0.001)
    weight_decay = training_cfg.get('weight_decay', 0.0)
    patience = training_cfg.get('early_stopping_patience', 10)
    
    # Setup optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Setup loss function
    criterion = nn.CrossEntropyLoss()
    
    # Setup early stopping
    early_stopping = EarlyStopping(patience=patience, min_delta=0.001)
    
    # Resume from checkpoint if provided
    start_epoch = 0
    if resume_checkpoint and Path(resume_checkpoint).exists():
        logger.info(f"Resuming from checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Resumed from epoch {start_epoch}")
    
    # Training history
    train_losses = []
    val_losses = []
    val_accuracies = []
    learning_rates = []
    
    best_val_acc = 0.0
    epochs_trained = 0
    
    for epoch in range(start_epoch, epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        
        # Calculate metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = 100.0 * train_correct / train_total
        val_acc = 100.0 * val_correct / val_total
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Store history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        learning_rates.append(current_lr)
        
        # Update best accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_trained = epoch + 1
        
        # Log progress
        if epoch % 5 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs}: "
                       f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                       f"Val Acc: {val_acc:.2f}%, LR: {current_lr:.6f}")
        
        # Early stopping
        if early_stopping(val_loss, model):
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # Save model
    model_path = _save_model(model, cfg, best_val_acc, epochs_trained)
    
    # Save checkpoint
    checkpoint_path = _save_checkpoint(model, optimizer, scheduler, epoch, cfg)
    
    logger.info(f"Training completed: Best Val Acc: {best_val_acc:.2f}% (epoch {epochs_trained})")
    logger.info(f"Model saved to: {model_path}")
    
    return {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'learning_rates': learning_rates,
        'best_val_acc': best_val_acc,
        'epochs_trained': epochs_trained,
        'model_path': model_path,
        'checkpoint_path': checkpoint_path
    }


def _save_model(model, cfg: Dict, best_val_acc: float, epochs_trained: int) -> str:
    """Save trained model"""
    models_dir = Path(cfg['paths']['models_dir'])
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine model filename based on model type
    model_type = cfg.get('training', {}).get('model_type', 'both')
    if 'cnn' in str(type(model)).lower():
        model_filename = cfg['paths']['cnn_model_filename']
    elif 'lstm' in str(type(model)).lower():
        model_filename = cfg['paths']['lstm_model_filename']
    else:
        model_filename = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
    
    model_path = models_dir / model_filename
    torch.save(model.state_dict(), model_path)
    
    # Save training info
    training_info = {
        'best_val_acc': best_val_acc,
        'epochs_trained': epochs_trained,
        'model_type': str(type(model).__name__),
        'timestamp': datetime.now().isoformat()
    }
    
    info_path = models_dir / f"{model_filename.replace('.pth', '_training_info.json')}"
    with open(info_path, 'w') as f:
        json.dump(training_info, f, indent=2)
    
    return str(model_path)


def _save_checkpoint(model, optimizer, scheduler, epoch, cfg: Dict) -> str:
    """Save training checkpoint"""
    models_dir = Path(cfg['paths']['models_dir'])
    models_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'timestamp': datetime.now().isoformat()
    }
    
    checkpoint_path = models_dir / f"checkpoint_epoch_{epoch}.pth"
    torch.save(checkpoint, checkpoint_path)
    
    return str(checkpoint_path)


def train_multiple_models(cfg: Dict, train_loader, val_loader, input_dim: int) -> Dict:
    """
    Train multiple model architectures
    
    Args:
        cfg: Configuration dictionary
        train_loader: Training data loader
        val_loader: Validation data loader
        input_dim: Input feature dimension
        
    Returns:
        Dictionary with training results for each model
    """
    training_cfg = cfg.get('training', {})
    model_type = training_cfg.get('model_type', 'both')
    
    results = {}
    
    if model_type in ['cnn', 'both']:
        logger.info("Training CNN model...")
        cnn_model = build_model(cfg, input_dim, 'cnn')
        cnn_results = train(cnn_model, train_loader, val_loader, cfg)
        results['cnn'] = cnn_results
    
    if model_type in ['lstm', 'both']:
        logger.info("Training LSTM model...")
        lstm_model = build_model(cfg, input_dim, 'lstm')
        lstm_results = train(lstm_model, train_loader, val_loader, cfg)
        results['lstm'] = lstm_results
    
    return results


def save_training_results(results: Dict, cfg: Dict) -> str:
    """
    Save training results and metrics
    
    Args:
        results: Training results dictionary
        cfg: Configuration dictionary
        
    Returns:
        Path to saved metrics file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outputs_dir = Path(cfg['paths']['outputs_dir'])
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    metrics_path = outputs_dir / f"training_metrics_{timestamp}.json"
    
    json_results = {}
    for model_name, result in results.items():
        json_results[model_name] = {
            'train_losses': result['train_losses'],
            'val_losses': result['val_losses'],
            'val_accuracies': result['val_accuracies'],
            'learning_rates': result['learning_rates'],
            'best_val_acc': result['best_val_acc'],
            'epochs_trained': result['epochs_trained']
        }
    
    # Add training configuration
    json_results['training_config'] = {
        'model_type': cfg.get('training', {}).get('model_type', 'both'),
        'epochs': cfg.get('training', {}).get('epochs', 30),
        'batch_size': cfg.get('training', {}).get('batch_size', 32),
        'learning_rate': cfg.get('training', {}).get('learning_rate', 0.001),
        'early_stopping_patience': cfg.get('training', {}).get('early_stopping_patience', 10),
        'device': cfg.get('device', {}).get('device_str', 'cuda:0')
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    logger.info(f"Training metrics saved to {metrics_path}")
    return str(metrics_path)
