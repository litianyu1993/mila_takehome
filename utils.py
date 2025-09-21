"""
Utility functions for the Image-Caption Alignment project.

This module contains various utility functions for training, evaluation,
and general purpose operations.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
import json
import logging
from datetime import datetime


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For completely deterministic results
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device: str = "auto") -> torch.device:
    """Get the appropriate device for computation.
    
    Args:
        device: Device specification ("auto", "cuda", "cpu", "mps")
        
    Returns:
        PyTorch device object
    """
    if device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device)


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: nn.Module) -> float:
    """Get the model size in megabytes.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size in MB
    """
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 * 1024)


def save_model(model: nn.Module, path: str, config: Optional[Dict] = None) -> None:
    """Save model and optionally configuration.
    
    Args:
        model: PyTorch model to save
        path: Path to save the model
        config: Optional configuration dictionary to save alongside
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save model state dict
    torch.save(model.state_dict(), path)
    
    # Save configuration if provided
    if config is not None:
        config_path = path.replace('.pth', '_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)


def load_model(model: nn.Module, path: str, device: torch.device) -> nn.Module:
    """Load model from saved state dict.
    
    Args:
        model: Model instance to load weights into
        path: Path to saved model
        device: Device to load model on
        
    Returns:
        Model with loaded weights
    """
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    return model


def create_output_dir(base_dir: str = "./output") -> str:
    """Create output directory with timestamp.
    
    Args:
        base_dir: Base directory for outputs
        
    Returns:
        Path to created output directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Setup logging configuration.
    
    Args:
        log_file: Optional log file path
        level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger("image_caption_alignment")
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all values."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        """Update with new value.
        
        Args:
            val: New value
            n: Number of samples
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stopping utility to stop training when validation loss stops improving."""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.0, restore_best_weights: bool = True):
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            model: Model to potentially restore weights for
            
        Returns:
            True if training should stop, False otherwise
        """
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
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        
        return False
    
    def save_checkpoint(self, model: nn.Module):
        """Save model checkpoint.
        
        Args:
            model: Model to save
        """
        self.best_weights = model.state_dict().copy()


def get_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    """Get current learning rate from optimizer.
    
    Args:
        optimizer: PyTorch optimizer
        
    Returns:
        Current learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def adjust_learning_rate(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """Adjust learning rate of optimizer.
    
    Args:
        optimizer: PyTorch optimizer
        lr: New learning rate
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def cosine_annealing_lr(epoch: int, total_epochs: int, base_lr: float, 
                       min_lr: float = 0.0) -> float:
    """Compute cosine annealing learning rate.
    
    Args:
        epoch: Current epoch
        total_epochs: Total number of epochs
        base_lr: Base learning rate
        min_lr: Minimum learning rate
        
    Returns:
        Learning rate for current epoch
    """
    if epoch >= total_epochs:
        return min_lr
    
    return min_lr + (base_lr - min_lr) * 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))


def warmup_lr(epoch: int, warmup_epochs: int, base_lr: float) -> float:
    """Compute warmup learning rate.
    
    Args:
        epoch: Current epoch
        warmup_epochs: Number of warmup epochs
        base_lr: Base learning rate
        
    Returns:
        Learning rate for current epoch
    """
    if epoch >= warmup_epochs:
        return base_lr
    
    return base_lr * epoch / warmup_epochs


def get_lr_scheduler(optimizer: torch.optim.Optimizer, 
                    scheduler_type: str = "cosine",
                    total_epochs: int = 100,
                    warmup_epochs: int = 10,
                    min_lr: float = 0.0) -> torch.optim.lr_scheduler._LRScheduler:
    """Get learning rate scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_type: Type of scheduler ("cosine", "step", "plateau")
        total_epochs: Total number of epochs
        warmup_epochs: Number of warmup epochs
        min_lr: Minimum learning rate
        
    Returns:
        Learning rate scheduler
    """
    if scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs - warmup_epochs, eta_min=min_lr
        )
    elif scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=total_epochs // 3, gamma=0.1
        )
    elif scheduler_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=min_lr
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def format_time(seconds: float) -> str:
    """Format time in seconds to human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def print_model_summary(model: nn.Module, input_size: Tuple[int, ...] = None) -> None:
    """Print a summary of the model architecture.
    
    Args:
        model: PyTorch model
        input_size: Optional input size for computing FLOPs
    """
    print("=" * 80)
    print("MODEL SUMMARY")
    print("=" * 80)
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    # Model size
    model_size = get_model_size_mb(model)
    print(f"Model size: {model_size:.2f} MB")
    
    # Architecture details
    if hasattr(model, 'get_model_size_mb'):
        print(f"Model-specific size: {model.get_model_size_mb():.2f} MB")
    
    print("=" * 80)
