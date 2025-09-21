"""
Configuration management for the Image-Caption Alignment project.

This module provides centralized configuration management for all hyperparameters,
model architecture settings, and training parameters.
"""

import json
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple


@dataclass
class ModelConfig:
    """Configuration for the Vision Transformer image encoder."""
    # Architecture parameters
    image_size: Tuple[int, int] = (32, 64)
    patch_size: int = 8
    embed_dim: int = 512  # Match CLIP text encoder output dimension
    mlp_dim: int = 2048
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    activation: str = "gelu"
    
    # Projection head parameters
    use_projection_head: bool = False  # Not needed since dimensions already match
    projection_dim: int = 512
    projection_layers: int = 2


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    # Basic training parameters
    batch_size: int = 32
    num_epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 2
    
    # Optimization parameters
    gradient_clip_norm: float = 1.0
    temperature: float = 0.07

    # loss function parameters
    loss_type: str = "standard"
    
    # Data parameters
    num_workers: int = 0  # Set to 0 for Windows compatibility
    pin_memory: bool = True
    
    # Sampling parameters
    different_class_prob: float = 0.5  # Probability of sampling different classes


@dataclass
class DataConfig:
    """Configuration for data loading and augmentation."""
    # Dataset parameters
    data_root: str = "./data"
    download: bool = True
    
    # Augmentation parameters
    use_augmentation: bool = True
    horizontal_flip_prob: float = 0.5
    color_jitter_strength: float = 0.2
    rotation_degrees: float = 10.0
    
    # Normalization parameters
    mean: List[float] = None
    std: List[float] = None
    
    def __post_init__(self):
        if self.mean is None:
            self.mean = [0.485, 0.456, 0.406]
        if self.std is None:
            self.std = [0.229, 0.224, 0.225]


@dataclass
class Config:
    """Main configuration class combining all sub-configurations."""
    model: ModelConfig = None
    training: TrainingConfig = None
    data: DataConfig = None
    
    # System parameters
    seed: int = 42
    device: str = "auto"  # "auto", "cuda", "cpu", "mps"
    output_dir: str = "./output"
    save_best_only: bool = True
    save_frequency: int = 5  # Save every N epochs
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.data is None:
            self.data = DataConfig()
    
    @classmethod
    def from_json(cls, filepath: str) -> 'Config':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        # Handle nested configurations
        config = cls()
        for key, value in config_dict.items():
            if key == 'model' and isinstance(value, dict):
                config.model = ModelConfig(**value)
            elif key == 'training' and isinstance(value, dict):
                config.training = TrainingConfig(**value)
            elif key == 'data' and isinstance(value, dict):
                config.data = DataConfig(**value)
            else:
                setattr(config, key, value)
        
        return config
    
    def to_json(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        config_dict = asdict(self)
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def update(self, **kwargs) -> None:
        """Update configuration with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            elif hasattr(self.model, key):
                setattr(self.model, key, value)
            elif hasattr(self.training, key):
                setattr(self.training, key, value)
            elif hasattr(self.data, key):
                setattr(self.data, key, value)


def get_default_config() -> Config:
    """Get the default configuration."""
    return Config()


def get_enhanced_config() -> Config:
    """Get an enhanced configuration with improved parameters."""
    config = Config()
    
    # Enhanced model configuration
    config.model.embed_dim = 512  # Match CLIP text encoder
    config.model.num_layers = 6
    config.model.num_heads = 8
    config.model.mlp_dim = 2048
    config.model.use_projection_head = False  # Not needed since dimensions match
    config.model.projection_dim = 512
    
    # Enhanced training configuration
    config.training.batch_size = 32
    config.training.num_epochs = 20
    config.training.learning_rate = 1e-4
    config.training.weight_decay = 0.01
    config.training.temperature = 0.07
    
    # Enhanced data configuration
    config.data.use_augmentation = True
    config.data.different_class_prob = 0.5
    config.training.num_workers = 0  # Windows compatibility
    
    return config
