"""
Model definitions for the Image-Caption Alignment project.

This module contains the enhanced Vision Transformer image encoder and
utility functions for model initialization and management.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class PatchEmbedding(nn.Module):
    """Patch embedding layer for Vision Transformer."""
    
    def __init__(self, image_size: Tuple[int, int], patch_size: int, 
                 in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)
        self.patch_dim = in_channels * patch_size * patch_size
        
        # Patch embedding projection
        self.projection = nn.Linear(self.patch_dim, embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert image to patch embeddings.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Patch embeddings of shape (B, num_patches, embed_dim)
        """
        B, C, H, W = x.shape
        assert H % self.patch_size == 0, f"Height {H} not divisible by patch size {self.patch_size}"
        assert W % self.patch_size == 0, f"Width {W} not divisible by patch size {self.patch_size}"
        
        # Reshape to patches
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(B, C, -1, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B, self.num_patches, -1)
        
        # Project to embedding dimension
        x = self.projection(x)
        return x


class PositionalEncoding(nn.Module):
    """Learnable positional encoding for Vision Transformer."""
    
    def __init__(self, num_patches: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize positional embeddings."""
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings.
        
        Args:
            x: Input embeddings of shape (B, num_patches, embed_dim)
            
        Returns:
            Positionally encoded embeddings
        """
        x = x + self.pos_embedding
        return self.dropout(x)


class MLP(nn.Module):
    """Multi-layer perceptron for transformer blocks."""
    
    def __init__(self, in_features: int, hidden_features: int, 
                 out_features: int, dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)
        
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MLP.
        
        Args:
            x: Input tensor of shape (B, seq_len, in_features)
            
        Returns:
            Output tensor of shape (B, seq_len, out_features)
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm architecture."""
    
    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int, 
                 dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_dim, embed_dim, dropout, activation)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through transformer block.
        
        Args:
            x: Input tensor of shape (B, seq_len, embed_dim)
            
        Returns:
            Output tensor of shape (B, seq_len, embed_dim)
        """
        # Pre-norm attention
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        
        # Pre-norm MLP
        norm_x = self.norm2(x)
        mlp_out = self.mlp(norm_x)
        x = x + mlp_out
        
        return x


class ProjectionHead(nn.Module):
    """Projection head for better representation learning."""
    
    def __init__(self, input_dim: int, projection_dim: int, 
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, projection_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            current_dim = projection_dim
        
        # Final layer
        layers.append(nn.Linear(current_dim, projection_dim))
        
        self.projection = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through projection head.
        
        Args:
            x: Input tensor of shape (B, input_dim)
            
        Returns:
            Projected tensor of shape (B, projection_dim)
        """
        return self.projection(x)


class EnhancedImageTransformerEncoder(nn.Module):
    """Enhanced Vision Transformer encoder for image-caption alignment.
    
    This is an improved version of the original ImageTransformerEncoder with:
    - Better architecture (more layers, heads, embedding dimension)
    - Pre-norm architecture for better gradient flow
    - GELU activation instead of ReLU
    - Optional projection head for better representation learning
    - Better initialization strategies
    """
    
    def __init__(self, image_size: Tuple[int, int] = (32, 64), 
                 patch_size: int = 8, embed_dim: int = 512, 
                 mlp_dim: int = 2048, num_layers: int = 6, 
                 num_heads: int = 8, dropout: float = 0.1,
                 activation: str = "gelu", use_projection_head: bool = False,
                 projection_dim: int = 512, projection_layers: int = 2):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)
        
        # Patch embedding
        self.patch_embedding = PatchEmbedding(
            image_size, patch_size, in_channels=3, embed_dim=embed_dim
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            self.num_patches, embed_dim, dropout
        )
        
        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout, activation)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Projection head
        self.use_projection_head = use_projection_head
        if use_projection_head:
            self.projection_head = ProjectionHead(
                embed_dim, projection_dim, projection_layers, dropout
            )
        else:
            self.projection_head = None
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights using improved strategies."""
        # Initialize class token
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Initialize patch embedding
        nn.init.xavier_uniform_(self.patch_embedding.projection.weight)
        nn.init.zeros_(self.patch_embedding.projection.bias)
        
        # Initialize transformer blocks
        for block in self.blocks:
            # Attention weights
            nn.init.xavier_uniform_(block.attn.in_proj_weight)
            nn.init.zeros_(block.attn.in_proj_bias)
            nn.init.xavier_uniform_(block.attn.out_proj.weight)
            nn.init.zeros_(block.attn.out_proj.bias)
            
            # MLP weights
            nn.init.xavier_uniform_(block.mlp.fc1.weight)
            nn.init.zeros_(block.mlp.fc1.bias)
            nn.init.xavier_uniform_(block.mlp.fc2.weight)
            nn.init.zeros_(block.mlp.fc2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the image encoder.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Image embeddings of shape (B, embed_dim) or (B, projection_dim)
        """
        B = x.shape[0]
        
        # Convert to patch embeddings
        patch_embeddings = self.patch_embedding(x)  # (B, num_patches, embed_dim)
        
        # Add positional encoding
        patch_embeddings = self.pos_encoding(patch_embeddings)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, patch_embeddings], dim=1)  # (B, 1 + num_patches, embed_dim)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm
        x = self.norm(x)
        
        # Extract class token
        cls_token = x[:, 0]  # (B, embed_dim)
        
        # Apply projection head if enabled
        if self.use_projection_head:
            cls_token = self.projection_head(cls_token)
        
        return cls_token
    
    def get_num_parameters(self) -> int:
        """Get the total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_size_mb(self) -> float:
        """Get the model size in megabytes."""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / (1024 * 1024)


def create_image_encoder(config) -> EnhancedImageTransformerEncoder:
    """Create an image encoder from configuration.
    
    Args:
        config: Model configuration object
        
    Returns:
        Initialized image encoder
    """
    return EnhancedImageTransformerEncoder(
        image_size=config.model.image_size,
        patch_size=config.model.patch_size,
        embed_dim=config.model.embed_dim,
        mlp_dim=config.model.mlp_dim,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        dropout=config.model.dropout,
        activation=config.model.activation,
        use_projection_head=config.model.use_projection_head,
        projection_dim=config.model.projection_dim,
        projection_layers=config.model.projection_layers
    )


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model: nn.Module) -> str:
    """Get a summary of the model architecture and parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Formatted string with model summary
    """
    total_params = count_parameters(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    summary = f"Model Summary:\n"
    summary += f"  Total parameters: {total_params:,}\n"
    summary += f"  Trainable parameters: {trainable_params:,}\n"
    summary += f"  Non-trainable parameters: {total_params - trainable_params:,}\n"
    
    if hasattr(model, 'get_model_size_mb'):
        summary += f"  Model size: {model.get_model_size_mb():.2f} MB\n"
    
    return summary
