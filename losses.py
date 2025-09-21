"""
Loss functions for the Image-Caption Alignment project.

This module contains the contrastive loss function and related utilities
for training the vision-language model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def contrastive_loss(image_features: torch.Tensor, 
                    text_features: torch.Tensor, 
                    temperature: float = 0.07) -> torch.Tensor:
    """Compute bidirectional contrastive loss between image and text features.
    
    This loss encourages positive pairs (matching image-text pairs) to have
    high similarity and negative pairs to have low similarity. The loss is
    computed bidirectionally (image-to-text and text-to-image) and averaged.
    
    Args:
        image_features: Image embeddings of shape (batch_size, embed_dim)
        text_features: Text embeddings of shape (batch_size, embed_dim)
        temperature: Temperature parameter for scaling logits
        
    Returns:
        Contrastive loss scalar tensor
    """
    # Normalize features to unit sphere
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    
    # Compute cosine similarity matrix
    logits = torch.matmul(image_features, text_features.t()) / temperature
    
    # Labels for contrastive learning (diagonal should be positive pairs)
    batch_size = image_features.shape[0]
    labels = torch.arange(batch_size, device=image_features.device)
    
    # Cross-entropy loss for both directions
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)
    
    # Average the two directions
    loss = (loss_i2t + loss_t2i) / 2
    
    return loss


def hard_negative_contrastive_loss(image_features: torch.Tensor,
                                 text_features: torch.Tensor,
                                 temperature: float = 0.07,
                                 hard_negative_ratio: float = 0.5) -> torch.Tensor:
    """Compute contrastive loss with hard negative mining.
    
    This variant focuses on the hardest negative examples to improve
    training efficiency and model performance.
    
    Args:
        image_features: Image embeddings of shape (batch_size, embed_dim)
        text_features: Text embeddings of shape (batch_size, embed_dim)
        temperature: Temperature parameter for scaling logits
        hard_negative_ratio: Ratio of hard negatives to use
        
    Returns:
        Hard negative contrastive loss scalar tensor
    """
    # Normalize features
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    
    # Compute similarity matrix
    logits = torch.matmul(image_features, text_features.t()) / temperature
    
    batch_size = image_features.shape[0]
    labels = torch.arange(batch_size, device=image_features.device)
    
    # Image-to-text loss with hard negatives
    loss_i2t = 0
    for i in range(batch_size):
        # Get similarities for current image
        similarities = logits[i]
        
        # Remove positive pair (diagonal)
        mask = torch.ones_like(similarities, dtype=torch.bool)
        mask[i] = False
        
        # Get hard negatives (highest similarities among negatives)
        num_hard_negatives = int((batch_size - 1) * hard_negative_ratio)
        if num_hard_negatives > 0:
            hard_negatives = similarities[mask].topk(num_hard_negatives).indices
            hard_negatives = torch.where(mask)[0][hard_negatives]
        else:
            hard_negatives = torch.where(mask)[0]
        
        # Create logits with only positive and hard negatives
        selected_indices = torch.cat([torch.tensor([i], device=image_features.device), hard_negatives])
        selected_logits = similarities[selected_indices]
        selected_labels = torch.zeros(1, dtype=torch.long, device=image_features.device)
        
        loss_i2t += F.cross_entropy(selected_logits.unsqueeze(0), selected_labels)
    
    loss_i2t /= batch_size
    
    # Text-to-image loss (similar process)
    loss_t2i = 0
    for i in range(batch_size):
        similarities = logits[:, i]
        
        mask = torch.ones_like(similarities, dtype=torch.bool)
        mask[i] = False
        
        num_hard_negatives = int((batch_size - 1) * hard_negative_ratio)
        if num_hard_negatives > 0:
            hard_negatives = similarities[mask].topk(num_hard_negatives).indices
            hard_negatives = torch.where(mask)[0][hard_negatives]
        else:
            hard_negatives = torch.where(mask)[0]
        
        selected_indices = torch.cat([torch.tensor([i], device=image_features.device), hard_negatives])
        selected_logits = similarities[selected_indices]
        selected_labels = torch.zeros(1, dtype=torch.long, device=image_features.device)
        
        loss_t2i += F.cross_entropy(selected_logits.unsqueeze(0), selected_labels)
    
    loss_t2i /= batch_size
    
    return (loss_i2t + loss_t2i) / 2


def focal_contrastive_loss(image_features: torch.Tensor,
                          text_features: torch.Tensor,
                          temperature: float = 0.07,
                          alpha: float = 1.0,
                          gamma: float = 2.0) -> torch.Tensor:
    """Compute focal contrastive loss for handling hard examples.
    
    This loss focuses more on hard-to-classify examples by down-weighting
    easy examples and up-weighting hard examples.
    
    Args:
        image_features: Image embeddings of shape (batch_size, embed_dim)
        text_features: Text embeddings of shape (batch_size, embed_dim)
        temperature: Temperature parameter for scaling logits
        alpha: Weighting factor for positive/negative examples
        gamma: Focusing parameter (higher gamma focuses more on hard examples)
        
    Returns:
        Focal contrastive loss scalar tensor
    """
    # Normalize features
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    
    # Compute similarity matrix
    logits = torch.matmul(image_features, text_features.t()) / temperature
    
    batch_size = image_features.shape[0]
    labels = torch.arange(batch_size, device=image_features.device)
    
    # Compute probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Image-to-text focal loss
    focal_loss_i2t = 0
    for i in range(batch_size):
        # Positive probability
        pos_prob = probs[i, i]
        
        # Focal weight
        focal_weight = alpha * (1 - pos_prob) ** gamma
        
        # Cross-entropy loss
        ce_loss = F.cross_entropy(logits[i:i+1], labels[i:i+1])
        
        focal_loss_i2t += focal_weight * ce_loss
    
    focal_loss_i2t /= batch_size
    
    # Text-to-image focal loss
    focal_loss_t2i = 0
    for i in range(batch_size):
        pos_prob = probs[i, i]
        focal_weight = alpha * (1 - pos_prob) ** gamma
        ce_loss = F.cross_entropy(logits[:, i:i+1].t(), labels[i:i+1])
        focal_loss_t2i += focal_weight * ce_loss
    
    focal_loss_t2i /= batch_size
    
    return (focal_loss_i2t + focal_loss_t2i) / 2


class ContrastiveLoss(nn.Module):
    """PyTorch module wrapper for contrastive loss with different variants."""
    
    def __init__(self, loss_type: str = "standard", temperature: float = 0.07, **kwargs):
        """Initialize contrastive loss module.
        
        Args:
            loss_type: Type of contrastive loss ("standard", "hard_negative", "focal")
            temperature: Temperature parameter for scaling logits
            **kwargs: Additional arguments for specific loss types
        """
        super().__init__()
        self.loss_type = loss_type
        self.temperature = temperature
        self.kwargs = kwargs
    
    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss.
        
        Args:
            image_features: Image embeddings
            text_features: Text embeddings
            
        Returns:
            Contrastive loss
        """
        # print(f"Loss type: {self.loss_type}")
        if self.loss_type == "standard":
            return contrastive_loss(image_features, text_features, self.temperature)
        elif self.loss_type == "hard_negative":
            return hard_negative_contrastive_loss(
                image_features, text_features, self.temperature, **self.kwargs
            )
        elif self.loss_type == "focal":
            return focal_contrastive_loss(
                image_features, text_features, self.temperature, **self.kwargs
            )
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


def compute_similarity_matrix(image_features: torch.Tensor, 
                            text_features: torch.Tensor) -> torch.Tensor:
    """Compute cosine similarity matrix between image and text features.
    
    Args:
        image_features: Image embeddings of shape (batch_size, embed_dim)
        text_features: Text embeddings of shape (batch_size, embed_dim)
        
    Returns:
        Similarity matrix of shape (batch_size, batch_size)
    """
    # Normalize features
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    
    # Compute cosine similarity
    similarity_matrix = torch.matmul(image_features, text_features.t())
    
    return similarity_matrix


def compute_retrieval_metrics(similarity_matrix: torch.Tensor, 
                            k_values: list = [1, 5, 10, 100]) -> dict:
    """Compute retrieval metrics from similarity matrix.
    
    Args:
        similarity_matrix: Similarity matrix of shape (batch_size, batch_size)
        k_values: List of k values for top-k accuracy
        
    Returns:
        Dictionary with retrieval metrics
    """
    batch_size = similarity_matrix.shape[0]
    labels = torch.arange(batch_size, device=similarity_matrix.device)
    
    # Get top-k predictions
    top_k_indices = similarity_matrix.topk(max(k_values), dim=-1).indices
    
    metrics = {}
    for k in k_values:
        if k <= batch_size:
            # Check if correct label is in top-k
            correct = (top_k_indices[:, :k] == labels.unsqueeze(1)).any(dim=1)
            accuracy = correct.float().mean().item()
            metrics[f"top_{k}_accuracy"] = accuracy
        else:
            metrics[f"top_{k}_accuracy"] = 1.0  # All correct if k >= batch_size
    
    return metrics
