"""
Dataset classes and data augmentation for the Image-Caption Alignment project.

This module contains the enhanced dataset class with data augmentation
and improved sampling strategies for better model training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR100
from torchvision.transforms import (
    Compose, ToTensor, Normalize, RandomHorizontalFlip, 
    ColorJitter, RandomRotation, RandomAffine
)
import random
import numpy as np
from typing import Tuple, List, Optional, Callable
from config import DataConfig


class CIFAR100PairedWithCaption(Dataset):
    """Enhanced dataset for paired CIFAR-100 images with captions.
    
    This dataset creates pairs of CIFAR-100 images and generates corresponding
    captions. It includes data augmentation and improved sampling strategies.
    
    Returns:
        - stacked_image: torch.Tensor, shape (C, H, 2*W)
        - caption: str, e.g. "the photo on the left is apple and the photo on the right is bus"
        - labels: tuple of ints (label_left, label_right)
    """
    
    def __init__(self, root: str = "./data", train: bool = True, 
                 transform: Optional[Callable] = None, download: bool = True,
                 different_class_prob: float = 0.5, config: Optional[DataConfig] = None):
        """Initialize the dataset.
        
        Args:
            root: Root directory for CIFAR-100 dataset
            train: Whether to use training set
            transform: Optional transform to apply to images
            download: Whether to download the dataset
            different_class_prob: Probability of sampling different classes
            config: Data configuration object
        """
        self.cifar_dataset = CIFAR100(
            root=root,
            train=train,
            transform=None,  # We'll apply transforms manually
            download=download
        )
        self.transform = transform
        self.class_names = self.cifar_dataset.classes
        self.different_class_prob = different_class_prob
        self.config = config
        
        # Create class indices for efficient sampling
        self.class_to_indices = {}
        for idx, (_, label) in enumerate(self.cifar_dataset):
            if label not in self.class_to_indices:
                self.class_to_indices[label] = []
            self.class_to_indices[label].append(idx)
    
    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.cifar_dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, Tuple[int, int]]:
        """Get a paired image sample with caption.
        
        Args:
            idx: Index of the first image
            
        Returns:
            Tuple of (stacked_image, caption, (label_left, label_right))
        """
        # Get first image
        img1, label1 = self.cifar_dataset[idx]
        
        # Sample second image
        if random.random() < self.different_class_prob:
            # Sample from different class
            different_classes = [c for c in range(len(self.class_names)) if c != label1]
            label2 = random.choice(different_classes)
            idx2 = random.choice(self.class_to_indices[label2])
        else:
            # Sample from same class (or any class)
            idx2 = random.randint(0, len(self.cifar_dataset) - 1)
            _, label2 = self.cifar_dataset[idx2]
        
        img2, _ = self.cifar_dataset[idx2]
        
        # Apply transforms if provided
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        else:
            # Convert PIL to tensor if no transform
            img1 = ToTensor()(img1)
            img2 = ToTensor()(img2)
        
        # Stack horizontally: shape (C, H, 2*W)
        stacked_img = torch.cat([img1, img2], dim=2)
        
        # Generate caption
        class1 = self.class_names[label1]
        class2 = self.class_names[label2]
        caption = f"the photo on the left is {class1} and the photo on the right is {class2}"
        
        return stacked_img, caption, (label1, label2)
    
    def get_class_distribution(self) -> dict:
        """Get the distribution of classes in the dataset.
        
        Returns:
            Dictionary mapping class names to counts
        """
        class_counts = {}
        for _, label in self.cifar_dataset:
            class_name = self.class_names[label]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        return class_counts


def create_augmentation_transforms(config: DataConfig) -> Compose:
    """Create data augmentation transforms based on configuration.
    
    Args:
        config: Data configuration object
        
    Returns:
        Compose transform with augmentation
    """
    transforms = [ToTensor()]
    
    if config.use_augmentation:
        # Random horizontal flip
        if config.horizontal_flip_prob > 0:
            transforms.append(RandomHorizontalFlip(p=config.horizontal_flip_prob))
        
        # Color jitter
        if config.color_jitter_strength > 0:
            transforms.append(ColorJitter(
                brightness=config.color_jitter_strength,
                contrast=config.color_jitter_strength,
                saturation=config.color_jitter_strength,
                hue=config.color_jitter_strength * 0.1  # Smaller hue change
            ))
        
        # Random rotation
        if config.rotation_degrees > 0:
            transforms.append(RandomRotation(
                degrees=config.rotation_degrees,
                interpolation=torch.nn.functional.InterpolationMode.BILINEAR
            ))
    
    # Normalization
    transforms.append(Normalize(mean=config.mean, std=config.std))
    
    return Compose(transforms)


def worker_init_fn(worker_id):
    """Worker initialization function for DataLoader multiprocessing."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloaders(config, dataset_class=CIFAR100PairedWithCaption) -> Tuple[DataLoader, DataLoader, List[str]]:
    """Create training and validation dataloaders.
    
    Args:
        config: Configuration object
        dataset_class: Dataset class to use
        
    Returns:
        Tuple of (train_loader, val_loader, class_names)
    """
    # Create augmentation transforms
    transform = create_augmentation_transforms(config.data)
    
    # Create datasets
    train_dataset = dataset_class(
        root=config.data.data_root,
        train=True,
        transform=transform,
        download=config.data.download,
        different_class_prob=config.training.different_class_prob,
        config=config.data
    )
    
    val_dataset = dataset_class(
        root=config.data.data_root,
        train=True,  # Use training set for validation as per requirements
        transform=transform,
        download=False,
        different_class_prob=0.5,  # Fixed for validation
        config=config.data
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        worker_init_fn=worker_init_fn,
        drop_last=True  # Ensure consistent batch sizes
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        worker_init_fn=worker_init_fn,
        drop_last=False
    )
    
    return train_loader, val_loader, train_dataset.class_names


def get_dataset_info(dataset: CIFAR100PairedWithCaption) -> str:
    """Get information about the dataset.
    
    Args:
        dataset: Dataset object
        
    Returns:
        Formatted string with dataset information
    """
    info = f"Dataset Information:\n"
    info += f"  Total samples: {len(dataset):,}\n"
    info += f"  Number of classes: {len(dataset.class_names)}\n"
    info += f"  Different class probability: {dataset.different_class_prob:.2f}\n"
    
    # Class distribution
    class_dist = dataset.get_class_distribution()
    info += f"  Samples per class: {min(class_dist.values())}-{max(class_dist.values())}\n"
    
    return info


def visualize_samples(dataset: CIFAR100PairedWithCaption, num_samples: int = 4) -> None:
    """Visualize sample pairs from the dataset.
    
    Args:
        dataset: Dataset object
        num_samples: Number of samples to visualize
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 3, 6))
    
    for i in range(num_samples):
        img, caption, (label1, label2) = dataset[i]
        
        # Convert tensor to numpy for visualization
        img_np = img.permute(1, 2, 0).numpy()
        
        # Denormalize if normalized
        if img_np.min() < 0:
            img_np = (img_np + 1) / 2
            img_np = np.clip(img_np, 0, 1)
        
        # Plot image
        axes[0, i].imshow(img_np)
        axes[0, i].set_title(f"Sample {i+1}")
        axes[0, i].axis('off')
        
        # Plot caption
        axes[1, i].text(0.1, 0.5, caption, fontsize=8, wrap=True)
        axes[1, i].set_xlim(0, 1)
        axes[1, i].set_ylim(0, 1)
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()
