"""
Training module for the Image-Caption Alignment project.

This module contains the enhanced training loop with improved optimization
strategies, learning rate scheduling, and monitoring capabilities.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm
import time
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, Optional, Tuple

from models import EnhancedImageTransformerEncoder
from losses import ContrastiveLoss
from utils import (
    AverageMeter, EarlyStopping, get_learning_rate, 
    adjust_learning_rate, cosine_annealing_lr, warmup_lr,
    format_time, setup_logging
)
from config import Config


class LearningCurvePlotter:
    """Class to handle learning curve plotting during training."""
    
    def __init__(self, output_dir: str, save_plots: bool = True):
        """Initialize the plotter.
        
        Args:
            output_dir: Directory to save plots
            save_plots: Whether to save plots to disk
        """
        self.output_dir = output_dir
        self.save_plots = save_plots
        self.fig = None
        self.axes = None
        self.setup_plot()
    
    def setup_plot(self):
        """Setup the matplotlib figure and axes."""
        plt.style.use('default')
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
        
        # Configure subplots
        self.axes[0, 0].set_title('Loss Curves')
        self.axes[0, 0].set_xlabel('Epoch')
        self.axes[0, 0].set_ylabel('Loss')
        self.axes[0, 0].grid(True, alpha=0.3)
        
        self.axes[0, 1].set_title('Top-k Accuracy')
        self.axes[0, 1].set_xlabel('Epoch')
        self.axes[0, 1].set_ylabel('Accuracy (%)')
        self.axes[0, 1].grid(True, alpha=0.3)
        
        self.axes[1, 0].set_title('Learning Rate Schedule')
        self.axes[1, 0].set_xlabel('Epoch')
        self.axes[1, 0].set_ylabel('Learning Rate')
        self.axes[1, 0].set_yscale('log')
        self.axes[1, 0].grid(True, alpha=0.3)
        
        self.axes[1, 1].set_title('Training Progress Summary')
        self.axes[1, 1].axis('off')
        
        plt.tight_layout()
    
    def update_plots(self, training_history: Dict, learning_rates: list, 
                    current_epoch: int, total_epochs: int):
        """Update the learning curves with current data.
        
        Args:
            training_history: Dictionary containing training metrics
            learning_rates: List of learning rates per epoch
            current_epoch: Current epoch number
            total_epochs: Total number of epochs
        """
        epochs = list(range(1, current_epoch + 1))
        
        # Clear previous plots
        for ax in self.axes.flat:
            ax.clear()
        
        # Re-setup titles and labels
        self.axes[0, 0].set_title('Loss Curves')
        self.axes[0, 0].set_xlabel('Epoch')
        self.axes[0, 0].set_ylabel('Loss')
        self.axes[0, 0].grid(True, alpha=0.3)
        
        self.axes[0, 1].set_title('Top-k Accuracy')
        self.axes[0, 1].set_xlabel('Epoch')
        self.axes[0, 1].set_ylabel('Accuracy (%)')
        self.axes[0, 1].grid(True, alpha=0.3)
        
        self.axes[1, 0].set_title('Learning Rate Schedule')
        self.axes[1, 0].set_xlabel('Epoch')
        self.axes[1, 0].set_ylabel('Learning Rate')
        self.axes[1, 0].set_yscale('log')
        self.axes[1, 0].grid(True, alpha=0.3)
        
        self.axes[1, 1].set_title('Training Progress Summary')
        self.axes[1, 1].axis('off')
        
        # Plot loss curves
        if training_history['train_loss']:
            self.axes[0, 0].plot(epochs, training_history['train_loss'], 
                                'b-', label='Training Loss', linewidth=2, marker='o', markersize=4)
        if training_history['val_loss']:
            self.axes[0, 0].plot(epochs, training_history['val_loss'], 
                                'r-', label='Validation Loss', linewidth=2, marker='s', markersize=4)
        self.axes[0, 0].legend()
        
        # Plot accuracy curves
        if training_history['top1_acc']:
            self.axes[0, 1].plot(epochs, training_history['top1_acc'], 
                                'g-', label='Top-1', linewidth=2, marker='o', markersize=4)
        if training_history['top10_acc']:
            self.axes[0, 1].plot(epochs, training_history['top10_acc'], 
                                'orange', label='Top-10', linewidth=2, marker='s', markersize=4)
        if training_history['top100_acc']:
            self.axes[0, 1].plot(epochs, training_history['top100_acc'], 
                                'purple', label='Top-100', linewidth=2, marker='^', markersize=4)
        self.axes[0, 1].legend()
        
        # Plot learning rate schedule
        if learning_rates:
            self.axes[1, 0].plot(epochs, learning_rates, 
                                'b-', linewidth=2, marker='o', markersize=4)
        
        # Add progress summary text
        if training_history['train_loss']:
            current_train_loss = training_history['train_loss'][-1]
            current_val_loss = training_history['val_loss'][-1] if training_history['val_loss'] else 0
            current_top1 = training_history['top1_acc'][-1] if training_history['top1_acc'] else 0
            current_top10 = training_history['top10_acc'][-1] if training_history['top10_acc'] else 0
            current_top100 = training_history['top100_acc'][-1] if training_history['top100_acc'] else 0
            current_lr = learning_rates[-1] if learning_rates else 0
            
            summary_text = f"""
            Epoch: {current_epoch}/{total_epochs}
            
            Current Metrics:
            • Train Loss: {current_train_loss:.4f}
            • Val Loss: {current_val_loss:.4f}
            • Top-1 Acc: {current_top1:.2f}%
            • Top-10 Acc: {current_top10:.2f}%
            • Top-100 Acc: {current_top100:.2f}%
            • Learning Rate: {current_lr:.2e}
            
            Best Metrics:
            • Best Val Loss: {min(training_history['val_loss']) if training_history['val_loss'] else 'N/A':.4f}
            • Best Top-1: {max(training_history['top1_acc']) if training_history['top1_acc'] else 'N/A':.2f}%
            • Best Top-10: {max(training_history['top10_acc']) if training_history['top10_acc'] else 'N/A':.2f}%
            • Best Top-100: {max(training_history['top100_acc']) if training_history['top100_acc'] else 'N/A':.2f}%
            """
            
            self.axes[1, 1].text(0.05, 0.95, summary_text, transform=self.axes[1, 1].transAxes,
                                fontsize=10, verticalalignment='top', fontfamily='monospace',
                                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot if enabled
        if self.save_plots:
            plot_path = os.path.join(self.output_dir, 'learning_curves.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        
        # # Show plot (non-blocking)
        # plt.show(block=False)
        # plt.pause(0.1)
    
    def save_final_plot(self):
        """Save the final learning curve plot."""
        if self.save_plots:
            plot_path = os.path.join(self.output_dir, 'final_learning_curves.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Final learning curves saved to: {plot_path}")
    
    def close(self):
        """Close the matplotlib figure."""
        if self.fig:
            plt.close(self.fig)


class Trainer:
    """Enhanced trainer class for image-caption alignment model."""
    
    def __init__(self, config: Config, device: torch.device):
        """Initialize trainer.
        
        Args:
            config: Configuration object
            device: Device for training
        """
        self.config = config
        self.device = device
        self.logger = setup_logging()
        
        # Initialize models
        self.image_encoder = None
        self.text_encoder = None
        self.tokenizer = None
        
        # Initialize training components
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'top1_acc': [],
            'top10_acc': [],
            'top100_acc': []
        }
        
        # Learning curve plotting
        self.learning_rates = []
        self.plotter = None
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.training.num_epochs // 4,  # 25% of total epochs
            min_delta=0.001,
            restore_best_weights=True
        )
    
    def setup_models(self) -> None:
        """Setup image and text encoders."""
        self.logger.info("Setting up models...")
        
        # Image encoder
        self.image_encoder = EnhancedImageTransformerEncoder(
            image_size=self.config.model.image_size,
            patch_size=self.config.model.patch_size,
            embed_dim=self.config.model.embed_dim,
            mlp_dim=self.config.model.mlp_dim,
            num_layers=self.config.model.num_layers,
            num_heads=self.config.model.num_heads,
            dropout=self.config.model.dropout,
            activation=self.config.model.activation,
            use_projection_head=self.config.model.use_projection_head,
            projection_dim=self.config.model.projection_dim,
            projection_layers=self.config.model.projection_layers
        ).to(self.device)
        
        # Text encoder (frozen)
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.text_encoder.to(self.device)
        self.text_encoder.eval()  # Freeze text encoder
        
        # Log model information
        self.logger.info(f"Image encoder parameters: {self.image_encoder.get_num_parameters():,}")
        self.logger.info(f"Text encoder parameters: {sum(p.numel() for p in self.text_encoder.parameters()):,}")
        self.logger.info(f"Model size: {self.image_encoder.get_model_size_mb():.2f} MB")
        
        # Check output dimensions
        with torch.no_grad():
            # Test image encoder output dimension
            dummy_image = torch.randn(1, 3, 32, 64).to(self.device)
            image_output = self.image_encoder(dummy_image)
            self.logger.info(f"Image encoder output dimension: {image_output.shape[-1]}")
            self.logger.info(f"Image encoder config - embed_dim: {self.config.model.embed_dim}, use_projection_head: {self.config.model.use_projection_head}")
            
            # Test text encoder output dimension
            dummy_text = ["test caption"]
            text_output = self.encode_text(dummy_text)
            self.logger.info(f"Text encoder output dimension: {text_output.shape[-1]}")
            
            # Check if dimensions match
            if image_output.shape[-1] != text_output.shape[-1]:
                self.logger.error(f"Dimension mismatch! Image: {image_output.shape[-1]}, Text: {text_output.shape[-1]}")
                self.logger.error("This will cause training errors. Check projection head configuration.")
                self.logger.error(f"Expected image dimension: {self.config.model.embed_dim}")
                raise RuntimeError(f"Dimension mismatch: Image {image_output.shape[-1]} vs Text {text_output.shape[-1]}")
            else:
                self.logger.info("✓ Image and text encoder output dimensions match!")
    
    def setup_training(self) -> None:
        """Setup optimizer, scheduler, and loss function."""
        self.logger.info("Setting up training components...")
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.image_encoder.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate scheduler
        self.scheduler = self._get_scheduler()
        
        # Loss function
        self.criterion = ContrastiveLoss(
            temperature=self.config.training.temperature,
            loss_type=self.config.training.loss_type
        )
        
        self.logger.info(f"Optimizer: AdamW (lr={self.config.training.learning_rate})")
        self.logger.info(f"Scheduler: {type(self.scheduler).__name__}")
        self.logger.info(f"Loss: Contrastive Loss (temp={self.config.training.temperature})")
    
    def _get_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Get learning rate scheduler."""
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.training.num_epochs - self.config.training.warmup_epochs,
            eta_min=self.config.training.learning_rate * 0.01
        )
    
    def encode_text(self, texts: list) -> torch.Tensor:
        """Encode text using CLIP text encoder.
        
        Args:
            texts: List of text strings
            
        Returns:
            Text embeddings tensor
        """
        with torch.no_grad():
            inputs = self.tokenizer(
                texts, 
                padding=True, 
                truncation=True, 
                return_tensors="pt"
            )
            inputs = inputs.to(self.device)
            outputs = self.text_encoder(**inputs)
            text_features = outputs.pooler_output
        
        return text_features
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average training loss
        """
        self.image_encoder.train()
        
        losses = AverageMeter()
        progress_bar = tqdm(train_loader, desc=f'Epoch {self.current_epoch + 1}')
        
        for batch_idx, (images, captions, labels) in enumerate(progress_bar):
            # Move to device
            images = images.to(self.device)
            
            # Forward pass
            image_features = self.image_encoder(images)
            text_features = self.encode_text(captions)
            
            # Debug: Check dimensions
            if batch_idx == 0:
                self.logger.info(f"Image features shape: {image_features.shape}")
                self.logger.info(f"Text features shape: {text_features.shape}")
            
            # Compute loss
            loss = self.criterion(image_features, text_features)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.training.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.image_encoder.parameters(),
                    self.config.training.gradient_clip_norm
                )
            
            self.optimizer.step()
            
            # Update metrics
            losses.update(loss.item(), images.size(0))
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{losses.avg:.4f}',
                'LR': f'{get_learning_rate(self.optimizer):.2e}'
            })
        
        return losses.avg
    
    def validate_epoch(self, val_loader: DataLoader) -> float:
        """Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Average validation loss
        """
        self.image_encoder.eval()
        
        losses = AverageMeter()
        
        with torch.no_grad():
            for images, captions, labels in val_loader:
                # Move to device
                images = images.to(self.device)
                
                # Forward pass
                image_features = self.image_encoder(images)
                text_features = self.encode_text(captions)
                
                # Compute loss
                loss = self.criterion(image_features, text_features)
                
                # Update metrics
                losses.update(loss.item(), images.size(0))
        
        return losses.avg
    
    def update_learning_rate(self) -> None:
        """Update learning rate based on scheduler."""
        if self.current_epoch < self.config.training.warmup_epochs:
            # Warmup phase
            lr = warmup_lr(
                self.current_epoch,
                self.config.training.warmup_epochs,
                self.config.training.learning_rate
            )
            adjust_learning_rate(self.optimizer, lr)
        else:
            # Cosine annealing phase
            self.scheduler.step()
    
    def save_checkpoint(self, filepath: str, is_best: bool = False) -> None:
        """Save model checkpoint.
        
        Args:
            filepath: Path to save checkpoint
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.image_encoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'config': self.config.__dict__
        }
        
        torch.save(checkpoint, filepath)
        
        if is_best:
            best_path = filepath.replace('.pth', '_best.pth')
            torch.save(checkpoint, best_path)
            self.logger.info(f"Best model saved to {best_path}")
    
    def load_checkpoint(self, filepath: str) -> None:
        """Load model checkpoint.
        
        Args:
            filepath: Path to checkpoint file
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.image_encoder.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.training_history = checkpoint['training_history']
        
        self.logger.info(f"Checkpoint loaded from {filepath}")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              class_names: list, output_dir: str, enable_plotting: bool = True) -> None:
        """Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            class_names: List of class names for evaluation
            output_dir: Output directory for saving models
            enable_plotting: Whether to show learning curves during training
        """
        self.logger.info("Starting training...")
        self.logger.info(f"Training on {len(train_loader)} batches per epoch")
        self.logger.info(f"Validating on {len(val_loader)} batches per epoch")
        self.logger.info(f"Total epochs: {self.config.training.num_epochs}")
        
        # Initialize learning curve plotter
        if enable_plotting:
            try:
                self.plotter = LearningCurvePlotter(output_dir, save_plots=True)
                self.logger.info("Learning curve plotting enabled")
            except Exception as e:
                self.logger.warning(f"Failed to initialize plotting: {e}")
                self.plotter = None
        else:
            self.plotter = None
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config.training.num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Update learning rate
            self.update_learning_rate()
            
            # Track learning rate
            current_lr = get_learning_rate(self.optimizer)
            self.learning_rates.append(current_lr)
            
            # Training phase
            train_loss = self.train_epoch(train_loader)
            
            # Validation phase
            val_loss = self.validate_epoch(val_loader)
            
            # Evaluate top-k accuracy (if evaluation frequency allows)
            # if hasattr(self.config.training, 'eval_frequency') and (epoch + 1) % self.config.training.eval_frequency == 0:
            self.logger.info("Evaluating top-k accuracy...")
            eval_start_time = time.time()
            top1_acc, top10_acc, top100_acc = self.evaluate_model(val_loader, class_names)
            eval_time = time.time() - eval_start_time
            self.logger.info(f"Evaluation completed in {format_time(eval_time)}")
            
            # Update best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            # Update training history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['top1_acc'].append(top1_acc)
            self.training_history['top10_acc'].append(top10_acc)
            self.training_history['top100_acc'].append(top100_acc)
            
            # Log epoch results
            epoch_time = time.time() - epoch_start_time
            self.logger.info(
                f"Epoch {epoch + 1}/{self.config.training.num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"Top-1: {top1_acc:.2f}%, Top-10: {top10_acc:.2f}%, Top-100: {top100_acc:.2f}%, "
                f"Time: {format_time(epoch_time)}, LR: {get_learning_rate(self.optimizer):.2e}"
            )
            
            # Update learning curves
            if self.plotter is not None:
                try:
                    self.plotter.update_plots(
                        self.training_history, 
                        self.learning_rates, 
                        epoch + 1, 
                        self.config.training.num_epochs
                    )
                    self.plotter.save_final_plot()
                except Exception as e:
                    self.logger.warning(f"Failed to update plots: {e}")
            
            # Save checkpoint
            if is_best:
                checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch + 1}.pth")
                self.save_checkpoint(checkpoint_path, is_best)
            
            # Early stopping check
            if self.early_stopping(val_loss, self.image_encoder):
                self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {format_time(total_time)}")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Save final learning curves
        if self.plotter is not None:
            try:
                self.plotter.save_final_plot()
                self.plotter.close()
            except Exception as e:
                self.logger.warning(f"Failed to save final plot: {e}")
        
        # Print final results
        if self.training_history['top1_acc']:
            best_top1 = max(self.training_history['top1_acc'])
            best_top10 = max(self.training_history['top10_acc'])
            best_top100 = max(self.training_history['top100_acc'])
            
            self.logger.info("=" * 80)
            self.logger.info("TRAINING SUMMARY")
            self.logger.info("=" * 80)
            self.logger.info(f"Best Validation Loss: {self.best_val_loss:.4f}")
            self.logger.info(f"Best Top-1 Accuracy: {best_top1:.2f}%")
            self.logger.info(f"Best Top-10 Accuracy: {best_top10:.2f}%")
            self.logger.info(f"Best Top-100 Accuracy: {best_top100:.2f}%")
            self.logger.info("=" * 80)
            
            # Print training progress table
            self.logger.info("Training Progress:")
            self.logger.info("Epoch | Train Loss | Val Loss | Top-1 | Top-10 | Top-100")
            self.logger.info("-" * 60)
            for i in range(len(self.training_history['train_loss'])):
                epoch = i + 1
                train_loss = self.training_history['train_loss'][i]
                val_loss = self.training_history['val_loss'][i]
                top1 = self.training_history['top1_acc'][i]
                top10 = self.training_history['top10_acc'][i]
                top100 = self.training_history['top100_acc'][i]
                self.logger.info(f"{epoch:5d} | {train_loss:9.4f} | {val_loss:8.4f} | {top1:5.2f} | {top10:6.2f} | {top100:7.2f}")
            self.logger.info("=" * 80)
    
    def evaluate_model(self, val_loader: DataLoader, class_names: list) -> Tuple[float, float, float]:
        """Evaluate model performance.
        
        Args:
            val_loader: Validation data loader
            class_names: List of class names
            
        Returns:
            Tuple of (top1_accuracy, top10_accuracy, top100_accuracy)
        """
        from evaluation import evaluate_topk
        
        # Set models to eval mode
        self.image_encoder.eval()
        self.text_encoder.eval()
        
        # Run evaluation
        top1_acc, top10_acc, top100_acc = evaluate_topk(
            self.image_encoder, self.text_encoder, self.tokenizer,
            val_loader, class_names, self.device
        )
        
        return top1_acc, top10_acc, top100_acc
