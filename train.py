#!/usr/bin/env python3
"""
Training script for the Image-Caption Alignment project.

This script provides a command-line interface for training the enhanced
vision-language model with various configuration options.
"""

import argparse
import os
import json
import torch
from datetime import datetime

from config import Config, get_default_config, get_enhanced_config
from dataset import create_dataloaders
from training import Trainer
from utils import set_seed, get_device, create_output_dir, setup_logging
from models import create_image_encoder


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Image-Caption Alignment Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration options
    parser.add_argument(
        "--config", 
        type=str, 
        default=None,
        help="Path to configuration JSON file"
    )
    parser.add_argument(
        "--config_type",
        type=str,
        choices=["default", "enhanced"],
        default="enhanced",
        help="Type of configuration to use"
    )
    
    # Model parameters
    parser.add_argument("--embed_dim", type=int, default=512, help="Embedding dimension")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--patch_size", type=int, default=8, help="Patch size for ViT")
    parser.add_argument("--use_projection_head", action="store_true", help="Use projection head")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--temperature", type=float, default=0.07, help="Temperature for contrastive loss")
    parser.add_argument("--loss_type", type=str, default="standard", help="Loss type, standard, hard_negative, focal")
    parser.add_argument("--eval_frequency", type=int, default=1, help="Evaluate every N epochs (1 = every epoch)")
    
    # Data parameters
    parser.add_argument("--data_root", type=str, default="./data", help="Data root directory")
    parser.add_argument("--use_augmentation", action="store_true", help="Use data augmentation")
    parser.add_argument("--different_class_prob", type=float, default=0.5, help="Probability of sampling different classes")
    
    # System parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--eval_only", action="store_true", help="Only run evaluation")
    
    # Logging and visualization
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--no_plotting", action="store_true", help="Disable learning curve plotting")
    
    return parser.parse_args()


def load_config(args):
    """Load configuration from arguments and files."""
    if args.config:
        # Load from JSON file
        config = Config.from_json(args.config)
    elif args.config_type == "enhanced":
        # Use enhanced configuration
        config = get_enhanced_config()
    else:
        # Use default configuration
        config = get_default_config()
    
    # Override with command line arguments
    config.update(
        # Model parameters
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        patch_size=args.patch_size,
        use_projection_head=args.use_projection_head,
        
        # Training parameters
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        eval_frequency=args.eval_frequency,
        loss_type=args.loss_type,
        
        # Data parameters
        data_root=args.data_root,
        use_augmentation=args.use_augmentation,
        different_class_prob=args.different_class_prob,
        
        # System parameters
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir
    )
    
    return config


def main():
    """Main training function."""
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args)
    
    # Debug: Print configuration
    print(f"DEBUG: embed_dim = {config.model.embed_dim}")
    print(f"DEBUG: use_projection_head = {config.model.use_projection_head}")
    print(f"DEBUG: projection_dim = {config.model.projection_dim}")
    
    # Windows compatibility fix
    import platform
    if platform.system() == "Windows":
        config.training.num_workers = 0
        logger = setup_logging(level=getattr(__import__('logging'), args.log_level))
        logger.info("Windows detected: Setting num_workers=0 for compatibility")
    
    # Set random seed
    set_seed(config.seed)
    
    # Get device
    device = get_device(config.device)
    
    # Setup logging (if not already done)
    if platform.system() != "Windows":
        logger = setup_logging(level=getattr(__import__('logging'), args.log_level))
    
    # Create output directory
    output_dir = create_output_dir(config.output_dir)
    logger.info(f"Output directory: {output_dir}")
    
    # Save configuration
    config.to_json(os.path.join(output_dir, "config.json"))
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, class_names = create_dataloaders(config)
    
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    logger.info(f"Number of classes: {len(class_names)}")
    
    # Initialize trainer
    trainer = Trainer(config, device)
    
    # Setup models and training components
    trainer.setup_models()
    trainer.setup_training()
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    if not args.eval_only:
        # Train the model
        logger.info("Starting training...")
        enable_plotting = not args.no_plotting
        trainer.train(train_loader, val_loader, class_names, output_dir, enable_plotting)
        
        # Save final model
        final_model_path = os.path.join(output_dir, "final_model.pth")
        trainer.save_checkpoint(final_model_path)
        logger.info(f"Final model saved to {final_model_path}")
    
    # Evaluate the model
    logger.info("Evaluating model...")
    metrics = trainer.evaluate_model(val_loader, class_names)
    
    # Save evaluation results
    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Evaluation results saved to {results_path}")
    
    # Print final results
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
    logger.info(f"Top-1 Accuracy: {metrics['top1_accuracy']:.2f}%")
    logger.info(f"Top-10 Accuracy: {metrics['top10_accuracy']:.2f}%")
    logger.info(f"Top-100 Accuracy: {metrics['top100_accuracy']:.2f}%")


if __name__ == "__main__":
    main()
