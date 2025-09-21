#!/usr/bin/env python3
"""
Evaluation script for the Image-Caption Alignment project.

This script provides a command-line interface for evaluating trained models
on the image-caption alignment task.
"""

import argparse
import os
import json
import torch
from transformers import CLIPTextModel, CLIPTokenizer

from config import Config, get_default_config
from dataset import create_dataloaders
from models import create_image_encoder
from evaluation import evaluate_model_comprehensive
from utils import set_seed, get_device, setup_logging


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate Image-Caption Alignment Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model parameters
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--config_path", 
        type=str, 
        default=None,
        help="Path to model configuration file"
    )
    
    # Data parameters
    parser.add_argument("--data_root", type=str, default="./data", help="Data root directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    
    # System parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    
    # Evaluation options
    parser.add_argument("--comprehensive", action="store_true", help="Run comprehensive evaluation")
    parser.add_argument("--output_file", type=str, default=None, help="Output file for results")
    
    # Logging
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    return parser.parse_args()


def load_model_and_config(args):
    """Load model and configuration."""
    # Load configuration
    if args.config_path and os.path.exists(args.config_path):
        config = Config.from_json(args.config_path)
    else:
        # Try to load from checkpoint
        try:
            # First try with weights_only=False for compatibility
            checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False)
        except Exception as e:
            print(f"Warning: Could not load checkpoint with weights_only=False: {e}")
            print("Trying to load with weights_only=True...")
            try:
                checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=True)
            except Exception as e2:
                print(f"Error: Could not load checkpoint: {e2}")
                print("Using default configuration instead.")
                checkpoint = {}
        
        if 'config' in checkpoint:
            config_dict = checkpoint['config']
            config = Config()
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                elif hasattr(config.model, key):
                    setattr(config.model, key, value)
                elif hasattr(config.training, key):
                    setattr(config.training, key, value)
                elif hasattr(config.data, key):
                    setattr(config.data, key, value)
        else:
            # Use default configuration
            config = get_default_config()
    
    # Override with command line arguments
    config.data.data_root = args.data_root
    config.training.batch_size = args.batch_size
    config.seed = args.seed
    config.device = args.device
    
    return config


def load_models(config, device, model_path):
    """Load image and text encoders."""
    # Create image encoder
    image_encoder = create_image_encoder(config)
    
    # Load trained weights
    try:
        # First try with weights_only=False for compatibility
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"Warning: Could not load checkpoint with weights_only=False: {e}")
        print("Trying to load with weights_only=True...")
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        except Exception as e2:
            print(f"Error: Could not load checkpoint: {e2}")
            raise RuntimeError(f"Failed to load model from {model_path}")
    
    if 'model_state_dict' in checkpoint:
        image_encoder.load_state_dict(checkpoint['model_state_dict'])
    else:
        image_encoder.load_state_dict(checkpoint)
    
    image_encoder.to(device)
    image_encoder.eval()
    
    # Load text encoder
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder.to(device)
    text_encoder.eval()
    
    return image_encoder, text_encoder, tokenizer


def main():
    """Main evaluation function."""
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration
    config = load_model_and_config(args)
    
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
    
    logger.info(f"Evaluating model: {args.model_path}")
    logger.info(f"Device: {device}")
    
    # Load models
    logger.info("Loading models...")
    image_encoder, text_encoder, tokenizer = load_models(config, device, args.model_path)
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, class_names = create_dataloaders(config)
    
    logger.info(f"Evaluation samples: {len(val_loader.dataset)}")
    logger.info(f"Number of classes: {len(class_names)}")
    
    
    # Basic evaluation
    from evaluation import evaluate_topk
    logger.info("Running basic evaluation...")
    top1_acc, top10_acc, top100_acc = evaluate_topk(
        image_encoder, text_encoder, tokenizer,
        val_loader, class_names, device
    )
    metrics = {
        'top1_accuracy': top1_acc,
        'top10_accuracy': top10_acc,
        'top100_accuracy': top100_acc
    }
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    for metric, value in metrics.items():
        if 'accuracy' in metric or 'top_' in metric:
            print(f"{metric}: {value:.2f}%")
        else:
            print(f"{metric}: {value:.4f}")
    print("="*60)
    
    # Save results
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Results saved to {args.output_file}")
    
    # Save results with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output = f"evaluation_results_{timestamp}.json"
    with open(default_output, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Results also saved to {default_output}")
    
    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()
