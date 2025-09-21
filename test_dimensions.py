#!/usr/bin/env python3
"""
Test script to verify model dimensions are correct.

This script tests the image and text encoder output dimensions
to ensure they match for training.
"""

import torch
from transformers import CLIPTextModel, CLIPTokenizer
from models import create_image_encoder
from config import get_enhanced_config
from utils import get_device, set_seed


def test_dimensions():
    """Test image and text encoder output dimensions."""
    print("Testing model dimensions...")
    
    # Set up
    set_seed(42)
    device = get_device("auto")
    config = get_enhanced_config()
    
    print(f"Device: {device}")
    print(f"Image encoder embed_dim: {config.model.embed_dim}")
    print(f"Projection head enabled: {config.model.use_projection_head}")
    print(f"Projection dimension: {config.model.projection_dim}")
    
    # Create image encoder
    image_encoder = create_image_encoder(config).to(device)
    image_encoder.eval()
    
    # Create text encoder
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder.to(device)
    text_encoder.eval()
    
    # Test image encoder
    dummy_image = torch.randn(2, 3, 32, 64).to(device)
    with torch.no_grad():
        image_features = image_encoder(dummy_image)
        print(f"Image features shape: {image_features.shape}")
        print(f"Image encoder output dimension: {image_features.shape[-1]}")
    
    # Test text encoder
    dummy_texts = ["the photo on the left is apple and the photo on the right is bus",
                   "the photo on the left is car and the photo on the right is dog"]
    
    with torch.no_grad():
        inputs = tokenizer(dummy_texts, padding=True, truncation=True, return_tensors="pt")
        inputs = inputs.to(device)
        outputs = text_encoder(**inputs)
        text_features = outputs.pooler_output
        print(f"Text features shape: {text_features.shape}")
        print(f"Text encoder output dimension: {text_features.shape[-1]}")
    
    # Check compatibility
    if image_features.shape[-1] == text_features.shape[-1]:
        print("✓ SUCCESS: Image and text encoder dimensions match!")
        print(f"Both output {image_features.shape[-1]}-dimensional features")
        
        # Test similarity computation
        similarity = torch.matmul(image_features, text_features.t())
        print(f"Similarity matrix shape: {similarity.shape}")
        print("✓ Similarity computation works!")
        
    else:
        print("✗ ERROR: Dimension mismatch!")
        print(f"Image encoder: {image_features.shape[-1]} dimensions")
        print(f"Text encoder: {text_features.shape[-1]} dimensions")
        print("This will cause training errors.")
        
        # Suggest fix
        if config.model.use_projection_head:
            print(f"Current projection_dim: {config.model.projection_dim}")
            print(f"Should be: {text_features.shape[-1]}")
        else:
            print("Consider enabling projection head to match dimensions")


if __name__ == "__main__":
    test_dimensions()
