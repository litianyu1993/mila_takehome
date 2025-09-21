"""
Evaluation module for the Image-Caption Alignment project.

This module contains the evaluation functions for assessing model performance
on image-caption retrieval tasks.
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import List, Tuple
from transformers import CLIPTextModel, CLIPTokenizer


def encode_text(text_to_encode: List[str], text_encoder: CLIPTextModel, 
                tokenizer: CLIPTokenizer, device: torch.device) -> torch.Tensor:
    """Encode text using CLIP text encoder.
    
    Args:
        text_to_encode: List of text strings to encode
        text_encoder: CLIP text encoder model
        tokenizer: CLIP tokenizer
        device: Device to run on
        
    Returns:
        Text embeddings tensor of shape (batch_size, embed_dim)
    """
    inputs = tokenizer(
        text_to_encode, 
        padding=True, 
        truncation=True, 
        return_tensors="pt"
    )
    inputs = inputs.to(device)
    
    with torch.no_grad():
        outputs = text_encoder(**inputs)
        text_features = outputs.pooler_output
    
    return text_features


# DO NOT modify the evaluation function
def evaluate_topk(image_encoder, text_encoder, tokenizer, 
                val_loader, class_names, device):
    """
    Zero-shot evaluation: For each image, predict the closest caption from the prompts
    Returns top-1, top-10, top-100 recall.
    """
    image_encoder.eval()
    text_encoder.eval()
    # Prepare all combinations
    prompts = []
    for class_name_left in class_names: # 100x100
        for class_name_right in class_names:
            prompts.append(f"the photo on the left is {class_name_left} and the photo on the right is {class_name_right}")
    assert len(prompts) == len(set(prompts)), "Prompts must be unique!"
    # map prompt to index
    prompt_to_id = {}
    for i, prompt in enumerate(prompts):
        prompt_to_id[prompt] = i

    with torch.no_grad():
        # batch prompts to reduce peak memory
        batch_prompts = 256
        txt_features = []
        for index in range(0, len(prompts), batch_prompts):
            current_prompts = prompts[index: index + batch_prompts]
            text_features = encode_text(current_prompts, text_encoder, tokenizer, device)  
            # normalize 
            text_features = F.normalize(text_features, dim=-1)
            # aggregate 
            txt_features.append(text_features)
        # stack
        text_features = torch.concatenate(txt_features, dim=0)
        assert text_features.size(0) == len(prompts)

        top1, top10, top100, total = 0, 0, 0, 0

        for images, captions, labels in tqdm(val_loader, desc="Topk eval"):
            images = images.to(device)
            # Encode images
            image_features = image_encoder(images)
            image_features = F.normalize(image_features, dim=-1)

            # Compute similarity (batch_size, num_classes)
            logits = image_features @ text_features.T

            # Top-1 and Top-5 predictions
            top1_pred = logits.argmax(dim=-1)
            top10_pred = logits.topk(10, dim=-1).indices
            top100_pred = logits.topk(100, dim=-1).indices

            # get_labels
            idx_relative_to_prompt = [] 
            for cap in captions:
                idx_relative_to_prompt.append(prompt_to_id[cap])
            idx_relative_to_prompt = torch.tensor(idx_relative_to_prompt, device=device)

            top1 += (top1_pred == idx_relative_to_prompt).sum().item()
            top10 += sum([idx_relative_to_prompt[i] in top10_pred[i] for i in range(idx_relative_to_prompt.size(0))])
            top100 += sum([idx_relative_to_prompt[i] in top100_pred[i] for i in range(idx_relative_to_prompt.size(0))])

            total += idx_relative_to_prompt.size(0)

    top1_acc = 100 * top1 / total
    top10_acc = 100 * top10 / total
    top100_acc = 100 * top100 / total
    print(f"Zero-shot Top-1 Acc: {top1_acc:.2f}%, Top-10 Acc: {top10_acc:.2f}% Top-100 Acc: {top100_acc:.2f}%")
    return top1_acc, top10_acc, top100_acc
