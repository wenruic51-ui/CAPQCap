#!/usr/bin/env python3
"""
OFCap: Object-Focused Image Captioning Model
Inference Script

This is a simplified version for public release.
Core decoding strategies are abstracted for proprietary reasons.
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import random
import os
import json
import argparse
from typing import List, Dict
from datetime import datetime

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✅ Random seed set to: {seed}")

class SimpleDecoder:
    """
    Simplified Decoder for OFCap Model
    
    NOTE: Advanced decoding strategies are proprietary.
    This provides basic beam search functionality.
    """
    
    def __init__(self, model, dataset, device):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.model.eval()
    
    def decode(self, global_features, local_features_list, 
               max_length=80, beam_size=5, temperature=1.0):
        """
        Simple beam search decoding
        
        NOTE: Production version uses advanced decoding strategies
        including length control, diversity promotion, and quality filtering.
        """
        self.model.eval()
        
        with torch.no_grad():
            # Ensure correct input format
            if global_features.dim() == 1:
                global_features = global_features.unsqueeze(0).unsqueeze(0)
            elif global_features.dim() == 2:
                if global_features.shape[0] != 1:
                    global_features = global_features.unsqueeze(0)
            
            if not isinstance(local_features_list, list):
                local_features_list = [local_features_list]
            
            # Initialize beams
            beams = [([], 0.0)]
            
            for step in range(max_length):
                candidates = []
                
                for tokens, score in beams:
                    # Prepare input
                    if tokens:
                        current_tokens = torch.tensor([tokens], dtype=torch.long, device=self.device)
                    else:
                        current_tokens = torch.zeros(1, 1, dtype=torch.long, device=self.device)
                    
                    seq_len = current_tokens.shape[1]
                    total_len = self.dataset.prefix_length + seq_len
                    current_mask = torch.ones(1, total_len, device=self.device)
                    
                    # Forward pass
                    outputs, _ = self.model(
                        tokens=current_tokens,
                        mask=current_mask,
                        global_features=global_features,
                        local_features_list=local_features_list
                    )
                    
                    # Get next token probabilities
                    next_token_logits = outputs.logits[0, -1, :] / temperature
                    next_token_probs = F.softmax(next_token_logits, dim=-1)
                    
                    # Get top-k candidates
                    top_k = min(beam_size * 2, next_token_probs.shape[0])
                    top_probs, top_indices = torch.topk(next_token_probs, top_k)
                    
                    for i in range(top_k):
                        token = top_indices[i].item()
                        prob = top_probs[i].item()
                        
                        # Skip padding token
                        if token == 0:
                            continue
                        
                        new_tokens = tokens + [token]
                        new_score = score + np.log(prob + 1e-10)
                        
                        # Check for end of sequence
                        token_text = self.dataset.tokenizer.decode([token])
                        if '.' in token_text and len(new_tokens) > 5:
                            candidates.append((new_tokens, new_score, True))
                        else:
                            candidates.append((new_tokens, new_score, False))
                
                if not candidates:
                    break
                
                # Select top beams
                completed = [c for c in candidates if c[2]]
                ongoing = [c for c in candidates if not c[2]]
                
                if completed and len(completed) >= beam_size // 2:
                    beams = [(tokens, score) for tokens, score, _ in completed[:beam_size]]
                    break
                
                all_candidates = completed + ongoing
                all_candidates.sort(key=lambda x: x[1], reverse=True)
                beams = [(tokens, score) for tokens, score, _ in all_candidates[:beam_size]]
            
            # Get best caption
            if beams and beams[0][0]:
                best_tokens = beams[0][0]
                caption = self.dataset.tokenizer.decode(best_tokens, skip_special_tokens=True)
                return caption.strip()
            
            return ""

def get_unique_image_ids(dataset):
    """Get all unique image IDs from dataset"""
    image_ids = set()
    for sample in dataset.samples:
        image_ids.add(sample['image_id'])
    return sorted(list(image_ids))

def get_sample_by_image_id(dataset, image_id):
    """Get sample index by image ID"""
    for idx, sample in enumerate(dataset.samples):
        if sample['image_id'] == image_id:
            return idx
    return None

def batch_predict(model, dataset, device, image_ids: List[int], output_dir: str):
    """
    Batch prediction for image captioning
    
    Args:
        model: OFCap model
        dataset: Image caption dataset
        device: torch device
        image_ids: List of image IDs to predict
        output_dir: Output directory for results
    """
    set_seed(42)
    model.eval()
    model = model.to(device)
    
    # Initialize decoder
    decoder = SimpleDecoder(model, dataset, device)
    
    print(f"🚀 Starting batch prediction")
    print(f"   Number of images: {len(image_ids)}")
    print(f"   Output directory: {output_dir}")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    predictions = []
    
    for i, image_id in enumerate(image_ids):
        print(f"   [{i+1}/{len(image_ids)}] Processing image {image_id}")
        
        try:
            # Get sample
            sample_idx = get_sample_by_image_id(dataset, image_id)
            if sample_idx is None:
                print(f"      ⚠️  Image {image_id} not found")
                continue
            
            # Get features
            sample_data = dataset[sample_idx]
            tokens, mask, global_feat, local_feat, num_objects, target_length = sample_data
            
            global_feat = global_feat.to(device)
            local_feat = local_feat.to(device)
            
            if local_feat.shape[0] > 0:
                local_features_list = [local_feat]
            else:
                local_features_list = [torch.zeros(0, 512, device=device)]
            
            # Generate caption
            start_time = time.time()
            caption = decoder.decode(
                global_feat, 
                local_features_list,
                max_length=80,
                beam_size=5,
                temperature=0.9
            )
            decode_time = time.time() - start_time
            
            if caption:
                print(f"      Caption: {caption}")
                print(f"      Time: {decode_time:.3f}s")
                
                predictions.append({
                    "image_id": int(image_id),
                    "caption": caption
                })
            
        except Exception as e:
            print(f"      ❌ Error: {e}")
            continue
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"predictions_{timestamp}.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Prediction completed!")
    print(f"   Total predictions: {len(predictions)}")
    print(f"   Output file: {output_file}")
    
    return predictions

def main():
    parser = argparse.ArgumentParser(description='OFCap Inference')
    
    parser.add_argument('--data_path', type=str, required=True, help='Dataset path')
    parser.add_argument('--model_path', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--output_dir', type=str, default='./predictions', help='Output directory')
    parser.add_argument('--gpt2_path', type=str, default='gpt2', help='GPT-2 model path')
    parser.add_argument('--num_images', type=int, default=100, help='Number of images to predict')
    parser.add_argument('--qformer_num_queries', type=int, default=20, help='Q-Former queries')
    
    args = parser.parse_args()
    
    print("="*80)
    print("OFCap: Object-Focused Image Captioning Model")
    print("Inference Script (Simplified Public Version)")
    print("="*80)
    print("⚠️  Note: Advanced decoding strategies are proprietary")
    print("   This version provides basic beam search functionality")
    print("="*80)
    
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")
    
    # Import model and dataset from training script
    try:
        from train import OFCapModel, ImageCaptionDataset
        print("✅ Model components imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import: {e}")
        print("💡 Make sure train.py is in the same directory")
        return
    
    # Load dataset
    print("📍 Loading dataset...")
    dataset = ImageCaptionDataset(
        data_path=args.data_path,
        prefix_length=40,
        gpt2_type=args.gpt2_path,
        normalize_prefix=False,
        max_local_features=6
    )
    print(f"✅ Dataset loaded: {len(dataset)} samples")
    
    # Initialize model
    print("📍 Initializing model...")
    model = OFCapModel(
        prefix_length=40,
        gpt2_model_path=args.gpt2_path,
        qformer_num_queries=args.qformer_num_queries
    )
    
    # Load checkpoint
    if os.path.exists(args.model_path):
        print(f"📍 Loading checkpoint: {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            if 'epoch' in checkpoint:
                print(f"   Epoch: {checkpoint['epoch']}")
        else:
            model.load_state_dict(checkpoint, strict=False)
        
        print("✅ Checkpoint loaded successfully")
    else:
        print(f"⚠️  Checkpoint not found: {args.model_path}")
        print("⚠️  Using random initialization")
    
    # Get image IDs
    print("📍 Getting unique image IDs...")
    unique_image_ids = get_unique_image_ids(dataset)
    print(f"✅ Found {len(unique_image_ids)} unique images")
    
    # Sample random images
    target_image_ids = random.sample(
        unique_image_ids, 
        min(args.num_images, len(unique_image_ids))
    )
    
    # Run prediction
    predictions = batch_predict(
        model, dataset, device, target_image_ids, args.output_dir
    )
    
    print("\n🎉 Inference completed!")

if __name__ == "__main__":
    main()
