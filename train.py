#!/usr/bin/env python3
"""
OFCap: Object-Focused Image Captioning Model
Training Script

This is a simplified version for public release.
Core model architecture is abstracted for proprietary reasons.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
import os
import pickle
import argparse
import json
import gzip
import numpy as np
import random

# For reproducibility
def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"✅ Random seed set to: {seed}")

class ImageCaptionDataset(Dataset):
    """
    Dataset for image captioning with global and local features
    
    Expected data format:
    {
        'image_features': {
            image_id: {
                'global_features': tensor of shape [1, feature_dim],
                'local_features': tensor of shape [num_objects, feature_dim]
            }
        },
        'samples': [
            {'image_id': id, 'caption': text},
            ...
        ]
    }
    """
    
    def __init__(self, data_path: str, prefix_length: int, 
                 gpt2_type: str = "gpt2",
                 normalize_prefix=False, 
                 max_local_features=6):
        
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix
        self.max_local_features = max_local_features
        
        print(f"📍 Loading dataset from: {data_path}")
        
        # Load data
        if data_path.endswith('.gz'):
            with gzip.open(data_path, 'rb') as f:
                all_data = pickle.load(f)
        else:
            with open(data_path, 'rb') as f:
                all_data = pickle.load(f)
        
        self.image_features = all_data['image_features']
        self.samples = all_data['samples']
        
        print(f"   Images: {len(self.image_features)}")
        print(f"   Samples: {len(self.samples)}")
        
        # Initialize tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Preprocess captions
        self._prepare_captions()
        
    def _prepare_captions(self):
        """Tokenize all captions"""
        print("📍 Tokenizing captions...")
        self.captions_tokens = []
        max_seq_len = 0
        
        for sample in self.samples:
            caption = sample['caption']
            tokens = torch.tensor(self.tokenizer.encode(caption), dtype=torch.int64)
            self.captions_tokens.append(tokens)
            max_seq_len = max(max_seq_len, tokens.shape[0])
        
        # Set reasonable max length
        all_len = torch.tensor([len(tokens) for tokens in self.captions_tokens]).float()
        self.max_seq_len = min(int(all_len.mean() + all_len.std() * 2), int(all_len.max()))
        print(f"   Max sequence length: {self.max_seq_len}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_id = sample['image_id']
        
        # Get features
        image_feature = self.image_features[image_id]
        global_features = image_feature['global_features'].float()
        local_features = image_feature['local_features'].float()
        
        # Normalize if needed
        if self.normalize_prefix:
            global_features = F.normalize(global_features, p=2, dim=-1)
            if local_features.shape[0] > 0:
                local_features = F.normalize(local_features, p=2, dim=-1)
        
        # Limit local features
        if local_features.shape[0] > self.max_local_features:
            local_features = local_features[:self.max_local_features]
        
        # Get tokenized caption
        tokens = self.captions_tokens[idx]
        
        # Pad tokens
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
        
        # Create mask
        mask = tokens.ge(0)
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)
        
        num_objects = min(local_features.shape[0], self.max_local_features)
        target_length = len(tokens[tokens > 0])
        
        return tokens, mask, global_features, local_features, num_objects, target_length

def collate_fn(batch):
    """Custom collate function for dynamic local features"""
    tokens_list, mask_list, global_list, local_list, num_obj_list, target_len_list = [], [], [], [], [], []
    
    for item in batch:
        tokens, mask, global_feat, local_feat, num_obj, target_len = item
        tokens_list.append(tokens)
        mask_list.append(mask)
        global_list.append(global_feat)
        local_list.append(local_feat)
        num_obj_list.append(num_obj)
        target_len_list.append(target_len)
    
    tokens = torch.stack(tokens_list)
    mask = torch.stack(mask_list)
    global_features = torch.stack(global_list)
    num_objects = torch.tensor(num_obj_list)
    target_lengths = torch.tensor(target_len_list)
    
    return tokens, mask, global_features, local_list, num_objects, target_lengths

# ============================================================================
# PROPRIETARY MODEL ARCHITECTURE - ABSTRACTED
# ============================================================================
# The core model architecture is proprietary and has been abstracted.
# This placeholder demonstrates the expected interface.
# 
# Key components (implementation details removed):
# 1. Feature Fusion Module: Combines global and local visual features
# 2. Q-Former Bridge: Transforms visual features to language space
# 3. Prefix Mapper: Projects to GPT-2 embedding space
# ============================================================================

class OFCapModel(nn.Module):
    """
    OFCap Model - Public Interface
    
    Core architecture details are proprietary.
    This is a simplified interface for demonstration.
    """
    
    def __init__(self, prefix_length=40, gpt2_model_path="gpt2", **kwargs):
        super().__init__()
        self.prefix_length = prefix_length
        
        # Load pre-trained GPT-2 (frozen)
        self.gpt = GPT2LMHeadModel.from_pretrained(gpt2_model_path)
        for param in self.gpt.parameters():
            param.requires_grad = False
        
        # Proprietary components (implementations abstracted)
        self._init_proprietary_modules(**kwargs)
        
        print(f"✅ OFCap Model initialized")
        print(f"   Prefix length: {prefix_length}")
        print(f"   Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
    
    def _init_proprietary_modules(self, **kwargs):
        """
        Initialize proprietary model components
        
        NOTE: This is a placeholder. Actual implementation uses:
        - Advanced feature fusion mechanisms
        - Q-Former based cross-modal attention
        - Optimized prefix mapping strategies
        
        For research collaboration or licensing, please contact the authors.
        """
        # Placeholder for proprietary components
        embedding_size = self.gpt.transformer.wte.weight.shape[1]
        
        # Simple linear mapping as placeholder (actual model is more sophisticated)
        self.feature_fusion = nn.Identity()  # Placeholder
        self.prefix_mapper = nn.Linear(512 * 4, prefix_length * embedding_size)  # Simplified
        
        print("   ⚠️  Using simplified architecture (core modules abstracted)")
    
    def forward(self, tokens, mask, global_features, local_features_list, labels=None):
        """
        Forward pass
        
        Args:
            tokens: Text tokens [batch_size, seq_len]
            mask: Attention mask [batch_size, prefix_length + seq_len]
            global_features: Global visual features [batch_size, 1, feature_dim]
            local_features_list: List of local features per sample
            labels: Optional labels for training
        
        Returns:
            outputs: Model outputs
            info: Additional information dictionary
        """
        batch_size = global_features.shape[0]
        
        # Simplified feature processing (actual model uses advanced fusion)
        # This is a placeholder - real implementation is proprietary
        fused_features = torch.cat([
            global_features.squeeze(1),
            torch.stack([lf.mean(0) if lf.shape[0] > 0 else torch.zeros_like(global_features[0, 0]) 
                        for lf in local_features_list])
        ], dim=-1)
        
        # Project to prefix embeddings
        prefix_embeddings = self.prefix_mapper(fused_features).view(
            batch_size, self.prefix_length, -1
        )
        
        # Get text embeddings
        text_embeddings = self.gpt.transformer.wte(tokens)
        
        # Concatenate prefix and text
        full_embeddings = torch.cat([prefix_embeddings, text_embeddings], dim=1)
        
        # Prepare labels
        if labels is not None:
            dummy_labels = torch.zeros(batch_size, self.prefix_length, dtype=torch.long, device=tokens.device)
            full_labels = torch.cat([dummy_labels, tokens], dim=1)
        else:
            full_labels = None
        
        # GPT forward pass
        outputs = self.gpt(inputs_embeds=full_embeddings, labels=full_labels, attention_mask=mask)
        
        info = {
            'architecture': 'ofcap_simplified',
            'note': 'Core architecture is proprietary'
        }
        
        return outputs, info

class OFCapTrainer:
    """Trainer class for OFCap model"""
    
    def __init__(self, model, dataset, args, device):
        self.model = model.to(device)
        self.dataset = dataset
        self.args = args
        self.device = device
        
        # Setup optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            eps=1e-8
        )
        
        # Setup dataloader
        self.dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        # Setup scheduler
        total_steps = args.epochs * len(self.dataloader)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_steps
        )
        
        # Create output directory
        os.makedirs(args.out_dir, exist_ok=True)
        
        print(f"✅ Trainer initialized")
        print(f"   Batch size: {args.batch_size}")
        print(f"   Learning rate: {args.lr}")
        print(f"   Epochs: {args.epochs}")
    
    def train(self):
        """Training loop"""
        print(f"\n🚀 Starting training for {self.args.epochs} epochs")
        
        best_loss = float('inf')
        
        for epoch in range(self.args.epochs):
            self.model.train()
            epoch_loss = 0
            valid_batches = 0
            
            progress = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.args.epochs}")
            
            for batch in progress:
                tokens, mask, global_feat, local_list, num_obj, target_len = batch
                
                # Move to device
                tokens = tokens.to(self.device)
                mask = mask.to(self.device)
                global_feat = global_feat.to(self.device)
                local_list = [lf.to(self.device) for lf in local_list]
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs, info = self.model(tokens, mask, global_feat, local_list, labels=tokens)
                
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()
                
                # Track loss
                epoch_loss += loss.item()
                valid_batches += 1
                
                progress.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Epoch statistics
            avg_loss = epoch_loss / valid_batches
            print(f"\nEpoch {epoch+1} - Average Loss: {avg_loss:.4f}")
            
            # Save checkpoint
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_checkpoint(epoch, 'best')
                print(f"   ✅ New best model saved!")
            
            if (epoch + 1) % 3 == 0:
                self.save_checkpoint(epoch, f'epoch_{epoch+1}')
        
        print(f"\n🎉 Training completed!")
    
    def save_checkpoint(self, epoch, name):
        """Save model checkpoint"""
        save_path = os.path.join(self.args.out_dir, f"{self.args.prefix}_{name}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'args': self.args
        }, save_path)
        print(f"   💾 Checkpoint saved: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="OFCap Model Training")
    
    # Data parameters
    parser.add_argument('--data', required=True, help='Path to training data')
    parser.add_argument('--out_dir', default='./checkpoints', help='Output directory')
    parser.add_argument('--prefix', default='ofcap', help='Model prefix for saving')
    
    # Model parameters
    parser.add_argument('--prefix_length', type=int, default=40, help='Prefix length')
    parser.add_argument('--max_local_features', type=int, default=6, help='Max local features')
    parser.add_argument('--gpt2_model_path', type=str, default='gpt2', help='GPT-2 model path')
    parser.add_argument('--qformer_num_queries', type=int, default=20, help='Q-Former queries')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=15, help='Training epochs')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='Warmup steps')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--normalize_prefix', action='store_true', help='Normalize features')
    parser.add_argument('--deterministic_mode', action='store_true', help='Deterministic mode')
    parser.add_argument('--clip_length', type=int, default=10, help='CLIP sequence length')
    
    args = parser.parse_args()
    
    print("="*80)
    print("OFCap: Object-Focused Image Captioning Model")
    print("Training Script (Simplified Public Version)")
    print("="*80)
    print("⚠️  Note: Core architecture is proprietary and abstracted")
    print("   For research collaboration, please contact the authors")
    print("="*80)
    
    # Set seed
    set_seed(args.seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")
    
    # Load dataset
    dataset = ImageCaptionDataset(
        data_path=args.data,
        prefix_length=args.prefix_length,
        gpt2_type=args.gpt2_model_path,
        normalize_prefix=args.normalize_prefix,
        max_local_features=args.max_local_features
    )
    
    # Initialize model
    model = OFCapModel(
        prefix_length=args.prefix_length,
        gpt2_model_path=args.gpt2_model_path,
        qformer_num_queries=args.qformer_num_queries
    )
    
    # Train
    trainer = OFCapTrainer(model, dataset, args, device)
    trainer.train()

if __name__ == "__main__":
    main()
