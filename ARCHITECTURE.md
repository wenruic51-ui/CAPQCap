# Architecture

## Overview

OFCap generates image captions by combining global scene understanding with local object-level details.

```
┌─────────────┐
│    Image    │
└──────┬──────┘
       │
   ┌───┴────┐
   │        │
┌──▼──┐  ┌─▼──────┐
│CLIP │  │ Detic  │
└──┬──┘  └─┬──────┘
   │        │
   │[1,512] │[N,512]
   └───┬────┘
       │
┌──────▼──────────┐
│ Feature Fusion* │
└──────┬──────────┘
       │[4,512]
┌──────▼──────────┐
│  Q-Former*      │
└──────┬──────────┘
       │[M,768]
┌──────▼──────────┐
│ Prefix Mapper   │
└──────┬──────────┘
       │[40,768]
┌──────▼──────────┐
│  GPT-2 (frozen) │
└──────┬──────────┘
       │
    Caption

* Simplified in public version
```

## Components

### 1. Feature Extraction (Preprocessing)

**Global Features**
- Model: CLIP ViT-L/14
- Output: [1, 512] per image
- Purpose: Scene context

**Local Features**
- Model: Detic/DETR
- Output: [N, 512] (N objects)
- Purpose: Object details

### 2. Feature Fusion ⚠️

**Public version**: Simple concatenation
**Full version**: Multi-head cross-attention (proprietary)

```python
# Simplified public implementation
def fusion(global_feat, local_feats):
    local_avg = local_feats.mean(0)
    return torch.cat([global_feat, local_avg], dim=-1)
```

### 3. Q-Former Bridge ⚠️

**Public version**: Linear transformation
**Full version**: Query-based cross-attention with 20 learnable queries (proprietary)

Inspired by BLIP-2, compresses visual features efficiently.

### 4. Prefix Mapper

Uses Performer-based transformer to map compressed features to GPT-2 embedding space.

- Input: Fused features
- Output: [40, 768] visual prefix
- Architecture: 8-layer Performer

### 5. GPT-2 Decoder

Pre-trained GPT-2 (frozen) generates captions conditioned on visual prefix.

## Training

```python
# Forward pass
fused = feature_fusion(global_feat, local_feats)
compressed = qformer(fused)
prefix = prefix_mapper(compressed)
text_embed = gpt2.embed(tokens)
full_embed = concat(prefix, text_embed)
output = gpt2(full_embed)
```

**Loss**: Cross-entropy on text tokens (prefix tokens ignored)

**Optimizer**: AdamW with linear warmup + cosine decay

## Inference

**Public version**: Basic beam search

**Full version**: Advanced decoding with length control, diversity promotion, and quality filtering (proprietary)

## Model Size

| Component | Parameters |
|-----------|------------|
| Feature Fusion | ~1M |
| Q-Former | ~2M |
| Prefix Mapper | ~20M |
| GPT-2 (frozen) | 124M |
| **Total trainable** | **~23M** |

## Design Choices

**Why freeze GPT-2?**
- Reduces trainable parameters by 80%+
- Maintains pre-trained language knowledge
- Faster training convergence

**Why prefix approach?**
- Modular design (easy to swap visual encoders)
- Efficient training
- Competitive performance

**Why Q-Former?**
- Efficient feature compression
- Handles variable number of objects
- Cross-modal alignment

## Proprietary Components

The following use advanced implementations (abstracted in public version):

1. **Feature Fusion**: Multi-head attention vs. simple concatenation
2. **Q-Former**: 20-query cross-attention vs. linear transform
3. **Decoding**: Length-aware beam search vs. basic search

## Performance Gap

- Public version: ~70-80% of full performance
- Main differences: Fusion quality, Q-Former capacity, decoding sophistication

## References

- **CLIP** (Radford et al., 2021)
- **BLIP-2** (Li et al., 2023)
- **Performer** (Choromanski et al., 2020)

For full architecture details, contact: your.email@institution.edu
