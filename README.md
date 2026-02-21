# OFCap: Object-Focused Image Captioning

[![License](https://img.shields.io/badge/license-Proprietary-red.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)

> ⚠️ **Simplified Public Version**: Core architecture is proprietary. This release is for academic research and demonstration purposes.

An image captioning model combining global scene understanding with local object-level details for generating natural language descriptions.

## Features

- **Multi-scale visual features** (global + local objects)
- **Q-Former architecture** for efficient feature compression
- **Frozen GPT-2** for language generation
- **Prefix-based training** for computational efficiency

## Installation

```bash
git clone https://github.com/yourusername/ofcap.git
cd ofcap
pip install -r requirements.txt
```

**Requirements**: Python 3.8+, PyTorch 2.0+, transformers 4.30+

## Quick Start

### Training

```bash
python train.py \
    --data data.pkl.gz \
    --out_dir ./checkpoints \
    --batch_size 32 \
    --epochs 15 \
    --lr 1e-5
```

### Inference

```bash
python predict.py \
    --model_path checkpoints/best.pt \
    --data_path test_data.pkl.gz \
    --num_images 100
```

## Data Format

```python
{
    'image_features': {
        image_id: {
            'global_features': Tensor,  # [1, 512]
            'local_features': Tensor    # [N, 512]
        }
    },
    'samples': [
        {'image_id': int, 'caption': str}
    ]
}
```

Extract features using CLIP (global) and Detic/DETR (local objects).

## Architecture

```
Image → Global + Local Features
    → Feature Fusion
    → Q-Former Bridge
    → Visual Prefix (40 tokens)
    → GPT-2 Decoder
    → Caption
```

**Note**: Feature Fusion, Q-Former, and advanced decoding strategies use simplified implementations in this public version.

## Citation

```bibtex
@article{ofcap2024,
  title={OFCap: Object-Focused Image Captioning},
  author={Your Name},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

## License

Proprietary - Academic use permitted with citation. Commercial use requires licensing. See [LICENSE](LICENSE).

## Contact

- **Academic collaboration**: your.email@institution.edu
- **Commercial licensing**: business@institution.edu
- **Issues**: [GitHub Issues](https://github.com/yourusername/ofcap/issues)

## Acknowledgments

Built upon [CLIP](https://github.com/openai/CLIP), [GPT-2](https://github.com/openai/gpt-2), [BLIP-2](https://github.com/salesforce/LAVIS)
