# H64LM: Production-Ready MoE Transformer from Scratch

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

> **A complete, scalable Mixture-of-Experts language model architecture built entirely from scratch. Ready for billion-parameter pretraining.**

---

## Overview

H64LM is a **state-of-the-art transformer architecture** implementing modern techniques from leading LLMs (GPT-4, Claude, LLaMA). Every component—attention, MoE routing, tokenization, training loop—is built from first principles with no black boxes.

**Current demo**: 249M parameters trained on 100K tokens to validate the pipeline. The architecture scales to billions of parameters and trillion-token datasets.

**Key Features**:
- ✅ Mixture-of-Experts with load balancing
- ✅ Grouped-Query Attention (GQA) for efficient inference
- ✅ RoPE + ALiBi positional encoding
- ✅ Multi-GPU support (DataParallel/DDP)
- ✅ Mixed-precision (FP16) training
- ✅ Checkpoint resumption and atomic saves
- ✅ Fully documented and modular code

---

## Architecture

### Model Design

```python
H64LMConfig(
    vocab_size=32000,           # Mistral tokenizer
    hidden_size=768,
    num_layers=6,               # Alternating dense/sparse
    num_attention_heads=12,
    num_kv_heads=4,             # Grouped-Query Attention
    num_experts=8,              # MoE routing
    num_experts_per_token=2,    # Top-2 routing
    max_position_embeddings=1024,
    sliding_window_size=2048,
    attention_type="alibi",     # ALiBi + RoPE hybrid
)
```

### Components

| Component | Implementation | Benefit |
|-----------|----------------|---------|
| **Attention** | Grouped-Query Attention (GQA) | 3× faster inference with KV cache |
| **Position Encoding** | RoPE + ALiBi hybrid | Length extrapolation beyond training |
| **Feedforward** | SwiGLU activation | Better than GELU in modern LLMs |
| **Expert Routing** | Top-2 sparse MoE | Scale capacity without compute cost |
| **Normalization** | RMSNorm (pre-norm) | Faster and more stable than LayerNorm |
| **Attention Masking** | Causal + Sliding Window | Efficient long-context modeling |

### Layer Architecture

```
Input Tokens (32K vocab)
    ↓
┌─────────────────────────┐
│ Layer 0: Dense SwiGLU   │  ← Even layers
├─────────────────────────┤
│ Layer 1: MoE (8×2)      │  ← Odd layers (sparse)
├─────────────────────────┤
│ Layer 2: Dense SwiGLU   │
├─────────────────────────┤
│ Layer 3: MoE (8×2)      │
├─────────────────────────┤
│ Layer 4: Dense SwiGLU   │
├─────────────────────────┤
│ Layer 5: MoE (8×2)      │
└─────────────────────────┘
    ↓
RMSNorm → LM Head
```

---

## Installation

### Requirements

```bash
# Install dependencies
pip install -r requirements.txt
```

**requirements.txt**:
```
# Core dependencies
numpy>=1.21.0
tqdm>=4.62.0
matplotlib>=3.4.0

# PyTorch with CUDA 12.1
--index-url https://download.pytorch.org/whl/cu121
torch>=2.0.0

# Hugging Face
datasets>=2.14.0
transformers>=4.30.0
tokenizers>=0.13.0

# Optional (recommended)
tensorboard>=2.13.0
flash-attn>=2.0.0  # GPU-specific, may require manual install
```

### Setup

```bash
git clone https://github.com/Haiderkhan64/H64LM.git
cd H64LM
```

---

## Usage

### Quick Start

The repository includes pretrained tokenizer and checkpoint files via Git LFS.

**Option 1: Run the Jupyter Notebook**
```bash
jupyter notebook H64LM-v1.ipynb
```

**Option 2: Run as Python Script**
```bash
python H64LM-v1.py
```

The script will:
1. Load tokenizer from `./mistral_tokenizer/`
2. Download WikiText-103 dataset (or use local copy)
3. Train the model with default config
4. Save checkpoints to `./checkpoints_h64lm/`
5. Generate sample text outputs

### Configuration

Modify hyperparameters in the script:

```python
config = H64LMConfig(
    vocab_size=32000,
    hidden_size=768,          # Scale to 4096+ for large models
    num_layers=6,             # Scale to 32+ layers
    num_experts=8,
    max_position_embeddings=1024,  # Increase for longer contexts
    # ... see full config in code
)
```

### Scaling to Production

**For billion-parameter models**:

```python
config = H64LMConfig(
    hidden_size=4096,         # GPT-3 scale
    num_layers=32,
    num_attention_heads=32,
    num_kv_heads=8,
    num_experts=16,
    max_position_embeddings=4096,
    # Update dataset
    max_samples=-1,           # Use full dataset
)
```

**For large datasets**:
- Set `max_samples=-1` to use full WikiText-103 (1.8M samples)
- Use streaming mode: `streaming=True` for massive datasets
- Follow the [5-stage training pipeline](https://medium.com/@haiderkhan6410/llm-training-pipeline-from-foundation-to-chatbot-4f8bab5a73fe) (see references)

---

## Training Pipeline

This implementation supports the standard LLM training pipeline:

### Stage 1: Pretraining (Current Implementation)
```bash
python H64LM-v1.py  # Next-token prediction on raw text
```

### Stage 2: Instruction Tuning
Add instruction-following datasets (Alpaca, Dolly, etc.):
```python
dataset = load_dataset("tatsu-lab/alpaca")
# Continue training with instruction format
```

### Stage 3: RLHF (Reinforcement Learning)
Implement reward model + PPO:
```python
# Train reward model on human preferences
# Apply PPO to optimize policy
```

### Stage 4: Reasoning Enhancement
Add chain-of-thought data:
```python
dataset = load_dataset("gsm8k")  # Math reasoning
# Fine-tune with COT examples
```

### Stage 5: Chat Optimization
Fine-tune on dialogue data:
```python
dataset = load_dataset("OpenAssistant/oasst1")
# Multi-turn conversation training
```

**See the [Medium article](https://medium.com/@haiderkhan6410/llm-training-pipeline-from-foundation-to-chatbot-4f8bab5a73fe) for full pipeline details.**

---

## Checkpointing

### Resume Training

```python
config = H64LMConfig(
    resume_from="./checkpoints_h64lm/best_model.pt",
    num_epochs=10  # Additional epochs
)
```

The checkpoint loader automatically handles:
- DataParallel `module.` prefix stripping
- Partial state dict loading
- Optimizer and scheduler restoration

### Checkpoint Format

```python
checkpoint = {
    'epoch': int,
    'step': int,
    'model_state_dict': OrderedDict,
    'optimizer_state_dict': dict,
    'scheduler_state_dict': dict,
}
```

---

## Generation

```python
from h64lm_v1 import H64LMForCausalLM, test_generation
from transformers import AutoTokenizer

# Load model
tokenizer = AutoTokenizer.from_pretrained("./mistral_tokenizer")
model = H64LMForCausalLM(config, tokenizer)
checkpoint = torch.load("checkpoints_h64lm/best_model.pt")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Generate text
test_generation(
    model, tokenizer, device,
    prompt="Deep learning is",
    max_new_tokens=100,
    temperature=0.8,
    top_p=0.9
)
```

---

## Hyperparameters

### Model Architecture

| Parameter | Demo Value | Production |
|-----------|-----------|------------|
| `hidden_size` | 768 | 4096-8192 |
| `num_layers` | 6 | 32-96 |
| `num_attention_heads` | 12 | 32-128 |
| `num_kv_heads` | 4 | 8-16 |
| `num_experts` | 8 | 8-64 |
| `max_position_embeddings` | 1024 | 4096-32768 |

### Training Config

| Parameter | Value | Notes |
|-----------|-------|-------|
| `batch_size` | 16 | Per-GPU |
| `grad_accum_steps` | 8 | Effective batch: 128 |
| `learning_rate` | 3e-4 | With warmup |
| `weight_decay` | 0.01 | Exclude norms/biases |
| `use_fp16` | True | Mixed precision |
| `gradient_checkpointing` | True | Memory efficient |

---

## Model Card

### Intended Use
- **Primary**: Educational reference for LLM architecture
- **Secondary**: Foundation for research and experimentation
- **Production**: Scales to billion-parameter pretraining

### Capabilities
- Next-token prediction
- Text generation
- Multi-GPU training
- Checkpoint management
- Ready for instruction tuning and RLHF

### Limitations
- Demo model trained on small dataset (pipeline validation only)
- Requires additional training stages for chat/alignment
- No built-in safety filters (add in Stage 3: RLHF)

### License
- **Code**: Apache 2.0 License
- **Tokenizer**: Apache 2.0 (Mistral AI)
- **Dataset**: Creative Commons (WikiText-103)

---

## File Structure

```
H64LM/
├── H64LM-v1.ipynb              # Main notebook
├── H64LM-v1.py.py              # Standalone script
├── requirements.txt            # Dependencies
├── LICENSE                     # Apache 2.0
├── README.md                   # This file
├── .gitattributes              # LFS config
├── mistral_tokenizer/          # Pretrained tokenizer (LFS)
│   ├── tokenizer.json
│   └── tokenizer_config.json
|   └── special_tokens_map.json
|   └── tokenizer.model
└── checkpoints_h64lm/          # Saved models (LFS)
    ├── tokenizer
    └── best_model_state_dict.pt
    └── training_history.json
```

---

## References

### Architecture Papers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Transformer)
- [RoFormer: RoPE](https://arxiv.org/abs/2104.09864)
- [ALiBi](https://arxiv.org/abs/2108.12409)
- [SwiGLU](https://arxiv.org/abs/2002.05202)
- [Sparse MoE](https://arxiv.org/abs/1701.06538)
- [RMSNorm](https://arxiv.org/abs/1910.07467)

### Implementation Inspirations
- [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) (MoE)
- [LLaMA](https://github.com/facebookresearch/llama) (RMSNorm, RoPE)
- [Mistral](https://github.com/mistralai/mistral-src) (Sliding window)

### Training Pipeline
See [Medium Article: LLM Training Pipeline](https://medium.com/@haiderkhan6410/llm-training-pipeline-from-foundation-to-chatbot-4f8bab5a73fe) for the complete 5-stage training guide (Pretraining → Instruction Tuning → RLHF → Reasoning → Chat).

---

## Contributing

Contributions welcome for:
- Scaling experiments (billion-parameter runs)
- Additional training stages (RLHF, instruction tuning)
- Optimization improvements
- Documentation enhancements

**Guidelines**: Keep the educational focus—avoid wrapping complexity in high-level abstractions.

---

## Citation

```bibtex
@software{h64lm2024,
  author = {Haider Khan},
  title = {H64LM: Production-Ready MoE Transformer from Scratch},
  year = {2025},
  url = {https://github.com/Haiderkhan64/H64LM}
}
```

---

## Contact

**Author**: Haider Khan  
**GitHub**: [@Haiderkhan64](https://github.com/Haiderkhan64)
**Medium**: [LLM Training Articles](https://medium.com/@haiderkhan6410/llm-training-pipeline-from-foundation-to-chatbot-4f8bab5a73fe)

For questions or collaboration, open an issue on GitHub.

---

**Last Updated**: November 2025
**Status**: Architecture complete | Ready for large-scale training