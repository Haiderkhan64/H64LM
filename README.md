# H64LM: From-Scratch Mixture-of-Experts Language Model

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **A 249M-parameter Mixture-of-Experts transformer built from scratch to deeply understand modern LLM architecture, training dynamics, and infrastructure.**

---

## 🎯 Project Goals & Motivation

This project was created to:

1. **Master LLM internals** by implementing every component from scratch (no high-level abstractions)
2. **Understand training dynamics** including overfitting, loss curves, and optimization
3. **Validate infrastructure** for multi-GPU training, checkpointing, and model resumption
4. **Build towards SOTA** with a production-ready architecture (currently limited by hardware/data)

**Current Status**: ✅ Infrastructure validated | ⚠️ Severe overfitting observed | 🚧 Requires scaling

---

## 📊 Quick Stats

| Metric | Value |
|--------|-------|
| **Model Size** | 249,720,576 parameters |
| **Architecture** | 6-layer MoE Transformer |
| **Hardware** | 4× NVIDIA L4 GPUs (12-hour limit) |
| **Training Data** | ~100K tokens (WikiText-103 subset) |
| **Final Train Loss** | 1.54 (PPL: 4.65) |
| **Final Val Loss** | 4.32 (PPL: 75.31) |
| **Throughput** | ~10,000 tokens/sec |
| **Training Time** | ~20 epochs × 25 min/epoch |

**⚠️ Overfitting Alert**: The 2.78-point train/val loss gap indicates severe memorization due to the intentionally tiny dataset used for pipeline validation.

---

## 🏗️ Architecture Overview

### Model Configuration

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

### Key Architectural Features

| Component | Implementation | Purpose |
|-----------|----------------|---------|
| **Attention** | Grouped-Query Attention (GQA) | Memory-efficient KV caching |
| **Position Encoding** | RoPE + ALiBi hybrid | Extrapolation beyond training length |
| **Feedforward** | SwiGLU activation | Improved gating vs GELU |
| **MoE Routing** | Top-2 sparse experts | Scale capacity without compute overhead |
| **Normalization** | RMSNorm (pre-norm) | Faster + more stable than LayerNorm |
| **Masking** | Causal + Sliding Window | Efficient long-context modeling |

### Layer Structure

```
┌─────────────────────────────────────┐
│  Token Embeddings (32K vocab)      │
├─────────────────────────────────────┤
│  Layer 0: Dense SwiGLU             │ ← Even layers
│  Layer 1: MoE (8 experts, top-2)   │ ← Odd layers (sparse)
│  Layer 2: Dense SwiGLU             │
│  Layer 3: MoE (8 experts, top-2)   │
│  Layer 4: Dense SwiGLU             │
│  Layer 5: MoE (8 experts, top-2)   │
├─────────────────────────────────────┤
│  Final RMSNorm                      │
│  LM Head (tied embeddings)          │
└─────────────────────────────────────┘
```

---

## 🚀 Quickstart

### Prerequisites

```bash
# Core dependencies
pip install torch>=2.0.0 transformers datasets tokenizers
pip install numpy tqdm matplotlib

# Optional (for acceleration)
pip install flash-attn  # GPU attention optimization
```

### 1. Clone & Setup

```bash
git clone https://github.com/yourusername/h64lm.git
cd h64lm
mkdir -p data/wikitext-103 data/mistral_tokenizer checkpoints
```

### 2. Download Resources

```bash
# Option A: Use Kaggle datasets (as in original)
kaggle datasets download -d haiderkhan6410/wikitext-103-zip -p ./data --unzip
kaggle datasets download -d haiderkhan6410/mistral-tokenizer -p ./data --unzip

# Option B: Use Hugging Face (internet required)
# Dataset will auto-download on first run
# Tokenizer: huggingface-cli download mistralai/Mistral-7B-v0.1 tokenizer.json
```

### 3. Train from Scratch

```bash
# Full 20-epoch training (reproduces original results)
python h64lm_v1.py

# Expected output:
# - Training loss: ~1.5 (severe overfitting)
# - Validation loss: ~4.3
# - Checkpoints saved to ./checkpoints/
```

### 4. Resume from Checkpoint

```python
# Modify config in h64lm_v1.py:
config = H64LMConfig(
    resume_from="./checkpoints/best_model.pt",
    num_epochs=1  # Additional epochs
)

# Or use the saved checkpoint:
config.resume_from = "/path/to/best_model_state_dict.pt"
```

### 5. Generate Text

```python
from h64lm_v1 import H64LMForCausalLM, test_generation
from transformers import AutoTokenizer

# Load model & tokenizer
tokenizer = AutoTokenizer.from_pretrained("./data/mistral_tokenizer")
model = H64LMForCausalLM(config, tokenizer)
model.load_state_dict(torch.load("checkpoints/best_model.pt")["model_state_dict"])
model.eval()

# Generate
test_generation(
    model, tokenizer, device,
    prompt="Deep learning is",
    max_new_tokens=100
)
```

---

## 📈 Training Results & Analysis

### Loss Curves (20 Epochs)

| Epoch | Train Loss | Val Loss | Val PPL | Status |
|-------|------------|----------|---------|--------|
| 1 | 7.49 | 5.72 | 303.7 | Baseline |
| 5 | 3.95 | 4.00 | 54.7 | Learning |
| 10 | 3.01 | 3.70 | 40.5 | **Best Val** |
| 15 | 2.16 | 3.96 | 52.6 | Overfitting begins |
| 20 | **1.54** | **4.32** | **75.3** | Severe overfitting |

![Training Curves](checkpoints/training_curves.png)

### Key Observations

1. **Validation plateau at epoch 10**: Model stopped generalizing after 3.70 val loss
2. **Training continues to drop**: Memorization of the small dataset
3. **PPL divergence**: Train PPL 4.65 vs Val PPL 75.31 (16× gap)
4. **Throughput**: Consistent ~10K tokens/sec on 4× L4 GPUs

### Generated Samples (Epoch 20)

**Prompt**: `"Deep learning is a subfield of machine learning that"`

**Output**:
```
Deep learning is a subfield of machine learning that comes to the same
month after being given a further information regarding the following a
further information can beaches were also , Manning the next door @-'s
of course of the rightful and more recent development...
```

**Analysis**:
- ✅ Grammatically coherent sentences
- ✅ Proper punctuation and structure
- ❌ Semantic drift after ~20 tokens
- ❌ Repetitive phrases ("further information", "the following")
- ❌ Context loss (beaches? Manning? Unrelated to prompt)

**Root Cause**: Severe overfitting on WikiText-103's specific writing style patterns.

---

## 💾 Checkpoint Management

### Saved Checkpoints

```bash
checkpoints/
├── best_model.pt              # Lowest val loss (epoch 10: 3.70)
├── checkpoint_epoch0_final.pt  # Per-epoch snapshots
├── checkpoint_epoch19_final.pt
└── training_history.json      # Metrics log
```

### Checkpoint Format

```python
{
    'epoch': int,                    # Training epoch number
    'step': int,                     # Global step count
    'model_state_dict': OrderedDict, # Model parameters (CPU)
    'optimizer_state_dict': dict,    # AdamW state
    'scheduler_state_dict': dict,    # LR scheduler state
}
```

### Loading Checkpoints

```python
from h64lm_v1 import load_checkpoint_for_resume

# Automatically handles DataParallel prefix stripping
start_epoch, start_step = load_checkpoint_for_resume(
    model, optimizer, scheduler,
    checkpoint_path="checkpoints/best_model.pt",
    device=device
)
```

**Features**:
- ✅ Automatic `module.` prefix stripping (DataParallel compatibility)
- ✅ Graceful handling of partial state dicts
- ✅ Missing/unexpected key warnings (non-fatal)
- ✅ Atomic saves via temp files (crash-safe)

---

## 🔬 Reproducibility Checklist

### Current Implementation

✅ **Deterministic seeds**: `torch.manual_seed(42)`, `np.random.seed(42)`  
✅ **Pinned dependencies**: PyTorch 2.0+, Transformers, Datasets  
✅ **Checkpoint atomicity**: Temp file writes + rename  
✅ **Multi-GPU support**: DataParallel & DDP-ready  
✅ **Mixed precision**: FP16 training with GradScaler  
✅ **Gradient checkpointing**: Memory-efficient training  

### Missing for Full Reproducibility

❌ **requirements.txt**: Not provided (manual pip install needed)  
❌ **Docker container**: No containerized environment  
❌ **Data versioning**: No hash verification for WikiText-103  
❌ **Hyperparameter logging**: No MLflow/Weights & Biases integration  
❌ **Deterministic cuDNN**: `torch.backends.cudnn.deterministic=True` not set  

### Recommended Setup

```bash
# 1. Create requirements.txt
cat > requirements.txt << EOF
torch>=2.0.0
transformers>=4.30.0
datasets>=2.10.0
tokenizers>=0.13.0
numpy>=1.24.0
tqdm>=4.65.0
matplotlib>=3.7.0
EOF

# 2. Install in virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt

# 3. Set deterministic flags
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -c "import torch; torch.use_deterministic_algorithms(True)"
```

---

## 🔐 Security & Credentials

### Current Issues

⚠️ **Hardcoded Kaggle credentials**: `!cp kaggle.json /root/.kaggle`  
⚠️ **Credentials in notebook**: `kaggle.json` copied directly in code  

### Secure Handling

```bash
# 1. Set environment variables (preferred)
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"

# 2. Use Kaggle CLI config
mkdir -p ~/.kaggle
echo '{"username":"USER","key":"KEY"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# 3. Never commit credentials
echo "kaggle.json" >> .gitignore
```

---

## 🎨 Model Card

### Intended Use

**Primary**: Educational demonstration of LLM architecture and training dynamics  
**Secondary**: Infrastructure testing for checkpoint management and multi-GPU training  
**Not Suitable For**: Production text generation, factual QA, or real-world applications

### Limitations

| Category | Limitation |
|----------|------------|
| **Data** | Only ~100K tokens (0.001% of typical pretraining corpus) |
| **Overfitting** | Model memorizes training data (PPL gap: 16×) |
| **Context** | Max 1024 tokens (modern LLMs: 32K-128K+) |
| **Knowledge** | Limited to WikiText-103 domain (Wikipedia abstracts) |
| **Safety** | No alignment, content filtering, or bias mitigation |
| **Inference** | Slow generation (~2-3 tokens/sec without KV cache optimization) |

### Ethical Considerations

- **Training Data**: WikiText-103 (Wikipedia text, public domain)
- **Biases**: Inherits Wikipedia's documented biases (Western-centric, gender imbalances)
- **Misinformation Risk**: Overfitting causes factually incorrect but fluent outputs
- **Carbon Footprint**: ~8 GPU-hours for 20 epochs (~0.5 kWh on L4 GPUs)

### License

**Code**: MIT License  
**Model Weights**: Not for commercial use (educational only)  
**Tokenizer**: Mistral AI tokenizer (Apache 2.0)  
**Dataset**: WikiText-103 (Creative Commons)

---

## 🛠️ Next Steps & Recommended Experiments

### Priority 1: Fix Overfitting (HIGH IMPACT)

| Task | Effort | Expected Improvement | Commands |
|------|--------|---------------------|----------|
| **Scale dataset to 1B tokens** | High | -2.0 val loss | `load_dataset("wikitext-103-raw-v1", split="train[:100%]")` |
| **Add dropout (0.1 → 0.2)** | Low | -0.3 val loss | `config.dropout = 0.2` |
| **Reduce model size (6L → 4L)** | Low | -0.5 val loss | `config.num_layers = 4` |
| **Early stopping (patience=3)** | Medium | Stop at epoch 12 | Implement validation callback |

### Priority 2: Infrastructure Hardening (MEDIUM IMPACT)

| Task | Effort | Benefit |
|------|--------|---------|
| **Add requirements.txt** | Low | Reproducible installs |
| **Implement learning rate finder** | Medium | Faster convergence |
| **Add MLflow logging** | Medium | Experiment tracking |
| **Dockerize environment** | High | True reproducibility |
| **Implement data versioning (DVC)** | Medium | Dataset provenance |

### Priority 3: Architecture Improvements (LOW IMPACT)

| Task | Effort | Expected Gain |
|------|--------|---------------|
| **Enable FlashAttention** | Low | 1.5× speed | `config.use_flash_attention=True` |
| **Tune MoE load balancing** | Medium | Better expert utilization | `config.load_balance_loss_coeff=0.02` |
| **Add Z-loss regularization** | Low | Stable logits | Already implemented |
| **Implement GQA caching** | High | 3× inference speed | Refactor KV cache |

### Priority 4: Evaluation Suite (HIGH IMPACT)

| Task | Effort | Purpose |
|------|--------|---------|
| **Add perplexity benchmarks** | Medium | Compare to baselines |
| **Implement BLEU/ROUGE metrics** | Medium | Measure generation quality |
| **Test on downstream tasks** | High | Real-world validation |
| **Human evaluation protocol** | High | Qualitative assessment |

### Suggested Training Schedule

```bash
# Phase 1: Quick validation (current)
python h64lm_v1.py  # 100K tokens, 20 epochs → overfitting confirmed

# Phase 2: Scale dataset
python h64lm_v1.py --max_samples 1000000  # 1M tokens, 10 epochs

# Phase 3: Full-scale pretraining
python h64lm_v1.py --max_samples -1  # All WikiText-103, 5 epochs
# Expected: Val loss ~3.2, PPL ~25 (close to published baselines)

# Phase 4: Add regularization
python h64lm_v1.py --dropout 0.2 --weight_decay 0.1
```

---

## 📦 Appendix

### A. Hyperparameter Table

| Category | Parameter | Value | Notes |
|----------|-----------|-------|-------|
| **Model** | `vocab_size` | 32000 | Mistral tokenizer |
| | `hidden_size` | 768 | |
| | `num_layers` | 6 | 3 dense + 3 MoE |
| | `num_attention_heads` | 12 | |
| | `num_kv_heads` | 4 | GQA 3:1 ratio |
| | `num_experts` | 8 | Per MoE layer |
| | `num_experts_per_token` | 2 | Top-2 routing |
| | `max_position_embeddings` | 1024 | |
| | `sliding_window_size` | 2048 | |
| **Training** | `num_epochs` | 20 | Original run |
| | `batch_size` | 16 | Per-GPU |
| | `grad_accum_steps` | 8 | Effective batch: 512 |
| | `learning_rate` | 3e-4 | AdamW base LR |
| | `warmup_steps` | 10% of total | Linear warmup |
| | `weight_decay` | 0.01 | Exclude biases/norms |
| | `gradient_clip_norm` | 1.0 | |
| | `use_fp16` | True | Mixed precision |
| | `gradient_checkpointing` | True | Memory efficient |
| **Regularization** | `dropout` | 0.1 | Attention + MLP |
| | `expert_dropout` | 0.05 | MoE-specific |
| | `load_balance_loss_coeff` | 0.01 | Aux loss weight |
| | `diversity_loss_coeff` | 0.002 | Entropy term |
| | `z_loss_coeff` | 1e-3 | Logit regularization |
| **Data** | `max_samples` | 100000 | Training samples |
| | `train_split` | 90% | 51,044 samples |
| | `val_split` | 10% | 5,672 samples |
| | `dataset_source` | WikiText-103 | |

### B. File Structure

```
h64lm/
├── h64lm_v1.py              # Main training script (all-in-one)
├── checkpoints/             # Model checkpoints
│   ├── best_model.pt
│   ├── checkpoint_epoch{N}_final.pt
│   ├── training_history.json
│   └── training_curves.png
├── data/
│   ├── wikitext-103/        # Dataset (1.8M training samples)
│   └── mistral_tokenizer/   # Pretrained tokenizer
└── README.md                # This file
```

### C. Key Artifact Paths

| Artifact | Path | Size | Description |
|----------|------|------|-------------|
| **Best Checkpoint** | `checkpoints/best_model.pt` | ~950 MB | Epoch 10 (val loss: 3.70) |
| **Final Checkpoint** | `checkpoints/checkpoint_epoch19_final.pt` | ~950 MB | Epoch 20 (overfitted) |
| **Training Log** | `checkpoints/training_history.json` | ~5 KB | Metrics per epoch |
| **Loss Curves** | `checkpoints/training_curves.png` | ~100 KB | Matplotlib plot |
| **Tokenizer** | `data/mistral_tokenizer/` | ~5 MB | Mistral BPE 32K vocab |
| **Dataset** | `data/wikitext-103/` | ~500 MB | Raw Wikipedia text |

### D. Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| `CUDA out of memory` | Reduce `batch_size` or enable `gradient_checkpointing=True` |
| `Checkpoint save failed` | Disk full or NFS timeout → use local SSD |
| `DataLoader workers crash` | Set `num_workers=0` or upgrade PyTorch |
| `Module prefix mismatch` | Use `load_checkpoint_for_resume()` (auto-strips) |
| `NaN loss during training` | Lower LR to 1e-4, enable gradient clipping |
| `Generation is incoherent` | Expected with overfitting → scale dataset |

### E. Hardware Requirements

| Component | Minimum | Recommended | Original Setup |
|-----------|---------|-------------|----------------|
| **GPU** | 1× 16GB VRAM | 4× 24GB VRAM | 4× NVIDIA L4 (24GB) |
| **RAM** | 32 GB | 64 GB | 64 GB |
| **Storage** | 50 GB | 200 GB | 100 GB |
| **Training Time** | 10 hours (1 GPU) | 2.5 hours (4 GPU) | ~8 hours (20 epochs) |

---

## 🤝 Contributing

This is an educational project demonstrating LLM internals. Contributions welcome for:

- Fixing overfitting (dataset scaling, regularization)
- Adding evaluation metrics (BLEU, ROUGE, perplexity benchmarks)
- Infrastructure improvements (Docker, MLflow, DVC)
- Documentation enhancements

**Not Accepting**: Architectural changes that obscure learning goals (e.g., "just use HuggingFace Trainer").

---

## 📚 References & Acknowledgments

### Papers Implemented

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Vaswani et al., 2017 (Transformer)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) - Su et al., 2021 (RoPE)
- [Train Short, Test Long: Attention with Linear Biases (ALiBi)](https://arxiv.org/abs/2108.12409) - Press et al., 2021
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) - Shazeer, 2020 (SwiGLU)
- [Outrageously Large Neural Networks: The Sparsely-Gated MoE](https://arxiv.org/abs/1701.06538) - Shazeer et al., 2017
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) - Zhang & Sennrich, 2019

### Code Inspirations

- [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) - MoE architecture
- [LLaMA](https://github.com/facebookresearch/llama) - RMSNorm, RoPE
- [Mistral](https://github.com/mistralai/mistral-src) - Sliding window attention

### Datasets & Tokenizers

- [WikiText-103](https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/) - Merity et al., 2016
- [Mistral Tokenizer](https://huggingface.co/mistralai/Mistral-7B-v0.1) - Mistral AI, 2023

---

## 📬 Contact

**Author**: Haider Khan  
**Project**: Educational LLM Implementation  
**Status**: Infrastructure validated, awaiting dataset scaling  

For questions about reproducing results or architectural choices, open an issue on GitHub.

---

**Last Updated**: November 2024  
**Training Hardware**: 4× NVIDIA L4 (24GB each)  
**Total Training Time**: ~8 GPU-hours  
**Model Weights**: Not released (educational project only)