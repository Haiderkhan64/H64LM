# H64LM

A Mixture-of-Experts transformer built from scratch in PyTorch. Every component
attention, expert routing, normalization, training loop, checkpointing is
implemented directly, with no wrappers around the interesting parts.

**This is a research and learning codebase.** The 249M-parameter model included
was trained on a small WikiText-103 slice to validate the pipeline end-to-end,
not to produce a useful language model. The architecture is designed to scale.

---

## Table of Contents

- [Architecture](#architecture)
- [File Structure](#file-structure)
- [Requirements](#requirements)
- [Setup](#setup)
- [Training](#training)
- [Resuming from Checkpoint](#resuming-from-checkpoint)
- [Inference](#inference)
- [Scaling](#scaling)
- [Known Issues](#known-issues)
- [Training Results](#training-results)
- [References](#references)
- [License](#license)
- [Citation](#citation)

---

## Architecture

```
Input IDs
    │
    ▼
Token Embedding  (vocab_size=32000 → hidden=768)
    │
    ├── Layer 0  ──  GQAAttention  +  Dense SwiGLU
    ├── Layer 1  ──  GQAAttention  +  Sparse MoE  (8 experts, top-2)
    ├── Layer 2  ──  GQAAttention  +  Dense SwiGLU
    ├── Layer 3  ──  GQAAttention  +  Sparse MoE
    ├── Layer 4  ──  GQAAttention  +  Dense SwiGLU
    └── Layer 5  ──  GQAAttention  +  Sparse MoE
    │
    ▼
RMSNorm  →  LM Head  (hidden=768 → vocab_size=32000)
```

Even layers use a dense SwiGLU feedforward block. Odd layers use a sparse
Mixture-of-Experts block. All layers share the same GQA attention module.

### Component summary

| Component | Implementation | Why |
|---|---|---|
| Attention | Grouped-Query Attention | 12 query heads, 4 KV heads reduces KV cache size 3× |
| Position encoding | RoPE | Applied to Q and K; ALiBi available via config |
| Feedforward | SwiGLU | Gate projection × up projection, then down |
| Expert routing | Top-2 sparse MoE | 8 experts per layer, softmax gating with temperature |
| Normalization | RMSNorm pre-norm | Before attention and before MLP in each layer |
| Masking | Causal + sliding window | Window of 2048; causal mask built in the backbone |
| Inference | KV cache | Autoregressive decoding reuses past key/value tensors |

### Default config

```python
H64LMConfig(
    # Architecture
    vocab_size              = 32000,
    hidden_size             = 768,
    num_layers              = 6,
    num_attention_heads     = 12,
    num_kv_heads            = 4,
    max_position_embeddings = 1024,
    sliding_window_size     = 2048,

    # MoE
    num_experts             = 8,
    num_experts_per_token   = 2,
    moe_temperature         = 2.0,
    load_balance_loss_coeff = 0.01,
    diversity_loss_coeff    = 0.002,
    z_loss_coeff            = 1e-3,

    # Attention
    attention_type          = "rope",   # or "alibi"
    use_flash_attention     = False,

    # Training
    batch_size              = 16,
    grad_accum_steps        = 8,        # effective batch = 128
    num_epochs              = 1,
    use_fp16                = True,
    gradient_checkpointing  = True,
)
```

---

## File Structure

```
H64LM/
├── h64lm.py                     # All model code + training loop (single file)
├── H64LM-v1.ipynb               # Notebook version (same code)
├── requirements.txt
├── LICENSE                      # Apache 2.0
├── README.md
├── mistral_tokenizer/           # Pretrained tokenizer (Git LFS)
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── tokenizer.model
│   └── special_tokens_map.json
└── checkpoints_h64lm/           # Saved artifacts (Git LFS)
    ├── best_model_state_dict.pt
    ├── training_history.json
    └── tokenizer/
```

> **Note on the filename:** `H64LM-v1.py` cannot be imported in Python because
> the hyphen is not valid in a module name. Use `h64lm.py` for importing, or run
> the original as a script directly with `python H64LM-v1.py`.

---

## Requirements

Python 3.8+. CUDA 12.1 recommended for GPU training.

```
numpy>=1.21
tqdm>=4.62
matplotlib>=3.4
torch>=2.0.0
datasets>=2.14
transformers>=4.30
tokenizers>=0.13

# Optional but recommended
tensorboard>=2.13
flash-attn>=2.0        # GPU only — may need manual install
```

Install:

```bash
pip install -r requirements.txt
```

Flash Attention requires a compatible GPU and CUDA toolkit. If it is not
available the model falls back to standard scaled dot-product attention
automatically.

---

## Setup

```bash
git clone https://github.com/Haiderkhan64/H64LM.git
cd H64LM

# The tokenizer and checkpoint files are stored in Git LFS
git lfs install
git lfs pull
```

If you do not have Git LFS, download the LFS files manually from the GitHub
repository releases page or from the `checkpoints_h64lm/` folder on GitHub.

---

## Training

```bash
python h64lm.py
```

What happens:

1. `H64LMConfig()` is instantiated with defaults
2. Tokenizer is loaded from `./mistral_tokenizer/`
3. WikiText-103 is downloaded from Hugging Face (or loaded locally if present)
4. Up to `max_samples` examples are sampled, shuffled, and split 90/10
5. Model is initialised (~249M parameters at default config)
6. Training runs for `num_epochs` epochs with validation after each
7. Best checkpoint is saved to `checkpoint_dir/best_model.pt`
8. Training curves are saved to `checkpoint_dir/training_curves.png`

### Changing hyperparameters

Edit the `H64LMConfig` instantiation at the top of `main()`:

```python
config = H64LMConfig(
    num_epochs   = 5,
    batch_size   = 8,
    max_samples  = 50000,
    checkpoint_dir = "my_run",
)
```

### Multi-GPU

The script detects available GPUs automatically and wraps the model in
`torch.nn.DataParallel` when more than one GPU is present. No extra flags
needed.

```bash
# 4 GPUs — DataParallel is enabled automatically
CUDA_VISIBLE_DEVICES=0,1,2,3 python h64lm.py
```

True DistributedDataParallel (DDP) is stubbed in the config (`use_ddp=True`)
but currently falls back to DataParallel. Proper DDP requires launching with
`torchrun` and is left as a contribution opportunity.

---

## Resuming from Checkpoint

Set `resume_from` in the config:

```python
config = H64LMConfig(
    resume_from    = "checkpoints_h64lm/best_model_state_dict.pt",
    num_epochs     = 10,
    checkpoint_dir = "checkpoints_continued",
)
```

The loader handles:

- `module.` prefix stripping (DataParallel checkpoints)
- Full checkpoints (`model_state_dict` + `optimizer_state_dict` + `scheduler_state_dict`)
- State-dict-only files
- Partial matches (`strict=False`)

Checkpoint format on disk:

```python
{
    "epoch":                int,
    "step":                 int,
    "model_state_dict":     OrderedDict,   # CPU tensors
    "optimizer_state_dict": dict,
    "scheduler_state_dict": dict,
}
```

---

## Inference

```python
import torch
from transformers import AutoTokenizer

# Copy the class definitions from h64lm.py into your script,
# or add the repo root to sys.path and import directly.
from h64lm import H64LMConfig, H64LMForCausalLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load tokenizer ────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("./mistral_tokenizer", local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ── Build model ───────────────────────────────────────────────────────────────
config = H64LMConfig()
config.vocab_size   = tokenizer.vocab_size
config.pad_token_id = tokenizer.pad_token_id
config.eos_token_id = tokenizer.eos_token_id

model = H64LMForCausalLM(config, tokenizer).to(device)

# ── Load checkpoint ───────────────────────────────────────────────────────────
ckpt = torch.load("checkpoints_h64lm/best_model_state_dict.pt", map_location="cpu")
state_dict = ckpt.get("model_state_dict", ckpt)
state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
model.load_state_dict(state_dict, strict=False)
model.eval()

# ── Generate ──────────────────────────────────────────────────────────────────
inputs = tokenizer("Deep learning is", return_tensors="pt").to(device)
with torch.no_grad():
    output = model.generate(
        inputs["input_ids"],
        max_new_tokens = 100,
        temperature    = 0.8,
        top_p          = 0.9,
        top_k          = 50,
        do_sample      = True,
    )
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

### Generation parameters

| Parameter | Default | Effect |
|---|---|---|
| `max_new_tokens` | 50 | Maximum tokens to generate |
| `temperature` | 0.8 | Higher = more random |
| `top_p` | 0.9 | Nucleus sampling threshold |
| `top_k` | 50 | Restrict to top-k tokens at each step |
| `do_sample` | True | `False` = greedy decoding |

---

## Scaling

The architecture is parameterised end-to-end. To move from the demo config to a
GPT-3 scale model:

```python
config = H64LMConfig(
    hidden_size             = 4096,
    num_layers              = 32,
    num_attention_heads     = 32,
    num_kv_heads            = 8,
    num_experts             = 16,
    num_experts_per_token   = 2,
    max_position_embeddings = 4096,
    sliding_window_size     = 4096,
    max_samples             = -1,      # use full dataset
)
```

For datasets too large to hold in memory, enable streaming:

```python
dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
```

### Training pipeline stages

| Stage | What to do |
|---|---|
| 1  Pretraining | `python h64lm.py`  next-token prediction on raw text |
| 2  Instruction tuning | Fine-tune on Alpaca / Dolly with instruction format |
| 3  RLHF | Train a reward model; apply PPO |
| 4  Reasoning | Fine-tune on GSM8K or other chain-of-thought data |
| 5  Chat | Fine-tune on OpenAssistant or similar dialogue data |

Each stage resumes from the previous stage's best checkpoint via `resume_from`.

---

## Known Issues

| Issue | Details |
|---|---|
| `inline_container.cc` save errors on multi-GPU | Intermittent PyTorch serialisation bug under DataParallel. The `.tmp` → rename pattern means a partial save does not overwrite the last good checkpoint. |
| `use_fp16=True,` trailing comma | In the original `H64LMConfig` the trailing comma makes `use_fp16` a tuple `(True,)` instead of `bool`. Fixed in `h64lm.py`. |
| `sliding_window_size > max_position_embeddings` | At defaults (2048 vs 1024) the sliding window condition is never triggered. Either raise `max_position_embeddings` or lower `sliding_window_size`. |
| `attention_type="alibi"` in README config | The default is `"rope"`. ALiBi slopes are only allocated when `attention_type="alibi"` is set at init time. Mixing configs silently produces wrong results. |
| `H64LM-v1.py` not importable | Hyphen in filename is invalid Python syntax. Renamed to `h64lm.py`. |

---

## Training Results

Reference run: 20 epochs, 51K training samples, 4× GPU (DataParallel),
Mistral tokenizer (32K vocab), default 249M-parameter config.

| Epoch | Train loss | Val loss | Val perplexity |
|---|---|---|---|
| 1 | 7.49 | 5.72 | 303 |
| 2 | 5.25 | 4.89 | 133 |
| 5 | 3.95 | 4.00 | 54.7 |
| 10 | 3.01 | **3.70** | **40.5** ← best |
| 15 | 2.16 | 3.96 | 52.6 |
| 20 | 1.54 | 4.32 | 75.3 |

Validation loss diverges after epoch 10. Expected: 249M parameters on 51K
short samples is heavily overfit. This run was for pipeline validation only.

---

## References

| Paper | Used for |
|---|---|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Transformer backbone |
| [RoFormer](https://arxiv.org/abs/2104.09864) | Rotary Position Embeddings (RoPE) |
| [ALiBi](https://arxiv.org/abs/2108.12409) | Attention with Linear Biases |
| [GLU Variants](https://arxiv.org/abs/2002.05202) | SwiGLU activation |
| [Outrageously Large Neural Networks](https://arxiv.org/abs/1701.06538) | Sparse MoE routing |
| [Root Mean Square Layer Normalisation](https://arxiv.org/abs/1910.07467) | RMSNorm |
| [GQA](https://arxiv.org/abs/2305.13245) | Grouped-Query Attention |

Implementation references: [GPT-NeoX](https://github.com/EleutherAI/gpt-neox),
[LLaMA](https://github.com/facebookresearch/llama),
[Mistral](https://github.com/mistralai/mistral-src).

---

## License

Code: [Apache 2.0](LICENSE)  
Tokenizer: Apache 2.0 (Mistral AI)  
Dataset: CC BY-SA 3.0 (WikiText-103)

---

## Citation

```bibtex
@software{h64lm2025,
  author = {Haider Khan},
  title  = {H64LM: MoE Transformer},
  year   = {2025},
  url    = {https://github.com/Haiderkhan64/H64LM}
}
```

---

**Author:** [Haider Khan](https://github.com/Haiderkhan64)  
**Last updated:** April 2026