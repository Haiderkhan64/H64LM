# H64LM

A research-oriented, from-scratch implementation of a modern LLM stack in PyTorch. Every component attention, expert routing, normalization, training loop, checkpointing is implemented directly, with no wrappers around the interesting parts.

The 249M-parameter model included was trained on a small WikiText-103 slice to validate the pipeline end-to-end, not to produce a useful language model. The architecture is designed to scale, but the training infrastructure is single-node only.

**What makes this different from other transformer repos:**

- Full MoE implementation with three auxiliary routing losses not stubbed or approximated
- Explicit training loop with no Trainer abstractions: AMP, gradient scaling, accumulation, scheduling, and checkpoint safety all written directly
- End-to-end pipeline in a single inspectable file tokenizer training, dataset curation, model, and training loop together
- Architecture mirrors decisions in LLaMA, Mistral, and Mixtral, implemented from first principles so the mechanics are visible

---

## Table of Contents

- [Architecture](#architecture)
- [Design Philosophy](#design-philosophy)
- [File Structure](#file-structure)
- [Requirements](#requirements)
- [Setup](#setup)
- [Training](#training)
- [Resuming from Checkpoint](#resuming-from-checkpoint)
- [Inference](#inference)
- [Data Utilities](#data-utilities)
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

Even-indexed layers (0, 2, 4) use a dense SwiGLU feedforward block. Odd-indexed layers (1, 3, 5) use a sparse Mixture-of-Experts block. All layers share the same GQA attention module.

The alternating dense/MoE pattern balances compute and parameter capacity: dense layers provide stable shared representations that benefit all tokens, while MoE layers increase parameter count without a proportional increase in per-token compute cost. Every token passes through every dense layer but only through 2 of 8 experts in each MoE layer.

### Component Summary

| Component | Implementation | Notes |
|---|---|---|
| Attention | Grouped-Query Attention | 12 query heads, 4 KV heads reduces KV cache memory 3× vs MHA |
| Position encoding | RoPE | Applied to Q and K; ALiBi available via `attention_type="alibi"` |
| Feedforward | SwiGLU | Gate projection × up projection, then down projection |
| Expert routing | Top-2 sparse MoE | 8 experts per MoE layer, softmax gating with temperature scaling |
| Normalization | RMSNorm pre-norm | Applied before attention and before MLP in every layer |
| Masking | Causal + sliding window | Window size 512; combined with causal mask in the backbone |
| KV cache | Autoregressive decoding | Reuses past key/value tensors for efficient autoregressive decoding |

### MoE Auxiliary Losses

Each MoE layer returns three auxiliary losses that are added to cross-entropy during training:

| Loss | Coefficient | Purpose |
|---|---|---|
| Load balance loss | `load_balance_loss_coeff = 0.01` | Minimises variance in per-expert token usage |
| Diversity loss | `diversity_loss_coeff = 0.002` | Entropy of mean routing distribution; encourages all experts to receive traffic |
| Z-loss | `z_loss_coeff = 1e-3` | Penalises large router logits; stabilises softmax in FP16 |

These are averaged across all MoE layers before being added to the total loss. They are only computed during training (`model.training == True`), not during validation or inference.

### Config Classes

Two separate dataclasses keep model architecture and training settings cleanly separated.

**`H64LMConfig`** model architecture only:

```python
@dataclass
class H64LMConfig:
    vocab_size              : int   = 32000
    hidden_size             : int   = 768
    num_layers              : int   = 6
    num_attention_heads     : int   = 12
    num_kv_heads            : int   = 4
    max_position_embeddings : int   = 1024
    sliding_window_size     : int   = 512
    num_experts             : int   = 8
    num_experts_per_token   : int   = 2
    expert_hidden_size      : int   = None   # defaults to hidden_size * 4 = 3072
    use_flash_attention     : bool  = False
    kv_block_size           : int   = 256
    layer_norm_eps          : float = 1e-5
    dropout                 : float = 0.1
    expert_dropout          : float = 0.05
    capacity_factor         : float = 1.25
    load_balance_loss_coeff : float = 0.01
    pad_token_id            : int   = 0
    bos_token_id            : int   = 1
    eos_token_id            : int   = 2
    rope_theta              : float = 10000.0
    initializer_range       : float = 0.02
    tie_word_embeddings     : bool  = False
    moe_temperature         : float = 2.0
    diversity_loss_coeff    : float = 0.002
    z_loss_coeff            : float = 1e-3
    initializer_type        : str   = "scaled"
    attention_type          : str   = "rope"   # or "alibi"
    use_attention_sinks     : bool  = True
    kv_cache_fp16           : bool  = False
    use_cache               : bool  = True
    # __post_init__ computes:
    #   head_dim = hidden_size // num_attention_heads
    #   expert_hidden_size = hidden_size * 4 if None
    #   num_kv_heads = 1 if attention_type == "mqa"
```

> `vocab_size`, `pad_token_id`, `eos_token_id`, and `bos_token_id` are overwritten inside `main()` from the loaded tokenizer before the model is built. The dataclass defaults are placeholders only.

**`H64LMTrainingConfig`** training hyperparameters and runtime settings:

```python
@dataclass
class H64LMTrainingConfig:
    num_epochs               : int   = 1
    batch_size               : int   = 16
    grad_accum_steps         : int   = 8        # effective batch size = 128
    val_interval             : int   = None      # no-op; see Known Issues
    save_interval            : int   = 500000    # no-op; see Known Issues
    log_interval             : int   = 10
    use_tensorboard          : bool  = False     # no-op; see Known Issues
    checkpoint_dir           : str   = "checkpoints"
    resume_from              : str   = "checkpoints_h64lm/best_model_state_dict.pt"
    use_ddp                  : bool  = True      # no-op; see Known Issues
    use_pretrained_tokenizer : str   = "mistral_tokenizer"
    streaming                : bool  = False
    dataset                  : str   = "parquet"
    dataset_dir              : str   = "data/wikitext-103/wikitext-103-raw-v1"
    max_samples              : int   = 50000
    use_fp16                 : bool  = True
    gradient_checkpointing   : bool  = True
    # __post_init__ sets use_fp16=False and gradient_checkpointing=False on CPU
```

> `checkpoint_dir` is where new checkpoints are written. `resume_from` points to a previously saved checkpoint to load before training begins. Set `resume_from = None` to start from scratch.

> `val_interval`, `save_interval`, `use_tensorboard`, and `use_ddp` are accepted by the config but do not affect execution. They are placeholders for future features. See [Known Issues](#known-issues).

---

## Design Philosophy

H64LM is designed as a fully transparent implementation of a modern LLM stack.

- **No abstraction over core components.** Attention, MoE routing, normalization, and the training loop are all written directly. There are no Trainer APIs or wrapped kernels hiding the behaviour.
- **Explicit training mechanics.** Every training decision AMP, gradient accumulation, auxiliary losses, LR scheduling, checkpoint safety is visible in the source.
- **Educational clarity with real-world choices.** The architecture mirrors decisions found in LLaMA, Mistral, and Mixtral, implemented from first principles.

This codebase is a learning and research tool, not a production framework. It does not implement distributed training, expert parallelism, or fault-tolerant dataloading.

---

## File Structure

```
H64LM/
├── h64lm.py                     # All model code + training loop (single file)
├── H64LM-v1.ipynb               # Notebook version (identical code)
├── requirements.txt
├── LICENSE                      # Apache 2.0
├── README.md
├── mistral_tokenizer/           # Pretrained tokenizer (Git LFS)
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── tokenizer.model
│   └── special_tokens_map.json
└── checkpoints_h64lm/           # Reference run artifacts (Git LFS)
    ├── best_model_state_dict.pt
    ├── training_history.json
    └── tokenizer/
```

> The notebook is `H64LM-v1.ipynb`. The standalone script `h64lm.py` contains the same code and is the recommended entry point for command-line use. `H64LM-v1.py` has a hyphen in its name and cannot be imported as a Python module.

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
```

Optional:

```
tensorboard>=2.13   # imported if available; not wired into the training loop
flash-attn>=2.0     # GPU only; may require manual install
fastermoe           # GPU only; enables optimised MoE dispatch kernels
vllm                # not used during training
```

```bash
pip install -r requirements.txt
```

All optional libraries are imported with `try/except`. If unavailable, a log message is printed and the code continues with fallbacks. Flash Attention has a two-level fallback: (1) if `flash_attn` is not importable, standard attention is used; (2) even when available, each forward pass wraps `flash_attn_func` in `try/except` and falls back to standard attention if the kernel call fails at runtime.

---

## Setup

```bash
git clone https://github.com/Haiderkhan64/H64LM.git
cd H64LM

# The tokenizer and checkpoint files are stored in Git LFS
git lfs install
git lfs pull
```

If you do not have Git LFS, download the LFS files manually from the GitHub repository.

### Notebook Usage

The notebook (`H64LM-v1.ipynb`) is stateful and must be run **top-to-bottom**. Running cells out of order may cause missing dependencies, incorrect model state, or inconsistent training behaviour. For reproducible runs, use `h64lm.py`.

---

## Training

```bash
python h64lm.py
```

What happens:

1. `H64LMConfig()` and `H64LMTrainingConfig()` are instantiated with defaults inside `main()`.
2. The tokenizer is loaded from `./mistral_tokenizer/` with `local_files_only=True` no internet required.
3. `vocab_size`, `pad_token_id`, `eos_token_id`, and `bos_token_id` are copied from the tokenizer into `H64LMConfig` before the model is built.
4. WikiText-103 is loaded from `dataset_dir` if the path exists, otherwise downloaded from Hugging Face.
5. Up to `max_samples` rows are read, filtered to texts longer than 20 characters, shuffled, and split 90/10 into train and validation sets.
6. The model is initialised (~249M parameters at default config) with depth-aware weight initialisation.
7. An AdamW optimiser is created with parameter-group weight decay (bias and norm weights exempt). A linear LR schedule with 10% warmup is applied via `get_linear_schedule_with_warmup`.
8. Training runs for `num_epochs` epochs. The training loop is implemented fully from scratch AMP (`torch.cuda.amp.autocast` + `GradScaler`), gradient accumulation, MoE auxiliary losses, and LR scheduling are all written directly with no Trainer abstractions. The tqdm progress bar shows raw loss, a 20-batch smoothed loss, and the current learning rate.
9. Validation loss and perplexity are computed once per epoch on the full validation set.
10. The best checkpoint (lowest validation loss) is saved to `{checkpoint_dir}/best_model_state_dict.pt`.
11. Per-epoch checkpoints are saved to `{checkpoint_dir}/checkpoint_epoch{n}_final.pt`.
12. Training curves are saved to `checkpoints/training_curves.png` (fixed path; see Known Issues).
13. Training history is saved to `{checkpoint_dir}/training_history.json`.

### Changing Hyperparameters

```python
# Model architecture
config = H64LMConfig(
    hidden_size             = 1024,
    num_layers              = 12,
    num_attention_heads     = 16,
    num_kv_heads            = 4,
    sliding_window_size     = 512,
)

# Training settings
training_config = H64LMTrainingConfig(
    num_epochs     = 5,
    batch_size     = 8,
    max_samples    = 50000,
    checkpoint_dir = "my_run",
    resume_from    = None,    # start from scratch
)
```

### Optimiser and Scheduler

| Setting | Value |
|---|---|
| Optimiser | AdamW |
| Learning rate | `3e-4` |
| Betas | `(0.9, 0.95)` |
| Epsilon | `1e-8` |
| Weight decay | `0.01` on weights; `0.0` on bias and norm parameters |
| Scheduler | `get_linear_schedule_with_warmup` linear warmup for 10% of total steps, then linear decay to zero |
| AMP | `torch.cuda.amp.autocast` + `GradScaler`; disabled automatically on CPU |
| Gradient clipping | `max_norm=1.0` applied after unscaling |
| Gradient accumulation | Configurable via `grad_accum_steps`; default effective batch = 128 |

### Multi-GPU

The script detects available GPUs automatically. When more than one GPU is present, `torch.nn.DataParallel` is applied regardless of the `use_ddp` flag.

```bash
# DataParallel is enabled automatically
CUDA_VISIBLE_DEVICES=0,1,2,3 python h64lm.py
```

> `use_ddp=True` and `use_ddp=False` are equivalent both fall back to DataParallel. True `DistributedDataParallel` via `torchrun` is not implemented. See [Known Issues](#known-issues).

---

## Resuming from Checkpoint

Set `resume_from` in `H64LMTrainingConfig` to the path of any saved checkpoint:

```python
training_config = H64LMTrainingConfig(
    resume_from    = "checkpoints_h64lm/best_model_state_dict.pt",
    num_epochs     = 10,
    checkpoint_dir = "checkpoints_continued",
)
```

`load_checkpoint_for_resume` handles:

- **`module.` prefix stripping** DataParallel checkpoints prepend `module.` to every key; stripped automatically.
- **Full checkpoints** (`model_state_dict` + `optimizer_state_dict` + `scheduler_state_dict`).
- **State-dict-only files** (e.g. if only `model.state_dict()` was saved directly).
- **Partial key matches** (`strict=False`) with a warning listing missing or unexpected keys.

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

Saves are performed atomically: the checkpoint is written to a `.tmp` file first, then renamed over the previous file only if the write succeeds. A failed save never corrupts the last good checkpoint.

---

## Inference

```python
import torch
from transformers import AutoTokenizer
from h64lm import H64LMConfig, H64LMForCausalLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("./mistral_tokenizer", local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Build model
config = H64LMConfig()
config.vocab_size   = tokenizer.vocab_size
config.pad_token_id = tokenizer.pad_token_id
config.eos_token_id = tokenizer.eos_token_id
config.bos_token_id = tokenizer.bos_token_id or tokenizer.eos_token_id

model = H64LMForCausalLM(config, tokenizer).to(device)

# Load checkpoint
ckpt = torch.load("checkpoints_h64lm/best_model_state_dict.pt", map_location="cpu")
state_dict = ckpt.get("model_state_dict", ckpt)
state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
model.load_state_dict(state_dict, strict=False)
model.eval()

# Generate
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

### Generation Parameters

| Parameter | Default | Effect |
|---|---|---|
| `max_new_tokens` | 50 | Maximum tokens to generate |
| `temperature` | 0.8 | Higher = more random output |
| `top_p` | 0.9 | Nucleus sampling threshold |
| `top_k` | 50 | Restrict vocabulary to top-k tokens at each step |
| `do_sample` | True | `False` = greedy decoding |

> **Batch size 1 only.** The built-in `generate` method is only correct for batch size 1. It has structural batching bugs that silently produce wrong results for larger batches: (1) a single `set()` tracks generated tokens globally rather than per-sequence; (2) the repetition penalty loop hardcodes `logits[0, token_id]`, always modifying only the first sequence; (3) the attention mask is extended per step but not validated against the growing KV cache shape. For batched or production inference, use Hugging Face's generation utilities.

---

## Data Utilities

`DedupPipeline` and `train_tokenizer` are available in the codebase but are **not invoked by the default `main()` pipeline**. Use them as preprocessing steps when working with noisier or custom data sources.

### `DedupPipeline` Dataset Curation

```python
from h64lm import DedupPipeline

pipeline = DedupPipeline(tokenizer)
filtered = pipeline.process(
    sources      = ["wikitext"],   # any Hugging Face dataset name
    max_length   = 2048,
    max_examples = 10000,
)
# filtered is a list of {"text": str, "input_ids": List[int]}
```

Filtering criteria per document:

1. Text must be at least 50 characters.
2. Unique-word ratio must be ≥ 0.30.
3. SHA-256 hash must not have been seen before in this pipeline instance.
4. After tokenisation, unique-token ratio must be ≥ 0.15.

### `train_tokenizer` BPE Tokenizer from Scratch

```python
from datasets import load_dataset
from h64lm import train_tokenizer

dataset = load_dataset("your_dataset", split="train")
train_tokenizer(
    dataset    = dataset,
    vocab_size = 32000,
    save_path  = "my_tokenizer.json",
)
```

Load later with:

```python
from transformers import PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast(tokenizer_file="my_tokenizer.json")
```

Special tokens: `[PAD]` (id 0), `[BOS]` (id 1), `[EOS]` (id 2), `[UNK]` (id 3). Set `pad_token`, `bos_token`, and `eos_token` manually after loading.

---

## Scaling

The config is fully parameterised. The model architecture scales; the training infrastructure does not there is no DDP, FSDP, ZeRO sharding, or expert parallelism. For models beyond a few hundred million parameters, the infrastructure would need to be rebuilt. The configurations below are illustrative; training at these sizes is not feasible without distributed infrastructure.

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
)

training_config = H64LMTrainingConfig(
    max_samples      = -1,
    streaming        = True,
    batch_size       = 4,
    grad_accum_steps = 32,
    checkpoint_dir   = "run_large",
    resume_from      = None,
)
```

For datasets too large to hold in memory:

```python
dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
```

### Training Pipeline Stages

| Stage | What to do |
|---|---|
| 1 Pretraining | `python h64lm.py` next-token prediction on raw text |
| 2 Instruction tuning | Fine-tune on Alpaca / Dolly with an instruction format |
| 3 RLHF | Train a reward model; apply PPO |
| 4 Reasoning | Fine-tune on GSM8K or other chain-of-thought data |
| 5 Chat | Fine-tune on OpenAssistant or similar dialogue data |

Each stage resumes from the previous stage's best checkpoint via `training_config.resume_from`.

---

## Known Issues

| Issue | Status | Details |
|---|---|---|
| `generate()` only supports batch size 1 | Known limitation | Three structural bugs make batched generation silently incorrect: (1) a single `set()` tracks generated tokens globally; (2) the penalty loop hardcodes `logits[0, token_id]`; (3) the attention mask is not validated against the KV cache shape. Use Hugging Face generation utilities for batched inference. |
| `inline_container.cc` save errors on multi-GPU | Non-fatal | Intermittent PyTorch serialisation bug under DataParallel. The `.tmp` → rename pattern ensures a failed save never corrupts the last good checkpoint. Observed consistently from epoch 6 onward in the reference run. |
| True DDP not implemented; `use_ddp` has no effect | Known limitation | `DistributedDataParallel` via `torchrun` is not supported. Both `use_ddp=True` and `use_ddp=False` fall back to DataParallel. DataParallel is particularly inefficient for MoE layers expert dispatch is not distributed-aware. Proper DDP is a contribution opportunity. |
| `sliding_window_size` must be less than `max_position_embeddings` | Fixed | Earlier versions had `sliding_window_size=2048` with `max_position_embeddings=1024`, so the window condition was never triggered. Current defaults are `sliding_window_size=512` and `max_position_embeddings=1024`. |
| `attention_type="alibi"` must be set at construction time | Known | ALiBi slopes are allocated only when `attention_type="alibi"` is passed to `H64LMConfig` at model construction. Changing the field on an existing instance has no effect. |
| Training curves saved to a fixed path | Minor | `training_curves.png` is always written to `checkpoints/training_curves.png` regardless of the `checkpoint_dir` setting, contrary to what the log message implies. |
| `val_interval` has no effect | No-op | Reserved for future mid-epoch validation. Validation currently runs once per epoch unconditionally. |
| `save_interval` has no effect | No-op | Reserved for future mid-epoch checkpointing. Saves currently happen at epoch end only. |
| `use_tensorboard` has no effect | No-op | `SummaryWriter` is importable if TensorBoard is installed, but no writer is instantiated or called inside the training loop. |
| `CosineAnnealingLR` imported but unused | Informational | The active scheduler is `get_linear_schedule_with_warmup`. `CosineAnnealingLR` is never called. |
| `torch.profiler` imported but unused | Informational | `profile`, `record_function`, and `ProfilerActivity` are imported but never invoked. Can be activated manually for performance analysis. |

---

## Training Results

Reference run: 20 epochs, 51K training samples + 5.6K validation samples, 4× GPU (DataParallel), Mistral tokenizer (32K vocab), default 249M-parameter config.

**Optimiser:** AdamW (`lr=3e-4`), linear warmup for 10% of steps, then linear decay to zero.  
**Average throughput:** ~10,014 tokens/sec across all epochs.

| Epoch | Train Loss | Train PPL | Val Loss | Val PPL |
|---|---|---|---|---|
| 1 | 7.49 | 1785.9 | 5.72 | 303.7 |
| 2 | 5.25 | 190.5 | 4.89 | 132.5 |
| 5 | 3.95 | 52.2 | 4.00 | 54.7 |
| 10 | 3.01 | 20.3 | **3.70** | **40.5** ← best |
| 15 | 2.16 | 8.6 | 3.96 | 52.6 |
| 20 | 1.54 | 4.7 | 4.32 | 75.3 |

Validation loss diverges after epoch 10. This is expected: 249M parameters on 51K short samples is heavily overfit. This run was for pipeline validation only data volume is the binding constraint, not the architecture.

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

Implementation references: [GPT-NeoX](https://github.com/EleutherAI/gpt-neox), [LLaMA](https://github.com/facebookresearch/llama), [Mistral](https://github.com/mistralai/mistral-src).

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