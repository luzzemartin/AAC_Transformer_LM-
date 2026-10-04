# AAC v1.2 — Canonical Associative Architecture + Hybrid Canonical-Transformer

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Canonical Phase-4.5 Associative Architecture (AAC)** for language modeling, plus a new **8-block Canonical-Transformer hybrid**.

This repository contains a clean, self-contained implementation of the canonical AAC equations together with a hybrid architecture that routes data through the associative core *before* a standard Transformer block, stacked eight times.

---

## Key Ideas

### Canonical AAC Core (Phase-4.5)

At every token the model maintains:

| Component | Equation | Role |
|-----------|----------|------|
| Fast state | \( S_t = \tanh(W_{hh}S_{t-1} + W_{xh}e_t + b_h) \) | Recurrent dynamics |
| Temporary trace | \( T_t = \lambda_T T_{t-1} + g_{\text{keep},t}\, v_t \) | Short-term evidence (\(\lambda_T=0.9\)) |
| Promotion gate | \( p_t = \sigma(\text{Value}(S_t,T_t,e_t)) \) | Learned utility |
| Persistent matrix | \( P_t = \lambda_P P_{t-1} + p_t\,(k_t\otimes v_t) \) | Associative memory (\(\lambda_P=0.9999\) fixed) |
| Read | \( r_t = q_t^\top P_t \) | Content-based retrieval |
| Output | \( h_t = \tanh(W_{\text{comb}}[S_t;r_t] + b_{\text{comb}}) \) | Combined representation |

**Canonical design choices retained:**
- Fixed high-persistence decay on the associative matrix
- Direct outer-product write (no matrix-level reinforce / evict lifecycle)
- Temporary trace \(T\) is used only as evidence for the utility estimator
- Strictly causal / non-circular token order

### Canonical-Transformer Hybrid (new)

Each of the **8 hybrid blocks** performs:

1. **Canonical AAC** full-sequence pass → produces memory-augmented hidden states
2. **Pre-norm causal Transformer block** (Multi-Head Self-Attention + GELU FFN)

Residual connections surround both the AAC and the Transformer sub-block. A final LayerNorm and tied embedding head produce the logits.

```
Token Embedding
      │
      ▼
┌─────────────────────────────┐
│  Hybrid Block × 8           │
│  ┌───────────────────────┐  │
│  │ Canonical AAC Core    │  │
│  └───────────┬───────────┘  │
│              ▼              │
│  ┌───────────────────────┐  │
│  │ Causal Transformer    │  │
│  │ (MHA + FFN)           │  │
│  └───────────────────────┘  │
└─────────────────────────────┘
      │
      ▼
LayerNorm → Tied LM Head
```

---

## Installation

```bash
git clone https://github.com/<your-username>/aac-v1.2-canonical-transformer.git
cd aac-v1.2-canonical-transformer
pip install torch numpy
```

No other dependencies are required.

---

## Quick Start

```python
from aac_lm import CanonicalAACLM, CanonicalTransformerLM, ByteTokenizer

# Pure Canonical AAC
aac = CanonicalAACLM(d_model=48)
print(f"AAC parameters: {sum(p.numel() for p in aac.parameters()):,}")

# 8-block Canonical-Transformer hybrid
hybrid = CanonicalTransformerLM(d_model=48, n_layers=8, n_heads=4)
print(f"Hybrid parameters: {sum(p.numel() for p in hybrid.parameters()):,}")

# Tokenize & forward
tok = ByteTokenizer()
ids = tok.encode("the researcher ada chose the color ")
import torch
logits = hybrid(torch.tensor([ids]))
print(logits.shape)  # [1, T, vocab]
```

### Training the Hybrid

```bash
# From the repository root
PYTHONPATH=. python scripts/train_canonical_transformer.py
```

Or use the helper:

```bash
./scripts/run_hybrid_train.sh
```

---

## Repository Layout

```
.
├── README.md
├── aac_lm/
│   ├── __init__.py
│   ├── model.py                  # CanonicalAACLM + utilities
│   ├── canonical_transformer.py  # Hybrid model definition
│   └── README.md
├── checkpoints/
│   ├── aac_canonical_lm_synth.pt
│   └── aac_canonical_lm_continued.pt
└── scripts/
    ├── train_canonical_transformer.py
    ├── continue_train.py
    ├── test_aac_lm_capabilities.py
    ├── test_word_assoc.py
    └── run_hybrid_train.sh
```

---

## Results (Synthetic Structured Corpus)

### Pure Canonical AAC
| Stage                        | Loss (NLL) | Perplexity | Accuracy |
|-----------------------------|------------|------------|----------|
| Original checkpoint         | 2.825      | 16.85      | 31.0%    |
| +4 epochs continued         | **0.772**  | **2.16**   | **84.4%**|

### 8-block Canonical-Transformer Hybrid (`d_model=48`, ~431k params)
| Stage              | Loss (NLL) | Perplexity | Accuracy |
|--------------------|------------|------------|----------|
| Random init        | 5.57       | 262        | 0.1%     |
| After 1 epoch      | **~0.10**  | **~1.17**  | **~96%** |

The hybrid reaches near-perfect next-token prediction on the synthetic associative corpus significantly faster and with lower final perplexity than the single-layer AAC baseline, while preserving the canonical associative memory mechanism inside every block.

> **Caveat**  
> These numbers are obtained on a small, highly structured synthetic corpus designed to stress long-range associative recall. They are capability / smoke-test results, **not** claims of large-scale natural-language performance or Transformer-level sample efficiency on web text.

---

## Design Philosophy

- Keep the Phase-4.5 AAC equations **exactly** as specified.
- Add depth and non-local mixing via standard Transformer blocks *after* the associative core has already written to and read from persistent memory.
- Maintain full differentiability and causal autoregressive training.
- Provide both a pure AAC baseline and the hybrid for ablation and further research.

---

## Citation

If you use this code, please cite the original AAC work and this repository:

```bibtex
@software{aac_v12_canonical_transformer,
  title  = {AAC v1.2 Canonical Associative Architecture + Hybrid Canonical-Transformer},
  year   = {2026},
  url    = {https://github.com/<your-username>/aac-v1.2-canonical-transformer}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
