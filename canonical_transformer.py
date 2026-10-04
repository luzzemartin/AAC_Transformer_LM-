"""Canonical-Transformer hybrid language model (v1.2).

Architecture
------------
Data flows through 8 identical hybrid blocks.  Inside each block:

  1. Canonical AAC (Phase-4.5) processes the sequence recurrently,
     producing a sequence of hidden states that incorporate
     persistent associative memory.
  2. A standard pre-norm causal Transformer block (MHA + FFN)
     is applied to that sequence.

This preserves the canonical AAC equations exactly while adding
depth and non-local mixing via the transformer sub-layers.

Canonical core (unchanged):
  S_t = tanh(W_hh S_{t-1} + W_xh e_t + b_h)
  T_t = λ_T T_{t-1} + g_keep,t · v_t
  p_t = σ(Value(S_t, T_t, e_t))
  P_t = λ_P P_{t-1} + p_t · (k_t ⊗ v_t)          # λ_P = 0.9999 fixed
  r_t = q_tᵀ P_t
  h_t = tanh(W_comb [S_t ; r_t] + b_comb)
"""
from __future__ import annotations
import math
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Tokenizer (identical to the original)
# ---------------------------------------------------------------------------
class ByteTokenizer:
    PAD, BOS, EOS = 256, 257, 258
    vocab_size = 259

    def encode(self, text: str, add_bos=True, add_eos=True) -> List[int]:
        ids = list(text.encode("utf-8"))
        if add_bos:
            ids.insert(0, self.BOS)
        if add_eos:
            ids.append(self.EOS)
        return ids

    def decode(self, ids: List[int]) -> str:
        bs = bytes(int(i) for i in ids if 0 <= int(i) < 256)
        return bs.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Canonical AAC core (one full sequence pass)
# ---------------------------------------------------------------------------
class CanonicalAACCore(nn.Module):
    """Batched, sequence-level implementation of the Phase-4.5 AAC equations."""

    def __init__(
        self,
        d_model: int,
        value_hidden: int = 32,
        decay_p: float = 0.9999,
        decay_t: float = 0.9,
    ):
        super().__init__()
        self.d_model = d_model
        self.decay_p = decay_p
        self.decay_t = decay_t

        # Fast recurrent state
        self.Whh = nn.Parameter(torch.empty(d_model, d_model))
        self.Wxh = nn.Parameter(torch.empty(d_model, d_model))
        self.bh = nn.Parameter(torch.zeros(d_model))

        # Key / value / query projections
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wq = nn.Linear(d_model, d_model, bias=False)

        # Temporary-trace keep gate
        self.W_keep = nn.Linear(2 * d_model, 1)

        # Utility / value estimator → promotion probability
        self.Wv1 = nn.Linear(3 * d_model, value_hidden)
        self.Wv2 = nn.Linear(value_hidden, 1)

        # Combine fast state + associative read
        self.Wcomb = nn.Linear(2 * d_model, d_model)
        self.bcomb = nn.Parameter(torch.zeros(d_model))

        self._reset_parameters()

    def _reset_parameters(self):
        g = torch.Generator(device=self.Whh.device).manual_seed(17)
        A = torch.randn(self.d_model, self.d_model, generator=g) / math.sqrt(self.d_model)
        with torch.no_grad():
            rho = torch.linalg.matrix_norm(A, ord=2)
            self.Whh.copy_(0.80 * A / rho)
            nn.init.normal_(self.Wxh, 0, 1 / math.sqrt(self.d_model), generator=g)
            for m in [self.Wk, self.Wv, self.Wq, self.W_keep, self.Wv1, self.Wv2, self.Wcomb]:
                for p in m.parameters():
                    if p.ndim > 1:
                        nn.init.xavier_uniform_(p)
                    else:
                        nn.init.zeros_(p)

    def forward(
        self,
        E: torch.Tensor,                 # [B, T, D]
        memory_scale: float = 1.0,
        return_aux: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict]:
        B, T, D = E.shape
        device, dtype = E.device, E.dtype

        S = torch.zeros(B, D, device=device, dtype=dtype)
        Tr = torch.zeros_like(S)
        P = torch.zeros(B, D, D, device=device, dtype=dtype)

        hs = []
        p_proms, keeps, mem_norms = [], [], []

        prev = None
        for t in range(T):
            e = E[:, t]                                      # [B, D]
            S = torch.tanh(F.linear(S, self.Whh) + F.linear(e, self.Wxh) + self.bh)

            if prev is None:
                keep = torch.zeros(B, 1, device=device, dtype=dtype)
                p_prom = torch.zeros_like(keep)
                k = torch.zeros_like(S)
                v = torch.zeros_like(S)
            else:
                k = self.Wk(prev)
                v = self.Wv(e)
                keep = torch.sigmoid(self.W_keep(torch.cat([S, e], dim=-1)))
                Tr = self.decay_t * Tr + keep * v
                val_in = torch.cat([S, Tr, e], dim=-1)
                p_prom = torch.sigmoid(self.Wv2(torch.tanh(self.Wv1(val_in))))
                # Canonical fixed-λ direct outer-product write
                P = self.decay_p * P + p_prom.unsqueeze(-1) * torch.bmm(
                    k.unsqueeze(2), v.unsqueeze(1)
                )

            q = self.Wq(e)
            read = memory_scale * torch.bmm(q.unsqueeze(1), P).squeeze(1)
            h = torch.tanh(self.Wcomb(torch.cat([S, read], dim=-1)) + self.bcomb)
            hs.append(h)

            p_proms.append(p_prom.squeeze(-1))
            keeps.append(keep.squeeze(-1))
            mem_norms.append(P.flatten(1).norm(dim=-1))
            prev = e

        H = torch.stack(hs, dim=1)                           # [B, T, D]
        if return_aux:
            return H, {
                "promotion": torch.stack(p_proms, 1),
                "keep": torch.stack(keeps, 1),
                "memory_norm": torch.stack(mem_norms, 1),
            }
        return H


# ---------------------------------------------------------------------------
# Standard pre-norm causal Transformer block
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)                      # [3, B, H, T, Hd]
        q, k, v = qkv[0], qkv[1], qkv[2]

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        if mask is not None:
            att = att.masked_fill(mask[:, None, None, :], float("-inf"))
        else:
            # causal mask
            causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            att = att.masked_fill(causal, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = (att @ v).transpose(1, 2).reshape(B, T, D)
        return self.proj(y)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block (Attention + FFN)."""

    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Hybrid Canonical → Transformer block
# ---------------------------------------------------------------------------
class CanonicalTransformerBlock(nn.Module):
    """One hybrid block: Canonical AAC → Transformer."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        value_hidden: int = 32,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        decay_p: float = 0.9999,
        decay_t: float = 0.9,
    ):
        super().__init__()
        self.aac = CanonicalAACCore(
            d_model=d_model,
            value_hidden=value_hidden,
            decay_p=decay_p,
            decay_t=decay_t,
        )
        self.transformer = TransformerBlock(d_model, n_heads, mlp_ratio, dropout)
        # residual scale / optional projection if needed later
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, memory_scale: float = 1.0) -> torch.Tensor:
        # 1. Canonical AAC over the sequence
        h = self.aac(x, memory_scale=memory_scale)           # [B, T, D]
        # residual connection around AAC
        h = x + h
        # 2. Transformer block
        h = self.transformer(h)
        return self.norm(h)


# ---------------------------------------------------------------------------
# Full language model
# ---------------------------------------------------------------------------
class CanonicalTransformerLM(nn.Module):
    """8-block Canonical-Transformer language model."""

    def __init__(
        self,
        vocab_size: int = 259,
        d_model: int = 64,
        n_layers: int = 8,
        n_heads: int = 4,
        value_hidden: int = 32,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        seed: int = 0,
    ):
        super().__init__()
        torch.manual_seed(seed)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers

        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)    # simple absolute PE
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            CanonicalTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                value_hidden=value_hidden,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        # tied output head
        self.b_out = nn.Parameter(torch.zeros(vocab_size))

        self.apply(self._init_weights)
        # special init for AAC cores already done inside them

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        x: torch.Tensor,                     # [B, T]
        memory_scale: float = 1.0,
        return_aux: bool = False,
    ) -> torch.Tensor:
        B, T = x.shape
        device = x.device
        pos = torch.arange(T, device=device).unsqueeze(0)

        h = self.emb(x) + self.pos_emb(pos)
        h = self.drop(h)

        for block in self.blocks:
            h = block(h, memory_scale=memory_scale)

        h = self.ln_f(h)
        # tied head
        logits = h @ self.emb.weight.T + self.b_out
        return logits

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: List[int],
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 40,
    ) -> List[int]:
        self.eval()
        ids = list(prompt_ids)
        for _ in range(max_new_tokens):
            x = torch.tensor([ids[-512:]], dtype=torch.long, device=next(self.parameters()).device)
            logits = self(x)[:, -1]
            logits = logits / max(temperature, 1e-5)
            if top_k:
                vals, inds = torch.topk(logits, min(top_k, logits.size(-1)))
                probs = torch.softmax(vals, -1)
                nxt = inds.gather(-1, torch.multinomial(probs, 1)).item()
            else:
                nxt = torch.multinomial(torch.softmax(logits, -1), 1).item()
            ids.append(nxt)
            if nxt == ByteTokenizer.EOS:
                break
        return ids


# ---------------------------------------------------------------------------
# Training utilities (shared style with original)
# ---------------------------------------------------------------------------
def make_sequences(ids, seq_len=128, stride=None):
    stride = stride or seq_len
    xs = []
    for i in range(0, max(0, len(ids) - seq_len), stride):
        chunk = ids[i : i + seq_len + 1]
        if len(chunk) == seq_len + 1:
            xs.append(chunk)
    return xs


def batchify(chunks, device):
    return torch.tensor(chunks, dtype=torch.long, device=device)


def evaluate(model, chunks, device, batch_size=8, max_batches=None):
    """Evaluate; optionally limit number of batches for fast logging."""
    model.eval()
    total_loss = 0.0
    total_tok = 0
    correct = 0
    n_batches = 0
    with torch.no_grad():
        for i in range(0, len(chunks), batch_size):
            if max_batches is not None and n_batches >= max_batches:
                break
            b = batchify(chunks[i : i + batch_size], device)
            logits = model(b[:, :-1])
            y = b[:, 1:]
            loss = F.cross_entropy(
                logits.reshape(-1, model.vocab_size), y.reshape(-1), reduction="sum"
            )
            total_loss += loss.item()
            total_tok += y.numel()
            correct += (logits.argmax(-1) == y).sum().item()
            n_batches += 1
    return {
        "loss": total_loss / max(1, total_tok),
        "ppl": math.exp(min(20, total_loss / max(1, total_tok))),
        "accuracy": correct / max(1, total_tok),
    }


def synthetic_corpus(n=40000):
    parts = []
    names = ["ada", "ben", "cara", "dax", "eli", "faye", "glen", "hana"]
    colors = ["red", "blue", "green", "gold", "white", "black"]
    for i in range(n // 100):
        a = names[i % len(names)]
        b = colors[(i * 3) % len(colors)]
        c = names[(i * 5 + 2) % len(names)]
        parts.append(
            f"the researcher {a} chose the color {b}. "
            + "the system observed a long sequence of ordinary symbols. " * 3
            + f"later the record says that {a} chose {b}, while {c} stayed nearby.\n"
        )
    return "".join(parts)
