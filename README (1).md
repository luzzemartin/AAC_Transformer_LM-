# Canonical AAC v1.2 Language Model

This directory adds an autoregressive language-modeling stack around the **canonical Phase-4.5 `AACDiffFinal` architecture** from the supplied AAC v1.2 archive.

## Canonical core

At each token:

- token embedding `e_t`
- fast state `S_t`
- temporary evidence trace `T_t`
- learned utility/value promotion probability `p_t`
- persistent associative matrix `P_t`
- associative read `q_t^T P_t`
- output state and tied vocabulary head

The canonical Phase-4.5 choices are retained:

- fixed persistent decay `lambda_P = 0.9999`
- temporary trace decay `lambda_T = 0.9`
- **direct** `outer(k_t, v_t)` write into `P`
- no matrix-level reinforce/evict lifecycle
- `T` is evidence for the utility/value estimator, not the source of the persistent write
- causal/non-circular token order

## Language-model additions

- lossless UTF-8 byte tokenizer (256 byte symbols + BOS/EOS/PAD)
- learned token embeddings
- tied output projection (`hidden @ embedding.T`)
- autoregressive next-token cross entropy
- AdamW training and gradient clipping
- sampling/generation
- validation loss, perplexity, and token accuracy
- memory ablation support (`memory_scale=0`)

## Quick start

```bash
python aac_lm.py --synthetic --epochs 2 --seq-len 96 --d-model 48 --batch-size 16
```

For a controlled associative language test:

```bash
python aac_lm/test_mqar_lm.py
```

## Initial results

### Structured byte-level corpus

With 48-dimensional AAC and 2 epochs:

- validation perplexity: **263.2 -> 16.85**
- validation next-byte accuracy: **1.56% -> 30.98%**

### Randomized long-range associative LM

A smaller word-token MQAR-style test randomizes each key/value assignment per sequence. After 450 updates:

- overall token accuracy: **95.4%**
- query-value accuracy with AAC memory: **100.0%**
- query-value accuracy with persistent-memory reads ablated: **14.75%**

This demonstrates that the trained LM can use the persistent associative state to recover a value whose key occurred much earlier, rather than merely memorizing a fixed key/value mapping.

## Caveat

The supplied canonical Phase-4.5 implementation was developed around the synthetic associative-recall/MQAR task, not large-scale natural-language pretraining. The byte-level corpus result is therefore a capability smoke test, **not** a claim of transformer-level natural-language quality or scale efficiency.
