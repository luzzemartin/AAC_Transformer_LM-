#!/usr/bin/env python3
"""Train the 8-block Canonical-Transformer hybrid LM and report metrics."""
from __future__ import annotations
import json, math, random, time
from pathlib import Path

import torch
import torch.nn.functional as F

from aac_lm.canonical_transformer import (
    CanonicalTransformerLM,
    ByteTokenizer,
    synthetic_corpus,
    make_sequences,
    evaluate,
    batchify,
)


def train(
    model,
    train_chunks,
    val_chunks,
    device,
    epochs=3,
    batch_size=8,
    lr=3e-4,
    log_every=30,
):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    step = 0
    history = []
    for ep in range(epochs):
        random.shuffle(train_chunks)
        model.train()
        for i in range(0, len(train_chunks), batch_size):
            b = batchify(train_chunks[i : i + batch_size], device)
            logits = model(b[:, :-1])
            y = b[:, 1:]
            loss = F.cross_entropy(logits.reshape(-1, model.vocab_size), y.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % log_every == 0:
                val = evaluate(model, val_chunks, device, batch_size, max_batches=12)
                history.append(
                    {
                        "step": step,
                        "epoch": ep + 1,
                        "train_loss": float(loss.detach()),
                        "val": val,
                    }
                )
                print(
                    f"step={step:4d} train_nll={float(loss):.4f} "
                    f"val_ppl={val['ppl']:.2f} val_acc={val['accuracy']:.3f}",
                    flush=True,
                )
    return history


def main():
    random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cpu")
    out_dir = Path("/home/workdir/artifacts/AAC_v1.2/ct_run")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Model config – 8 hybrid blocks (memory-conscious for limited RAM)
    d_model = 48
    n_layers = 8
    n_heads = 4
    seq_len = 48
    batch_size = 4
    epochs = 1
    lr = 3e-4

    print("=== Canonical-Transformer LM (8 hybrid blocks) ===")
    print(f"d_model={d_model}  n_layers={n_layers}  n_heads={n_heads}")

    model = CanonicalTransformerLM(
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        seed=42,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")

    # Data
    tok = ByteTokenizer()
    text = synthetic_corpus(n=50000)
    ids = tok.encode(text)
    cut = int(0.9 * len(ids))
    train_ids, val_ids = ids[:cut], ids[cut:]
    train_chunks = make_sequences(train_ids, seq_len, seq_len // 2)
    val_chunks = make_sequences(val_ids, seq_len, seq_len)
    print(
        f"Corpus bytes: {len(text.encode()):,} | "
        f"train chunks: {len(train_chunks)} | val chunks: {len(val_chunks)}"
    )

    # Baseline (random init)
    before = evaluate(model, val_chunks, device, batch_size)
    print("\n=== BEFORE TRAINING ===")
    print(f"  loss (NLL) : {before['loss']:.4f}")
    print(f"  perplexity : {before['ppl']:.2f}")
    print(f"  accuracy   : {before['accuracy']*100:.2f}%")

    t0 = time.time()
    print(f"\n=== TRAINING for {epochs} epochs ===")
    history = train(
        model,
        train_chunks,
        val_chunks,
        device,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        log_every=40,
    )
    elapsed = time.time() - t0

    after = evaluate(model, val_chunks, device, batch_size)
    print("\n=== AFTER TRAINING ===")
    print(f"  loss (NLL) : {after['loss']:.4f}")
    print(f"  perplexity : {after['ppl']:.2f}")
    print(f"  accuracy   : {after['accuracy']*100:.2f}%")
    print(f"  wall time  : {elapsed/60:.1f} min")

    # Generation / fluency
    print("\n=== GENERATION SAMPLES ===")
    prompts = [
        "the researcher ada chose the color ",
        "the researcher ben chose the color ",
        "later the record says that ",
        "the system observed a long sequence of ",
    ]
    samples = []
    for p in prompts:
        prompt_ids = tok.encode(p, add_bos=True, add_eos=False)
        gen_ids = model.generate(prompt_ids, max_new_tokens=70, temperature=0.7, top_k=40)
        text_out = tok.decode(gen_ids[1:])
        samples.append({"prompt": p, "generation": text_out})
        print(f"Prompt : {p!r}")
        print(f"Output : {text_out!r}\n")

    print("--- lower temperature (0.45) ---")
    for p in prompts[:2]:
        prompt_ids = tok.encode(p, add_bos=True, add_eos=False)
        gen_ids = model.generate(prompt_ids, max_new_tokens=50, temperature=0.45, top_k=25)
        text_out = tok.decode(gen_ids[1:])
        print(f"Prompt : {p!r}")
        print(f"Output : {text_out!r}\n")

    # Save
    results = {
        "before": before,
        "after": after,
        "history": history,
        "samples": samples,
        "config": {
            "d_model": d_model,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "seq_len": seq_len,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "params": n_params,
            "architecture": "8x (CanonicalAAC → TransformerBlock)",
        },
        "elapsed_sec": elapsed,
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    torch.save(
        {"model": model.state_dict(), "config": results["config"]},
        out_dir / "canonical_transformer_8blocks.pt",
    )
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
