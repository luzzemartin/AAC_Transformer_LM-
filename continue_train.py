#!/usr/bin/env python3
"""Continue training the canonical AAC LM from the provided synth checkpoint
and report loss / perplexity / accuracy + generation samples for fluency.
"""
from __future__ import annotations
import json, math, random, os
from pathlib import Path

import torch
import torch.nn.functional as F

from aac_lm.model import (
    CanonicalAACLM, ByteTokenizer, synthetic_corpus,
    make_sequences, evaluate, train, batchify
)

def main():
    random.seed(0)
    torch.manual_seed(0)
    device = torch.device("cpu")
    out_dir = Path("/home/workdir/artifacts/AAC_v1.2/continue_run")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint (d_model=48, 2 epochs already done)
    ckpt_path = Path("aac_lm/aac_canonical_lm_synth.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    d_model = cfg.get("d_model", 48)
    seq_len = cfg.get("seq_len", 96)
    batch_size = 16
    lr = 3e-4

    print(f"Loading checkpoint from {ckpt_path}")
    print(f"Config: d_model={d_model}, seq_len={seq_len}, previous epochs={cfg.get('epochs')}")

    model = CanonicalAACLM(d_model=d_model, seed=0).to(device)
    model.load_state_dict(ckpt["model"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Larger synthetic corpus for continued training
    tok = ByteTokenizer()
    text = synthetic_corpus(n=40000)  # ~2x original
    ids = tok.encode(text)
    cut = int(0.9 * len(ids))
    train_ids, val_ids = ids[:cut], ids[cut:]
    train_chunks = make_sequences(train_ids, seq_len, seq_len // 2)
    val_chunks = make_sequences(val_ids, seq_len, seq_len)
    print(f"Corpus bytes: {len(text.encode()):,} | train chunks: {len(train_chunks)} | val chunks: {len(val_chunks)}")

    # Baseline after loading checkpoint
    before = evaluate(model, val_chunks, device, batch_size)
    print("\n=== AFTER LOADING CHECKPOINT (baseline) ===")
    print(f"  loss (NLL) : {before['loss']:.4f}")
    print(f"  perplexity : {before['ppl']:.2f}")
    print(f"  accuracy   : {before['accuracy']*100:.2f}%")

    # Continue training for more epochs
    extra_epochs = 4
    print(f"\n=== CONTINUING TRAINING for {extra_epochs} more epochs ===")
    history = train(
        model, train_chunks, val_chunks, device,
        epochs=extra_epochs, batch_size=batch_size, lr=lr, log_every=40
    )

    after = evaluate(model, val_chunks, device, batch_size)
    print("\n=== AFTER CONTINUED TRAINING ===")
    print(f"  loss (NLL) : {after['loss']:.4f}")
    print(f"  perplexity : {after['ppl']:.2f}")
    print(f"  accuracy   : {after['accuracy']*100:.2f}%")

    # Fluency / generation samples
    print("\n=== GENERATION SAMPLES (fluency check) ===")
    prompts = [
        "the researcher ada chose the color ",
        "the researcher ben chose the color ",
        "later the record says that ",
        "the system observed a long sequence of ",
    ]
    samples = []
    for p in prompts:
        prompt_ids = tok.encode(p, add_bos=True, add_eos=False)
        gen_ids = model.generate(prompt_ids, max_new_tokens=80, temperature=0.7, top_k=40)
        text_out = tok.decode(gen_ids[1:])  # drop BOS
        samples.append({"prompt": p, "generation": text_out})
        print(f"Prompt : {p!r}")
        print(f"Output : {text_out!r}\n")

    # Also try a slightly lower temperature for more coherent samples
    print("--- lower temperature (0.5) ---")
    for p in prompts[:2]:
        prompt_ids = tok.encode(p, add_bos=True, add_eos=False)
        gen_ids = model.generate(prompt_ids, max_new_tokens=60, temperature=0.5, top_k=30)
        text_out = tok.decode(gen_ids[1:])
        print(f"Prompt : {p!r}")
        print(f"Output : {text_out!r}\n")

    # Save results + continued checkpoint
    results = {
        "baseline_after_load": before,
        "after_continued": after,
        "history": history,
        "samples": samples,
        "config": {
            "d_model": d_model,
            "seq_len": seq_len,
            "extra_epochs": extra_epochs,
            "batch_size": batch_size,
            "lr": lr,
            "params": n_params,
        },
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    torch.save(
        {"model": model.state_dict(), "config": results["config"]},
        out_dir / "aac_canonical_lm_continued.pt",
    )
    print(f"\nSaved results + checkpoint to {out_dir}")

if __name__ == "__main__":
    main()
