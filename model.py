"""Canonical AAC v1.2 Phase-4.5 language model.

Canonical core:
  S_t = tanh(W_hh S_{t-1} + W_xh e_t + b_h)
  T_t = decay_t T_{t-1} + g_keep,t v_t
  p_t = sigmoid(Value(S_t,T_t,e_t))
  P_t = 0.9999 P_{t-1} + p_t outer(k_t,v_t)
  r_t = q_t^T P_t
  h_t = tanh(W_comb [S_t;r_t] + b_comb)
  logits_t = h_t E^T + b_out

This is the canonical AACDiffFinal choice:
utility/value promotion + fixed lambda=0.9999 + direct write,
with no matrix reinforce/evict and T used as evidence for utility.
"""
from __future__ import annotations
import argparse, json, math, os, re, random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ByteTokenizer:
    """Lossless UTF-8 byte tokenizer with three reserved IDs."""
    PAD, BOS, EOS = 256, 257, 258
    vocab_size = 259
    def encode(self, text: str, add_bos=True, add_eos=True) -> List[int]:
        ids = list(text.encode('utf-8'))
        if add_bos: ids.insert(0, self.BOS)
        if add_eos: ids.append(self.EOS)
        return ids
    def decode(self, ids: List[int]) -> str:
        bs = bytes(int(i) for i in ids if 0 <= int(i) < 256)
        return bs.decode('utf-8', errors='replace')


class CanonicalAACLM(nn.Module):
    """Batched differentiable implementation of AACDiffFinal for causal LM."""
    def __init__(self, vocab_size=259, d_model=64, value_hidden=32,
                 decay_p=0.9999, decay_t=0.9, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.decay_p = decay_p
        self.decay_t = decay_t
        self.emb = nn.Embedding(vocab_size, d_model)
        # Canonical fast state S.
        self.Whh = nn.Parameter(torch.empty(d_model, d_model))
        self.Wxh = nn.Parameter(torch.empty(d_model, d_model))
        self.bh = nn.Parameter(torch.zeros(d_model))
        # Binding/read maps.
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        # Temporary-trace keep gate.
        self.W_keep = nn.Linear(2*d_model, 1)
        # Utility/value estimator -> promotion probability.
        self.Wv1 = nn.Linear(3*d_model, value_hidden)
        self.Wv2 = nn.Linear(value_hidden, 1)
        # Readout from fast state + associative read.
        self.Wcomb = nn.Linear(2*d_model, d_model)
        self.bcomb = nn.Parameter(torch.zeros(d_model))
        # Tied output head uses emb.weight.T; only bias is separate.
        self.b_out = nn.Parameter(torch.zeros(vocab_size))
        self.reset_parameters(seed)

    def reset_parameters(self, seed=0):
        g = torch.Generator(device=self.emb.weight.device).manual_seed(seed + 17)
        nn.init.normal_(self.emb.weight, mean=0, std=1/math.sqrt(self.d_model), generator=g)
        # Stable recurrent initialization. This is the trainable counterpart
        # of the canonical structured fast state; a later structured variant
        # can replace Whh, but this LM keeps the Phase-4.5 equations intact.
        A = torch.randn(self.d_model, self.d_model, generator=g) / math.sqrt(self.d_model)
        # contractive spectral scaling
        with torch.no_grad():
            rho = torch.linalg.matrix_norm(A, ord=2)
            self.Whh.copy_(0.80 * A / rho)
            nn.init.normal_(self.Wxh, 0, 1/math.sqrt(self.d_model), generator=g)
            for m in [self.Wk, self.Wv, self.Wq, self.W_keep, self.Wv1, self.Wv2, self.Wcomb]:
                for p in m.parameters():
                    if p.ndim > 1: nn.init.xavier_uniform_(p, generator=g)
                    else: nn.init.zeros_(p)
            # High initial persistence is canonical; fixed, not learned.
            self.b_out.zero_()

    def forward(self, x: torch.Tensor, return_aux=False, memory_scale=1.0):
        # x: [B,T]
        B,T = x.shape
        E = self.emb(x)  # [B,T,D]
        S = torch.zeros(B, self.d_model, device=x.device, dtype=E.dtype)
        Tr = torch.zeros_like(S)
        P = torch.zeros(B, self.d_model, self.d_model, device=x.device, dtype=E.dtype)
        logits = []
        p_proms, keep_gates, trace_norms, memory_norms = [], [], [], []
        prev = None
        for t in range(T):
            e = E[:,t]
            S = torch.tanh(F.linear(S, self.Whh) + F.linear(e, self.Wxh) + self.bh)
            if prev is None:
                keep = torch.zeros(B,1,device=x.device,dtype=E.dtype)
                p_prom = torch.zeros_like(keep)
                k = torch.zeros_like(S); v = torch.zeros_like(S)
            else:
                k = self.Wk(prev)
                v = self.Wv(e)
                keep = torch.sigmoid(self.W_keep(torch.cat([S,e], dim=-1)))
                Tr = self.decay_t * Tr + keep * v
                val_in = torch.cat([S, Tr, e], dim=-1)
                p_prom = torch.sigmoid(self.Wv2(torch.tanh(self.Wv1(val_in))))
                # Canonical fixed lambda + direct write.
                P = self.decay_p * P + p_prom.unsqueeze(-1) * torch.bmm(k.unsqueeze(2), v.unsqueeze(1))
            q = self.Wq(e)
            read = memory_scale * torch.bmm(q.unsqueeze(1), P).squeeze(1)
            h = torch.tanh(self.Wcomb(torch.cat([S, read], dim=-1)) + self.bcomb)
            # tied output head
            logits.append(h @ self.emb.weight.T + self.b_out)
            p_proms.append(p_prom.squeeze(-1)); keep_gates.append(keep.squeeze(-1))
            trace_norms.append(Tr.norm(dim=-1)); memory_norms.append(P.flatten(1).norm(dim=-1))
            prev = e
        out = torch.stack(logits, dim=1)
        if return_aux:
            return out, {
                'promotion': torch.stack(p_proms,1),
                'keep': torch.stack(keep_gates,1),
                'trace_norm': torch.stack(trace_norms,1),
                'memory_norm': torch.stack(memory_norms,1),
            }
        return out

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens=100, temperature=0.8, top_k=40):
        self.eval()
        ids = list(prompt_ids)
        for _ in range(max_new_tokens):
            # Context window is only a compute bound; AAC's P is causal within it.
            x = torch.tensor([ids[-512:]], dtype=torch.long, device=next(self.parameters()).device)
            logits = self(x)[:,-1]
            logits = logits / max(temperature,1e-5)
            if top_k:
                vals, inds = torch.topk(logits, min(top_k, logits.size(-1)))
                probs = torch.softmax(vals, -1)
                nxt = inds.gather(-1, torch.multinomial(probs, 1)).item()
            else:
                nxt = torch.multinomial(torch.softmax(logits,-1),1).item()
            ids.append(nxt)
            if nxt == ByteTokenizer.EOS: break
        return ids


def make_sequences(ids, seq_len=128, stride=None):
    stride = stride or seq_len
    xs=[]
    for i in range(0, max(0,len(ids)-seq_len), stride):
        chunk=ids[i:i+seq_len+1]
        if len(chunk)==seq_len+1: xs.append(chunk)
    return xs


def batchify(chunks, device):
    return torch.tensor(chunks, dtype=torch.long, device=device)


def evaluate(model, chunks, device, batch_size=16):
    model.eval(); total_loss=0.; total_tok=0; correct=0
    with torch.no_grad():
        for i in range(0,len(chunks),batch_size):
            b=batchify(chunks[i:i+batch_size],device)
            logits=model(b[:,:-1]); y=b[:,1:]
            loss=F.cross_entropy(logits.reshape(-1,model.vocab_size),y.reshape(-1),reduction='sum')
            total_loss += loss.item(); total_tok += y.numel()
            correct += (logits.argmax(-1)==y).sum().item()
    return {'loss':total_loss/total_tok,'ppl':math.exp(min(20,total_loss/total_tok)),'accuracy':correct/total_tok}


def train(model, train_chunks, val_chunks, device, epochs=3, batch_size=16, lr=3e-4, log_every=100):
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=0.01)
    step=0
    history=[]
    for ep in range(epochs):
        random.shuffle(train_chunks)
        model.train()
        for i in range(0,len(train_chunks),batch_size):
            b=batchify(train_chunks[i:i+batch_size],device)
            logits=model(b[:,:-1]); y=b[:,1:]
            loss=F.cross_entropy(logits.reshape(-1,model.vocab_size),y.reshape(-1))
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            step+=1
            if step % log_every == 0:
                val=evaluate(model,val_chunks,device,batch_size)
                history.append({'step':step,'epoch':ep+1,'train_loss':float(loss),'val':val})
                print(f'step={step:4d} train_nll={float(loss):.4f} val_ppl={val["ppl"]:.2f} val_acc={val["accuracy"]:.3f}',flush=True)
    return history


def load_corpus(root):
    texts=[]
    for p in sorted(Path(root).rglob('*.md')):
        try: texts.append(p.read_text(encoding='utf-8'))
        except: pass
    text='\n\n'.join(texts)
    # Avoid exact result JSON / giant logs; markdown is documentation corpus.
    return text


def synthetic_corpus(n=20000):
    """A structured language-like corpus stressing long-range retrieval."""
    parts=[]
    names=['ada','ben','cara','dax','eli','faye','glen','hana']
    colors=['red','blue','green','gold','white','black']
    for i in range(n//100):
        a=names[i%len(names)]; b=colors[(i*3)%len(colors)]; c=names[(i*5+2)%len(names)]
        parts.append(f"the researcher {a} chose the color {b}. "+
                     "the system observed a long sequence of ordinary symbols. "*3+
                     f"later the record says that {a} chose {b}, while {c} stayed nearby.\n")
    return ''.join(parts)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',default='/mnt/data/aac_lm_run'); ap.add_argument('--epochs',type=int,default=3); ap.add_argument('--seq-len',type=int,default=128); ap.add_argument('--d-model',type=int,default=64); ap.add_argument('--batch-size',type=int,default=16); ap.add_argument('--lr',type=float,default=3e-4); ap.add_argument('--seed',type=int,default=0); ap.add_argument('--synthetic',action='store_true'); ap.add_argument('--steps-log',type=int,default=50)
    args=ap.parse_args(); os.makedirs(args.out,exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    tok=ByteTokenizer()
    text=synthetic_corpus() if args.synthetic else load_corpus('/mnt/data/aac_work/AAC_v1.2')
    ids=tok.encode(text)
    cut=int(.9*len(ids)); train_ids, val_ids=ids[:cut], ids[cut:]
    train_chunks=make_sequences(train_ids,args.seq_len,args.seq_len//2)
    val_chunks=make_sequences(val_ids,args.seq_len,args.seq_len)
    device=torch.device('cpu')
    model=CanonicalAACLM(d_model=args.d_model,seed=args.seed).to(device)
    print(json.dumps({'corpus_bytes':len(text.encode()),'train_chunks':len(train_chunks),'val_chunks':len(val_chunks),'vocab':tok.vocab_size,'params':sum(p.numel() for p in model.parameters()),'canonical':{'lambda_p':0.9999,'lambda_t':0.9,'write':'direct','lifecycle':'no_matrix_reinforce_evict'}},indent=2))
    before=evaluate(model,val_chunks,device,args.batch_size); print('before',before)
    hist=train(model,train_chunks,val_chunks,device,args.epochs,args.batch_size,args.lr,args.steps_log)
    after=evaluate(model,val_chunks,device,args.batch_size); print('after',after)
    prompt=tok.encode('the researcher ada chose the color ',add_bos=True,add_eos=False)
    gen=tok.decode(model.generate(prompt,80,0.8,40)[1:])
    print('sample:',repr(gen))
    torch.save({'model':model.state_dict(),'config':vars(args)},Path(args.out)/'aac_canonical_lm.pt')
    (Path(args.out)/'results.json').write_text(json.dumps({'before':before,'after':after,'history':hist,'sample':gen},indent=2))

if __name__=='__main__': main()
