import random, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from aac_lm import CanonicalAACLM
import torch, torch.nn.functional as F

K=4
V=['<pad>']+[f'K{i}' for i in range(K)]+[f'V{i}' for i in range(K)]+[f'Q{i}' for i in range(K)]+['x','y','z','a']
ix={s:i for i,s in enumerate(V)}

def data(n=800,gap=20,seed=0):
 r=random.Random(seed); out=[]
 for _ in range(n):
  perm=list(range(K)); r.shuffle(perm)
  s=[]
  for i in range(K): s += [ix[f'K{i}'], ix[f'V{perm[i]}']]
  s += [ix['x'],ix['y'],ix['z']]*gap
  for i in range(K): s += [ix[f'Q{i}'], ix[f'V{perm[i]}']]
  out.append(s)
 return out

def run():
 random.seed(0); torch.manual_seed(0)
 d=data(n=500,gap=10); tr=d[:400]; va=d[400:]
 m=CanonicalAACLM(vocab_size=len(V),d_model=24,seed=0)
 opt=torch.optim.AdamW(m.parameters(),lr=2e-3)
 for step in range(450):
  b=torch.tensor(random.sample(tr,16)); lg=m(b[:,:-1]); y=b[:,1:]
  loss=F.cross_entropy(lg.reshape(-1,len(V)),y.reshape(-1))
  opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
 with torch.no_grad():
  qhit=qtot=nom=0; allhit=alltot=0
  for seq in va:
   b=torch.tensor([seq]); inp=b[:,:-1]; y=b[:,1:]; lg=m(inp); lgn=m(inp,memory_scale=0)
   allhit += (lg.argmax(-1)==y).sum().item(); alltot += y.numel()
   for pos in range(len(seq)-1):
    if ix[f'Q0'] <= seq[pos] < ix[f'Q0']+K:
     qtot+=1; expected=seq[pos-K-0] if False else None
     # target is the following V token, so use ground truth directly.
     target=seq[pos+1]
     qhit += int(lg[0,pos].argmax().item()==target)
     nom += int(lgn[0,pos].argmax().item()==target)
  print({'loss':float(loss),'accuracy':allhit/alltot,'query_value_accuracy':qhit/qtot,'query_value_accuracy_no_memory':nom/qtot,'queries':qtot})
if __name__=='__main__':run()
