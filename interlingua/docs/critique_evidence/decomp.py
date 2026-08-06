import numpy as np, os
os.environ["HF_HUB_OFFLINE"]="1"
from transformers import AutoTokenizer
tk=AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
D=np.load("D.npy",allow_pickle=True).item()
P={k:v.mean(0) for k,v in D.items()}
def jsd_terms(a,b):
    m=0.5*(a+b); eps=1e-30
    return 0.5*(a*np.log2((a+eps)/(m+eps))+b*np.log2((b+eps)/(m+eps)))
# EN-FR signal decomposition: which tokens drive (between - within)?
w=0.5*(jsd_terms(P["EN_SG"],P["FR_SG"])+jsd_terms(P["EN_PL"],P["FR_PL"]))
b=0.5*(jsd_terms(P["EN_SG"],P["FR_PL"])+jsd_terms(P["EN_PL"],P["FR_SG"]))
sig=b-w
print("EN-FR total signal (bits):", sig.sum())
idx=np.argsort(-np.abs(sig))[:25]
print(f"\n{'token':>18s} {'signal_contrib':>15s} {'P(EN_SG)':>10s} {'P(FR_SG)':>10s} {'P(FR_PL)':>10s}")
for i in idx:
    print(f"{repr(tk.decode([int(i)])):>18s} {sig[i]:+15.6f} {P['EN_SG'][i]:10.5f} {P['FR_SG'][i]:10.5f} {P['FR_PL'][i]:10.5f}")
# how concentrated?
s=np.sort(-np.abs(sig)); tot=np.abs(sig).sum()
for k in [5,20,100,1000]:
    print(f"top-{k} tokens carry {100*np.abs(sig)[np.argsort(-np.abs(sig))[:k]].sum()/tot:.1f}% of |signal|")
# language-exclusivity of mass
for a,bb in [("EN_SG","FR_SG"),("EN_SG","TR_SG")]:
    ov=np.minimum(P[a],P[bb]).sum()
    print(f"overlap mass {a} vs {bb}: {ov:.4f}  -> {100*(1-ov):.1f}% of mass is language-exclusive")
