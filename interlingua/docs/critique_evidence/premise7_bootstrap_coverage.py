import numpy as np
from premise7_power import power, cp_binseg, cp_sigmoid_fit

# sigma_seed grounded in arXiv:2603.25009: Transformer grokking step 50,800 +/- 22,565
# CV = 0.444 -> sd in log10 units ~ log10(e)*CV = 0.434*0.444 = 0.193
# At optimal reg: 50,800 +/- 38,745 -> CV 0.763 -> sd_log10 = 0.331
# Optimistic (MLP, matched): 45,600 +/- 5,550 -> CV 0.122 -> sd_log10 = 0.053
for cv in (0.122, 0.444, 0.763):
    print(f"  CV={cv:.3f} -> sd_log10 = {0.4343*cv:.3f}  (step-ratio spread {10**(0.4343*cv):.2f}x)")
print()
deltas=[0.05,0.10,0.15,0.20,0.30,0.50,0.75,1.00,1.50]
print("MDE at 80% power, detector=binseg (the plan's method), rho = seed-jitter correlation")
print(f"{'rho':>4} {'sd_seed':>8} {'n':>3} | " + " ".join(f"{d:5.2f}" for d in deltas))
for rho in (0.0,0.5,0.9):
  for ss in (0.053,0.193,0.331):
    for n in (5,10,20):
        row=[power(d,n,ss,rho=rho,detector=cp_binseg,n_trials=250) for d in deltas]
        mde=next((d for d,p in zip(deltas,row) if p>=0.80), None)
        m = f"{mde:.2f} ({10**mde:.2f}x)" if mde else ">1.50 (>32x)"
        print(f"{rho:4.1f} {ss:8.3f} {n:3d} | " + " ".join(f"{p:5.2f}" for p in row) + f"  MDE={m}")
