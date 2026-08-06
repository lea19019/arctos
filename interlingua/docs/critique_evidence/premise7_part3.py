"""VALIDITY: does the operationalized Delta_t measure LAG or SHARPNESS-DIFFERENCE?

Ground truth in every cell below: mu_m == mu_b == log10(1000). TRUE LAG = 0.
Only the transition WIDTH differs between the two measures.
A valid lag estimator must return 0. Anything else is confound, not lag.
"""
import numpy as np
from premise7_power import X, cp_binseg, cp_sigmoid_fit, minmax, sigmoid, boot_ci

rng = np.random.default_rng(11)
MU = 3.0
SIG_OBS = 0.03

def est(mu, w, det, rng):
    y = sigmoid(X, mu, w) + rng.normal(0, SIG_OBS, X.size)
    return det(minmax(y))

print("="*94)
print("C. SHARPNESS CONFOUND.  TRUE LAG = 0 IN EVERY ROW.  Only widths differ.")
print("   Reported: mean estimated Delta_t (log10 steps), and false-positive rate of")
print("   the 'CI excludes zero' test with n=5 seeds, sigma_seed=0.05.")
print("="*94)
print(f"  {'w_mech':>7} {'w_behav':>8} | {'binseg dt':>10} {'ratio':>7} {'FPR':>6} | {'sigfit dt':>10} {'ratio':>7} {'FPR':>6}")
for wm, wb in [(0.30,0.30),(0.15,0.30),(0.30,0.15),(0.10,0.40),(0.40,0.10),(0.08,0.50),(0.50,0.08)]:
    out = {}
    for det, nm in ((cp_binseg,'binseg'), (cp_sigmoid_fit,'sigfit')):
        dts_all, fp = [], 0
        for _ in range(300):
            dts = []
            for _ in range(5):
                j = rng.normal(0, 0.05)          # common seed jitter, rho=1
                dts.append(est(MU+j, wb, det, rng) - est(MU+j, wm, det, rng))
            dts_all.append(np.nanmean(dts))
            lo, hi = boot_ci(dts, B=1200, rng=rng)
            if np.isfinite(lo) and (lo > 0 or hi < 0):
                fp += 1
        out[nm] = (np.nanmean(dts_all), fp/300)
    b, s = out['binseg'], out['sigfit']
    print(f"  {wm:7.2f} {wb:8.2f} | {b[0]:10.3f} {10**b[0]:7.2f} {b[1]:6.2f} |"
          f" {s[0]:10.3f} {10**s[0]:7.2f} {s[1]:6.2f}")

print()
print("="*94)
print("D. FLOOR/CEILING TRUNCATION: what if a measure has not finished transitioning")
print("   by step 7600 (mu near or past the end of training)? TRUE LAG = 0.20 log10.")
print("="*94)
for mu_m in (2.0, 2.5, 3.0, 3.3, 3.6, 3.8):
    dts = []
    for _ in range(300):
        j = rng.normal(0, 0.05)
        dts.append(est(mu_m+0.20+j, 0.30, cp_binseg, rng) - est(mu_m+j, 0.30, cp_binseg, rng))
    d = np.nanmean(dts)
    print(f"  mu_mech = {mu_m:.1f} (step {10**mu_m:7.0f})  estimated Delta_t = {d:.3f}"
          f"   (true 0.200)   bias = {d-0.20:+.3f}")

print()
print("="*94)
print("E. RESOLUTION FLOOR: sd of a single changepoint estimate at zero seed jitter,")
print("   pure quantization + sigma_obs=0.03. This is the noise floor per run.")
print("="*94)
for w in (0.10, 0.20, 0.30, 0.50):
    for det, nm in ((cp_binseg,'binseg'),(cp_sigmoid_fit,'sigfit')):
        v = [est(MU, w, det, rng) for _ in range(600)]
        print(f"  w={w:.2f} {nm:7s} mean={np.nanmean(v):.3f} sd={np.nanstd(v):.4f} log10 steps")
