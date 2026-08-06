import numpy as np
from premise7_power import (CKPTS, X, cp_binseg, cp_sigmoid_fit, minmax,
                            power, boot_ci, sigmoid, make_run, trial)

print("="*90)
print("A. BOOTSTRAP COVERAGE OF THE PERCENTILE CI AT SMALL n  (nominal 95%)")
print("="*90)
rng = np.random.default_rng(7)
for n in (5, 10, 20, 50):
    for dist, nm in ((lambda k: rng.normal(0,1,k), "normal"),
                     (lambda k: rng.standard_t(3, k), "heavy-tail t3"),
                     (lambda k: rng.lognormal(0,1,k), "skewed lognorm")):
        cov = 0; T = 4000
        for _ in range(T):
            v = dist(n)
            lo, hi = boot_ci(v, B=1200, rng=rng)
            true = 0.0 if nm != "skewed lognorm" else np.exp(0.5)
            cov += (lo <= true <= hi)
        print(f"  n={n:3d}  {nm:16s} coverage = {cov/T:.3f}   (miss rate {1-cov/T:.3f} vs nominal 0.050)")

print()
print("="*90)
print("B. POWER CURVE.  detector = single-changepoint L2 (PELT/Binseg class)")
print("   sigma_obs=0.03 (post-minmax), w=0.30 log10 both measures, mu0=log10(1000)")
print("="*90)
deltas = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00]
for rho in (0.0, 0.5, 0.9):
    for ss in (0.05, 0.10, 0.20):
        for n in (5, 10, 20):
            row = [power(d, n, ss, rho=rho, detector=cp_binseg, n_trials=300)
                   for d in deltas]
            print(f"  rho={rho:.1f} sigma_seed={ss:.2f} n={n:2d} | " +
                  " ".join(f"{p:.2f}" for p in row))
    print(f"  {'-'*80}")
print("  delta (log10 steps):        " + " ".join(f"{d:.2f}" for d in deltas))
print("  as step-ratio:              " + " ".join(f"{10**d:.2f}" for d in deltas))
