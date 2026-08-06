"""
Power analysis for Delta_t = t_behavioral - t_mechanistic under the Tier-1
interlingua design:  60 log-spaced checkpoints, steps 1..7600, 5 seeds,
changepoint detection on the log(step) axis, percentile bootstrap over seeds.

All times are in log10(step) units. Resolution floor = one checkpoint interval.
"""
import numpy as np

RNG = np.random.default_rng(20260806)

# ---------------------------------------------------------------- checkpoints
N_CKPT_NOMINAL = 60
S_MAX = 7600
CKPTS = np.unique(np.round(np.logspace(0, np.log10(S_MAX), N_CKPT_NOMINAL)).astype(int))
X = np.log10(CKPTS.astype(float))          # sampling grid in log10(step)
DX_NOMINAL = np.log10(S_MAX) / (N_CKPT_NOMINAL - 1)

# ------------------------------------------------------------------ generator
def sigmoid(x, mu, w):
    return 1.0 / (1.0 + np.exp(-(x - mu) / w))

def make_run(mu_m, mu_b, w_m, w_b, sigma_obs, rng):
    """One seed: mechanistic + behavioral trajectory sampled at CKPTS."""
    m = sigmoid(X, mu_m, w_m) + rng.normal(0, sigma_obs, X.size)
    b = sigmoid(X, mu_b, w_b) + rng.normal(0, sigma_obs, X.size)
    return m, b

def minmax(y):
    lo, hi = y.min(), y.max()
    return (y - lo) / (hi - lo) if hi > lo else np.zeros_like(y)

# ------------------------------------------------------------------ detectors
def cp_binseg(y, x=X, min_size=3):
    """Single-changepoint L2 detector = PELT/Binseg with one breakpoint.
    Returns x-location of the best split (the plan's stated method class)."""
    n = y.size
    best, best_i = np.inf, None
    cs = np.cumsum(y); cs2 = np.cumsum(y**2)
    for i in range(min_size, n - min_size):
        n1, n2 = i, n - i
        s1, s2 = cs[i-1], cs[-1] - cs[i-1]
        q1, q2 = cs2[i-1], cs2[-1] - cs2[i-1]
        sse = (q1 - s1*s1/n1) + (q2 - s2*s2/n2)
        if sse < best:
            best, best_i = sse, i
    return x[best_i]

def cp_sigmoid_fit(y, x=X):
    """Generous alternative: least-squares sigmoid, return fitted midpoint.
    This is a *correctly specified* estimator -- an optimistic upper bound."""
    from scipy.optimize import curve_fit
    def f(xx, mu, w, a, c):
        return a / (1 + np.exp(-(xx - mu) / max(w, 1e-3))) + c
    try:
        p0 = [x.mean(), 0.3, y.max() - y.min(), y.min()]
        p, _ = curve_fit(f, x, y, p0=p0, maxfev=20000)
        mu = p[0]
        return float(np.clip(mu, x[0], x[-1]))
    except Exception:
        return np.nan

# ------------------------------------------------------------------ bootstrap
def boot_ci(vals, B=4000, rng=RNG, alpha=0.05):
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        return np.nan, np.nan
    idx = rng.integers(0, vals.size, size=(B, vals.size))
    means = vals[idx].mean(axis=1)
    return np.percentile(means, [100*alpha/2, 100*(1-alpha/2)])

# ------------------------------------------------------------------ one trial
def trial(delta, n_seeds, sigma_seed, sigma_obs, rho, mu0, w_m, w_b,
          detector, rng):
    dts = []
    for _ in range(n_seeds):
        # common (run-level) jitter shared by both measures + idiosyncratic part
        z_common = rng.normal(0, 1)
        z_m = rng.normal(0, 1); z_b = rng.normal(0, 1)
        jm = sigma_seed * (np.sqrt(rho)*z_common + np.sqrt(1-rho)*z_m)
        jb = sigma_seed * (np.sqrt(rho)*z_common + np.sqrt(1-rho)*z_b)
        mu_m = mu0 + jm
        mu_b = mu0 + delta + jb
        m, b = make_run(mu_m, mu_b, w_m, w_b, sigma_obs, rng)
        tm = detector(minmax(m)); tb = detector(minmax(b))
        dts.append(tb - tm)
    lo, hi = boot_ci(dts, rng=rng)
    if not np.isfinite(lo):
        return False, np.nan
    return (lo > 0), float(np.mean(dts))

def power(delta, n_seeds, sigma_seed, sigma_obs=0.03, rho=0.5, mu0=3.0,
          w_m=0.30, w_b=0.30, detector=cp_binseg, n_trials=400, seed=0):
    rng = np.random.default_rng(1000 + seed)
    hits = 0
    for _ in range(n_trials):
        ok, _ = trial(delta, n_seeds, sigma_seed, sigma_obs, rho, mu0,
                      w_m, w_b, detector, rng)
        hits += ok
    return hits / n_trials

# ------------------------------------------------------------------ reporting
def mdl(n_seeds, sigma_seed, detector, target=0.80, **kw):
    """Minimum detectable lag at `target` power, by grid + refine."""
    grid = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5]
    prev = 0.0
    for d in grid:
        p = power(d, n_seeds, sigma_seed, detector=detector, **kw)
        if p >= target:
            # linear interpolate between prev and d
            return d, p
        prev = d
    return np.inf, p

if __name__ == "__main__":
    print("=" * 78)
    print("CHECKPOINT GRID")
    print("=" * 78)
    print(f"nominal checkpoints          : {N_CKPT_NOMINAL}")
    print(f"unique after integer rounding: {CKPTS.size}")
    print(f"nominal log10 spacing        : {DX_NOMINAL:.4f}  (ratio {10**DX_NOMINAL:.4f}x)")
    print(f"checkpoints below step 760   : {(CKPTS<760).sum()}")
    print(f"checkpoints in [1000, 7600]  : {((CKPTS>=1000)&(CKPTS<=7600)).sum()}")
    print(f"absolute gap near step 1000  : {np.diff(CKPTS)[np.searchsorted(CKPTS,1000)-1]} steps")
    print(f"absolute gap near step 7000  : {np.diff(CKPTS)[-1]} steps")

    print()
    print("=" * 78)
    print("TYPE-I ERROR CHECK  (delta = 0, should be <= 0.05)")
    print("=" * 78)
    for ss in (0.02, 0.05, 0.10):
        for det, nm in ((cp_binseg, "binseg"), (cp_sigmoid_fit, "sigmoid")):
            p = power(0.0, 5, ss, detector=det, n_trials=400)
            print(f"  sigma_seed={ss:.2f} n=5  {nm:8s}  false-positive rate = {p:.3f}")
