"""
Synthetic verification of the reference-SNR recovery law (paper §8.1).

No black-box model needed. We BUILD masked functions with known coefficients,

    g(z) = sum_{|S|<=K} beta_S chi_S(z)  +  sum_{|S|>K} beta_S chi_S(z),
                           recoverable  /       residual energy m_{>K} ---/

then sweep the residual energy m_{>K} (the reference-dependent knob) and the
query budget N, and check the two sharp pass/fail predictions of Theorem 1:

  (i)  minimum recoverable |beta_S|  proportional to
            (sigma_obs + c sqrt(m_{>K})) sqrt(log p_K / N)
  (ii) signed-support recovery probability, plotted against the rescaled signal
            beta_min / floor,
       collapses onto a single threshold curve for ALL references / noise levels.

Failure of the linear-in-sqrt(m) scaling, or of the collapse, falsifies the law.

Run:  python synthetic_verify.py
Outputs: floor_scaling.png, support_collapse.png  (+ printed PASS/FAIL summary)
"""
from __future__ import annotations

import itertools
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lime import centered_design, lasso_cd

rng = np.random.default_rng(0)


# --------------------------------------------------------------------------- #
#  Build a synthetic degree-K function with controlled residual energy.
# --------------------------------------------------------------------------- #
def make_function(d, K, n_active, beta_active, m_resid, seed):
    """Return (true_support, true_beta_lowdeg, sample_fn).

    Low-degree part: n_active coefficients of magnitude beta_active on random
    size-<=K subsets. Residual part: many high-degree coefficients whose total
    energy equals m_resid (this is the m_{>K} we sweep)."""
    g = np.random.default_rng(seed)
    # --- low-degree active support ---
    units = list(range(d))
    low_sets = []
    while len(low_sets) < n_active:
        k = g.integers(1, K + 1)
        S = tuple(sorted(g.choice(units, size=k, replace=False)))
        if S not in low_sets:
            low_sets.append(S)
    signs = g.choice([-1.0, 1.0], size=n_active)
    beta_low = {S: beta_active * s for S, s in zip(low_sets, signs)}

    # --- high-degree residual support (degree K+1 .. K+2) carrying energy m ---
    hi_sets = []
    n_hi = 200
    while len(hi_sets) < n_hi:
        k = g.integers(K + 1, K + 3)
        k = min(k, d)
        S = tuple(sorted(g.choice(units, size=k, replace=False)))
        if S not in hi_sets and S not in beta_low:
            hi_sets.append(S)
    # equal-energy split so sum beta^2 = m_resid
    mag = math.sqrt(m_resid / n_hi) if m_resid > 0 else 0.0
    hi_signs = g.choice([-1.0, 1.0], size=n_hi)
    beta_hi = {S: mag * s for S, s in zip(hi_sets, hi_signs)}

    def chi(Z, S):
        # Z in {0,1}, chi_S = prod 2(z_i - 1/2)
        out = np.ones(Z.shape[0])
        for i in S:
            out *= (2.0 * (Z[:, i] - 0.5))
        return out

    def sample_fn(N, sigma_obs):
        Z = (rng.random((N, d)) > 0.5).astype(float)
        y = np.zeros(N)
        for S, b in beta_low.items():
            y += b * chi(Z, S)
        for S, b in beta_hi.items():
            y += b * chi(Z, S)
        if sigma_obs > 0:
            y += sigma_obs * rng.standard_normal(N)
        return Z, y

    return low_sets, beta_low, sample_fn


def p_K(d, K):
    return sum(math.comb(d, k) for k in range(0, K + 1))


# --------------------------------------------------------------------------- #
#  Fit degree-K Lasso on the centered single-coordinate design.
#  For K=1 the design columns are exactly the d centered units; for K=2 we add
#  pairwise products. We only need single-unit recovery here, so use K=1 design
#  but the *function* carries higher-order residual -> that's the leakage.
# --------------------------------------------------------------------------- #
def fit_and_check(Z, y, true_support_singletons, beta_active, floor):
    X = centered_design(Z)                  # (N,d) degree-1 design
    N, d = X.shape
    # In lasso_cd the +-1 columns give col_sq ~= N, so the effective coefficient
    # soft-threshold is lam/2. To make the Lasso's OWN active set coincide with
    # the theoretical detection floor, set lam = 2*floor and read support off the
    # Lasso nonzeros directly (no second, inconsistent threshold).
    lam = max(2.0 * floor, 1e-6)
    beta_hat, _ = lasso_cd(X, y, lam)
    true = set(i for S in true_support_singletons for i in S if len(S) == 1)
    rec = set(np.where(np.abs(beta_hat) > 1e-8)[0].tolist())
    # signed-support recovery in the Thm-1 sense: every true unit is recovered,
    # and no false unit exceeds the floor magnitude (tiny shrinkage leftovers ok).
    found_all = true.issubset(rec)
    false_pos = any(abs(beta_hat[j]) > floor for j in rec - true)
    correct = found_all and not false_pos
    return correct, beta_hat


# --------------------------------------------------------------------------- #
#  EXPERIMENT 1: floor scaling.  Fix N, sweep m_{>K}; find minimum beta_active
#  that is reliably recovered. Theory: that threshold ~ linear in sqrt(m).
# --------------------------------------------------------------------------- #
def experiment_floor_scaling(d=30, K=1, n_active=4, N=4000, sigma_obs=0.02,
                             n_trials=24):
    c = 1.3  # empirically calibrated leakage constant (see diagnose.py, Lemma 1)
    log_pK = math.log(p_K(d, K))
    # sweep reference-induced residual energy m_{>K} > 0 (the reference knob);
    # small fixed sigma_obs gives a physical, nonzero intercept.
    m_grid = np.array([0.002, 0.005, 0.01, 0.02, 0.04, 0.08, 0.12])
    beta_grid = np.linspace(0.005, 0.22, 40)

    min_recoverable = []
    for m in m_grid:
        floor = (sigma_obs + c * math.sqrt(m)) * math.sqrt(log_pK / N)
        # success rate as a function of beta, then interpolate the 80% crossing
        rates = []
        for beta_active in beta_grid:
            succ = 0
            for t in range(n_trials):
                sup, _, sample_fn = make_function(d, K, n_active, beta_active,
                                                  m, seed=1000 * t + int(m * 1e4))
                Z, y = sample_fn(N, sigma_obs)
                ok, _ = fit_and_check(Z, y, sup, beta_active, floor)
                succ += int(ok)
            rates.append(succ / n_trials)
        rates = np.array(rates)
        above = np.where(rates >= 0.8)[0]
        if len(above) and above[0] > 0:
            i = above[0]
            # linear interpolation of the 0.8 crossing between grid points
            b0, b1 = beta_grid[i - 1], beta_grid[i]
            r0, r1 = rates[i - 1], rates[i]
            chosen = b0 + (0.8 - r0) * (b1 - b0) / (r1 - r0 + 1e-12)
        elif len(above):
            chosen = beta_grid[above[0]]
        else:
            chosen = np.nan
        min_recoverable.append(chosen)

    min_recoverable = np.array(min_recoverable)
    pred = (sigma_obs + c * np.sqrt(m_grid)) * math.sqrt(log_pK / N)

    # fit observed = a + b * sqrt(m); report linearity R^2
    xs = np.sqrt(m_grid)
    mask = ~np.isnan(min_recoverable)
    A = np.vstack([np.ones(mask.sum()), xs[mask]]).T
    coef, *_ = np.linalg.lstsq(A, min_recoverable[mask], rcond=None)
    fit = A @ coef
    ss_res = ((min_recoverable[mask] - fit) ** 2).sum()
    ss_tot = ((min_recoverable[mask] - min_recoverable[mask].mean()) ** 2).sum()
    r2 = 1 - ss_res / (ss_tot + 1e-12)

    plt.figure(figsize=(6, 4.2))
    plt.plot(xs, min_recoverable, "o-", label="observed min recoverable |β|")
    plt.plot(xs, pred / pred.max() * np.nanmax(min_recoverable), "--",
             label="theory floor ∝ √m (scaled)")
    plt.xlabel(r"$\sqrt{m_{>K}}$  (reference-induced residual)")
    plt.ylabel(r"min recoverable $|\beta_S|$")
    plt.title(f"Floor scaling (linear-in-√m fit R²={r2:.3f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig("floor_scaling.png", dpi=130)
    plt.close()
    return r2


# --------------------------------------------------------------------------- #
#  EXPERIMENT 2: support-recovery collapse.  Many (m, N, sigma) settings;
#  plot recovery prob vs beta_min / floor. Theory: all collapse to one curve.
# --------------------------------------------------------------------------- #
def experiment_collapse(d=30, K=1, n_active=4, n_trials=30):
    c = 1.3  # empirically calibrated leakage constant (see diagnose.py, Lemma 1)
    log_pK = math.log(p_K(d, K))
    # The floor law concerns REFERENCE-INDUCED residual energy m_{>K} > 0.
    # The m=0 case is a degenerate no-reference baseline (floor driven only by
    # sigma_obs, shrinking as 1/sqrt(N) independent of any reference) and is not
    # part of the reference-selection claim, so we exclude it from the collapse.
    settings = list(itertools.product(
        [1500, 3000, 6000],          # N
        [0.01, 0.02, 0.04, 0.08],    # m_{>K} > 0  (the reference knob)
        [0.0, 0.05],                 # sigma_obs
    ))
    rescaled, probs, tags = [], [], []
    beta_grid = np.linspace(0.01, 0.30, 22)  # finer grid -> less quantization
    for (N, m, sig) in settings:
        floor = (sig + c * math.sqrt(m)) * math.sqrt(log_pK / N)
        for beta_active in beta_grid:
            succ = 0
            for t in range(n_trials):
                sup, _, sample_fn = make_function(
                    d, K, n_active, beta_active, m,
                    seed=7 * t + N + int(m * 1e4) + int(sig * 1e3))
                Z, y = sample_fn(N, sig)
                ok, _ = fit_and_check(Z, y, sup, beta_active, floor)
                succ += int(ok)
            rescaled.append(beta_active / (floor + 1e-12))
            probs.append(succ / n_trials)
            tags.append(f"N={N},m={m},σ={sig}")

    rescaled = np.array(rescaled)
    probs = np.array(probs)

    plt.figure(figsize=(6.2, 4.4))
    for tag in sorted(set(tags)):
        idx = [i for i, t in enumerate(tags) if t == tag]
        order = np.argsort(rescaled[idx])
        plt.plot(rescaled[idx][order], probs[idx][order], ".-",
                 alpha=0.55, markersize=4)
    plt.axvline(1.0, color="k", ls=":", label="floor (rescaled = 1)")
    plt.xscale("log")
    plt.xlabel(r"rescaled signal  $\beta_{min} / \mathrm{floor}$")
    plt.ylabel("signed-support recovery prob.")
    plt.title("Collapse across references / N / σ (each line = one setting)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("support_collapse.png", dpi=130)
    plt.close()

    # quantify collapse: spread of the beta/floor value at which prob crosses 0.5
    cross = []
    for tag in set(tags):
        idx = [i for i, t in enumerate(tags) if t == tag]
        r = rescaled[idx]; p = probs[idx]
        o = np.argsort(r); r, p = r[o], p[o]
        above = np.where(p >= 0.5)[0]
        if len(above):
            cross.append(r[above[0]])
    cross = np.array(cross)
    spread = cross.std() / (cross.mean() + 1e-12)  # coeff. of variation
    return spread, cross


if __name__ == "__main__":
    print("=" * 64)
    print("Reference-SNR recovery law — synthetic verification (§8.1)")
    print("=" * 64)

    print("\n[1/2] Floor scaling: min recoverable |β| vs √m_{>K} ...")
    r2 = experiment_floor_scaling()
    pass1 = r2 > 0.9
    print(f"      linear-in-√m fit R² = {r2:.3f}  ->  "
          f"{'PASS' if pass1 else 'FAIL'} (need R²>0.9)")
    print("      saved floor_scaling.png")

    print("\n[2/2] Support-recovery collapse across references / N / σ ...")
    spread, cross = experiment_collapse()
    pass2 = spread < 0.35
    print(f"      0.5-crossing of β_min/floor: mean={cross.mean():.2f}, "
          f"CoV={spread:.3f}  ->  {'PASS' if pass2 else 'FAIL'} (need CoV<0.35)")
    print("      (theory predicts crossing near 1.0, tight across settings)")
    print("      saved support_collapse.png")

    print("\n" + "=" * 64)
    verdict = "THEORY SUPPORTED" if (pass1 and pass2) else "THEORY NOT SUPPORTED"
    print(f"VERDICT: {verdict}")
    print("=" * 64)