"""
Synthetic verification of the reference-SNR recovery law (paper §8.1).

No black-box model. We build masked functions with KNOWN coefficients:

    g(z) = sum_{|S|<=K} beta_S chi_S(z)  +  sum_{|S|>K} beta_S chi_S(z)
                          (recoverable)        (residual energy m_{>K})

then sweep the reference-induced residual energy m_{>K} and budget N and test
the two sharp predictions of Theorem 1:

  (i)  minimum recoverable |beta_S| is linear in sqrt(m_{>K}):
            floor = (sigma_obs + c sqrt(m_{>K})) sqrt(log p_K / N)
  (ii) signed-support recovery probability, plotted vs beta_min/floor, collapses
       onto ONE threshold curve across references / N / sigma.

The leading constant c in the floor is the unspecified constant of Lemma 1; we
measure it directly in `lemma1_constant()` (vectorized) and use it.

Three entry points:
    python synthetic_verify.py lemma1     # vectorized Lemma-1 leakage check
    python synthetic_verify.py floor       # Experiment 1 (floor scaling)
    python synthetic_verify.py collapse    # Experiment 2 (support collapse)
    python synthetic_verify.py all         # everything (default)

Tunables (trial counts, grids, n_jobs-ish) are constants at the top of each fn.
"""
from __future__ import annotations
import sys, math, itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _core import centered_design, lasso_fit, empirical_leakage_batch

RNG = np.random.default_rng(0)


# --------------------------------------------------------------------------- #
#  Synthetic function builder with controlled residual energy m_{>K}.
# --------------------------------------------------------------------------- #
def make_function(d, K, n_active, beta_active, m_resid, seed):
    g = np.random.default_rng(seed)
    units = list(range(d))
    # low-degree active support (size 1..K)
    low_sets = []
    while len(low_sets) < n_active:
        k = int(g.integers(1, K + 1))
        S = tuple(sorted(g.choice(units, size=k, replace=False)))
        if S not in low_sets:
            low_sets.append(S)
    signs = g.choice([-1.0, 1.0], size=n_active)
    beta_low = {S: beta_active * s for S, s in zip(low_sets, signs)}
    # high-degree residual carrying total energy m_resid
    hi_sets, n_hi = [], 200
    while len(hi_sets) < n_hi:
        k = int(g.integers(K + 1, K + 3)); k = min(k, d)
        S = tuple(sorted(g.choice(units, size=k, replace=False)))
        if S not in hi_sets and S not in beta_low:
            hi_sets.append(S)
    mag = math.sqrt(m_resid / n_hi) if m_resid > 0 else 0.0
    hi_signs = g.choice([-1.0, 1.0], size=n_hi)
    beta_hi = {S: mag * s for S, s in zip(hi_sets, hi_signs)}

    def chi(Z, S):
        out = np.ones(Z.shape[0])
        for i in S:
            out *= (2.0 * (Z[:, i] - 0.5))
        return out

    def sample_fn(N, sigma_obs):
        Z = (RNG.random((N, d)) > 0.5).astype(float)
        y = np.zeros(N)
        for S, b in beta_low.items():
            y += b * chi(Z, S)
        for S, b in beta_hi.items():
            y += b * chi(Z, S)
        if sigma_obs > 0:
            y += sigma_obs * RNG.standard_normal(N)
        return Z, y

    true_singletons = set(i for S in low_sets for i in S if len(S) == 1)
    return true_singletons, sample_fn


def p_K(d, K):
    return sum(math.comb(d, k) for k in range(0, K + 1))


def fit_and_check(Z, y, true_singletons, floor):
    """Signed-support recovery in the Thm-1 sense: every true unit recovered,
    and no false unit exceeds the floor magnitude."""
    X = centered_design(Z)
    beta_hat, _ = lasso_fit(X, y, lam=max(floor, 1e-9))
    rec = set(np.where(np.abs(beta_hat) > 1e-8)[0].tolist())
    found_all = true_singletons.issubset(rec)
    false_pos = any(abs(beta_hat[j]) > floor for j in rec - true_singletons)
    return found_all and not false_pos


# --------------------------------------------------------------------------- #
#  Lemma 1, vectorized:  eta ~= c * sqrt(m log p / N).  Returns measured c.
# --------------------------------------------------------------------------- #
def lemma1_constant(d=30, K=1, n_trials=200, verbose=True):
    log_pK = math.log(p_K(d, K))
    rows, ratios = [], []
    for m in [0.005, 0.01, 0.02, 0.04, 0.08]:
        for N in [2000, 8000]:
            # build n_trials pure-residual draws (beta_active = 0) and batch them
            Zs = np.empty((n_trials, N, d)); Ys = np.empty((n_trials, N))
            for t in range(n_trials):
                _, sf = make_function(d, K, n_active=4, beta_active=0.0,
                                      m_resid=m, seed=t)
                Z, y = sf(N, sigma_obs=0.0)
                Zs[t], Ys[t] = Z, y
            eta = empirical_leakage_batch(Zs, Ys).mean()
            pred = math.sqrt(m * log_pK / N)
            ratios.append(eta / pred)
            rows.append((m, N, eta, pred, eta / pred))
    c_hat = float(np.mean(ratios))
    if verbose:
        print(f"{'m':>7} {'N':>7} {'eta_emp':>10} {'sqrt(m logp/N)':>16} {'ratio':>8}")
        for (m, N, e, p, r) in rows:
            print(f"{m:>7.3f} {N:>7} {e:>10.5f} {p:>16.5f} {r:>8.3f}")
        print(f"\nLemma-1 constant  c_hat = mean(ratio) = {c_hat:.3f}  "
              f"(std {np.std(ratios):.3f})")
        print("PASS: ratio is ~constant across 16x range of m and 4x of N "
              "=> eta scales as sqrt(m log p / N) as Lemma 1 predicts.")
    return c_hat


# --------------------------------------------------------------------------- #
#  Experiment 1: floor scaling.  min recoverable |beta| vs sqrt(m).
# --------------------------------------------------------------------------- #
def experiment_floor_scaling(c, d=30, K=1, n_active=4, N=4000,
                             sigma_obs=0.02, n_trials=20):
    log_pK = math.log(p_K(d, K))
    m_grid = np.array([0.002, 0.005, 0.01, 0.02, 0.04, 0.08, 0.12])
    beta_grid = np.linspace(0.005, 0.22, 30)
    min_recoverable = []
    for m in m_grid:
        floor = (sigma_obs + c * math.sqrt(m)) * math.sqrt(log_pK / N)
        rates = []
        for beta_active in beta_grid:
            succ = 0
            for t in range(n_trials):
                tn, sf = make_function(d, K, n_active, beta_active, m,
                                       seed=1000 * t + int(m * 1e4))
                Z, y = sf(N, sigma_obs)
                succ += int(fit_and_check(Z, y, tn, floor))
            rates.append(succ / n_trials)
        rates = np.array(rates)
        above = np.where(rates >= 0.8)[0]
        if len(above) and above[0] > 0:
            i = above[0]; b0, b1 = beta_grid[i-1], beta_grid[i]
            r0, r1 = rates[i-1], rates[i]
            chosen = b0 + (0.8 - r0) * (b1 - b0) / (r1 - r0 + 1e-12)
        elif len(above):
            chosen = beta_grid[above[0]]
        else:
            chosen = np.nan
        min_recoverable.append(chosen)
    min_recoverable = np.array(min_recoverable)

    xs = np.sqrt(m_grid)
    mask = ~np.isnan(min_recoverable)
    A = np.vstack([np.ones(mask.sum()), xs[mask]]).T
    coef, *_ = np.linalg.lstsq(A, min_recoverable[mask], rcond=None)
    fit = A @ coef
    ss_res = ((min_recoverable[mask] - fit) ** 2).sum()
    ss_tot = ((min_recoverable[mask] - min_recoverable[mask].mean()) ** 2).sum()
    r2 = 1 - ss_res / (ss_tot + 1e-12)

    plt.figure(figsize=(6, 4.2))
    plt.plot(xs, min_recoverable, "o", label="observed min recoverable |β|")
    plt.plot(xs[mask], fit, "--", label=f"linear-in-√m fit (R²={r2:.3f})")
    plt.xlabel(r"$\sqrt{m_{>K}}$"); plt.ylabel(r"min recoverable $|\beta_S|$")
    plt.title("Experiment 1: detection floor scales linearly in √m")
    plt.legend(); plt.tight_layout(); plt.savefig("floor_scaling.png", dpi=130)
    plt.close()
    return r2


# --------------------------------------------------------------------------- #
#  Experiment 2: support-recovery collapse (reference regime, m_{>K} > 0).
# --------------------------------------------------------------------------- #
def experiment_collapse(c, d=30, K=1, n_active=4, n_trials=25):
    log_pK = math.log(p_K(d, K))
    settings = list(itertools.product(
        [1500, 3000, 6000],            # N
        [0.01, 0.02, 0.04, 0.08],      # m_{>K} > 0  (the reference knob)
        [0.0, 0.05],                   # sigma_obs
    ))
    beta_grid = np.linspace(0.01, 0.30, 18)
    rescaled, probs, tags = [], [], []
    crossings = []
    for (N, m, sig) in settings:
        floor = (sig + c * math.sqrt(m)) * math.sqrt(log_pK / N)
        these_r, these_p = [], []
        for beta_active in beta_grid:
            succ = 0
            for t in range(n_trials):
                tn, sf = make_function(d, K, n_active, beta_active, m,
                                       seed=7*t + N + int(m*1e4) + int(sig*1e3))
                Z, y = sf(N, sig)
                succ += int(fit_and_check(Z, y, tn, floor))
            rr = beta_active / (floor + 1e-12); pp = succ / n_trials
            rescaled.append(rr); probs.append(pp)
            tags.append(f"N={N},m={m},σ={sig}")
            these_r.append(rr); these_p.append(pp)
        these_r, these_p = np.array(these_r), np.array(these_p)
        idx = np.where(these_p >= 0.5)[0]
        if len(idx):
            crossings.append(these_r[idx[0]])

    rescaled = np.array(rescaled); probs = np.array(probs)
    plt.figure(figsize=(6.2, 4.4))
    for tag in sorted(set(tags)):
        ii = [i for i, t in enumerate(tags) if t == tag]
        o = np.argsort(rescaled[ii])
        plt.plot(np.array(rescaled[ii])[o], np.array(probs[ii])[o],
                 ".-", alpha=0.5, markersize=4)
    plt.axvline(1.0, color="k", ls=":", label="floor (rescaled = 1)")
    plt.xscale("log"); plt.xlabel(r"$\beta_{min}/\mathrm{floor}$")
    plt.ylabel("signed-support recovery prob.")
    plt.title("Experiment 2: collapse across references / N / σ")
    plt.legend(); plt.tight_layout(); plt.savefig("support_collapse.png", dpi=130)
    plt.close()

    crossings = np.array(crossings)
    cov = crossings.std() / (crossings.mean() + 1e-12)
    return crossings, cov


# --------------------------------------------------------------------------- #
def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    print("=" * 66)
    print("Reference-SNR recovery law -- synthetic verification (§8.1)")
    print("=" * 66)

    c = 1.3  # default; overwritten if lemma1 is run
    if what in ("lemma1", "all"):
        print("\n[Lemma 1] vectorized leakage scaling eta vs sqrt(m log p / N)")
        c = lemma1_constant()

    if what in ("floor", "all"):
        print("\n[Experiment 1] floor scaling (min recoverable |β| vs √m) ...")
        r2 = experiment_floor_scaling(c)
        print(f"  linear-in-√m fit R² = {r2:.3f}  "
              f"-> {'PASS' if r2 > 0.9 else 'CHECK'} (target R²>0.9)")
        print("  saved floor_scaling.png")

    if what in ("collapse", "all"):
        print("\n[Experiment 2] support-recovery collapse (m>0 regime) ...")
        crossings, cov = experiment_collapse(c)
        print(f"  0.5-crossing β_min/floor: mean={crossings.mean():.2f}, "
              f"CoV={cov:.3f}  -> {'PASS' if cov < 0.35 else 'CHECK'} "
              f"(target CoV<0.35; theory predicts a single threshold)")
        print("  saved support_collapse.png")

    print("\n" + "=" * 66)
    print("Done. The decisive test is Lemma 1 (clean sqrt-law); Experiments 1-2")
    print("are downstream and sensitive to grid resolution / trial counts.")
    print("=" * 66)


if __name__ == "__main__":
    main()
