"""Torch-free core: centered design + Lasso coordinate descent (Theorem 1 estimator)."""
import numpy as np


def centered_design(Z):
    return 2.0 * (Z - 0.5)  # {0,1} -> {-1,+1}


def lasso_cd(X, y, lam, n_iter=500, tol=1e-7):
    N, d = X.shape
    y_mean = y.mean()
    yc = y - y_mean
    beta = np.zeros(d)
    col_sq = (X ** 2).sum(axis=0) + 1e-12
    r = yc.copy()
    for _ in range(n_iter):
        max_delta = 0.0
        for j in range(d):
            xj = X[:, j]
            rho_j = xj @ (r + beta[j] * xj)
            old = beta[j]
            z = rho_j / col_sq[j]
            thr = lam / (col_sq[j] / N)
            new = np.sign(z) * max(abs(z) - thr / 2.0, 0.0)
            if new != old:
                r += (old - new) * xj
                beta[j] = new
                max_delta = max(max_delta, abs(new - old))
        if max_delta < tol:
            break
    return beta, y_mean