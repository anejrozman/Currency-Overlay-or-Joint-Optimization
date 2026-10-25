"""Mean and covariance estimators used by the optimizers."""

from __future__ import annotations

import numpy as np


def ledoit_wolf_cc(returns: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf (2004) covariance, shrunk toward the constant-correlation target (their covCor routine).

    Zero-variance columns, such as the base currency, get zero rows and columns.
    """
    t, n = returns.shape
    x = returns - returns.mean(axis=0)
    sample = (x.T @ x) / t  # MLE sample covariance (1 / T)

    if n < 2:
        return sample

    var = np.diag(sample)

    # zero-variance columns carry no correlation; shrink the informative block only
    nz = var > 0.0
    if not nz.all():
        out = np.zeros((n, n))
        idx = np.where(nz)[0]
        if idx.size:
            out[np.ix_(idx, idx)] = ledoit_wolf_cc(returns[:, idx])
        return out

    sqrtvar = np.sqrt(var)
    outer_sqrt = np.outer(sqrtvar, sqrtvar)

    # constant-correlation target F
    rho_bar = (np.sum(sample / outer_sqrt) - n) / (n * (n - 1))
    prior = rho_bar * outer_sqrt
    np.fill_diagonal(prior, var)

    # pi-hat: asymptotic variance of the sample covariance entries
    y = x ** 2
    phi_mat = (y.T @ y) / t - 2.0 * (x.T @ x) * sample / t + sample ** 2
    phi = phi_mat.sum()

    # rho-hat: asymptotic covariance of the target and the sample covariance
    term1 = ((x ** 3).T @ x) / t
    help_ = (x.T @ x) / t
    term2 = np.diag(help_)[:, None] * sample
    term3 = help_ * var[:, None]
    term4 = var[:, None] * sample
    theta_mat = term1 - term2 - term3 + term4
    np.fill_diagonal(theta_mat, 0.0)
    rho = np.diag(phi_mat).sum() + rho_bar * np.sum(
        np.outer(1.0 / sqrtvar, sqrtvar) * theta_mat
    )

    # gamma-hat: Frobenius distance of target and sample covariance
    gamma = np.linalg.norm(sample - prior, "fro") ** 2
    if gamma == 0.0:
        return sample

    shrinkage = max(0.0, min(1.0, ((phi - rho) / gamma) / t))
    return shrinkage * prior + (1.0 - shrinkage) * sample


def sample_cross_cov(Y1: np.ndarray, Y2: np.ndarray) -> np.ndarray:
    """Sample cross-covariance (ddof = 1) of two return matrices over the same observations."""
    t = Y1.shape[0]
    x1 = Y1 - Y1.mean(axis=0)
    x2 = Y2 - Y2.mean(axis=0)
    return (x1.T @ x2) / (t - 1)


def james_stein_mean(Y: np.ndarray) -> tuple[np.ndarray, float]:
    """James-Stein mean shrunk toward zero, (1 - w) * mean, and the weight w.

    w = min(1, (d - 2) * average variance / (q * ||mean||^2)); with d <= 2 or a zero mean the estimate is zero.
    """
    mu_hat = Y.mean(axis=0)
    q, d = Y.shape

    ss = float(mu_hat @ mu_hat)
    if d <= 2 or ss == 0.0:
        return np.zeros_like(mu_hat), 1.0

    sigma_bar_sq = float(Y.var(axis=0, ddof=1).mean())
    w = min(1.0, (d - 2) * sigma_bar_sq / (q * ss))
    return (1.0 - w) * mu_hat, w


def pooled_asset_mean(
    Y_fh: np.ndarray, economy_ids: np.ndarray
) -> tuple[np.ndarray, float]:
    """Asset means pooled by economy: each economy's equally weighted return series is James-Stein shrunk
    and every asset takes its economy's mean. Returns the means and the shrinkage weight.
    """
    economy_ids = np.asarray(economy_ids, dtype=int)
    if economy_ids.shape[0] != Y_fh.shape[1]:
        raise ValueError(
            f"economy_ids has length {economy_ids.shape[0]}, "
            f"expected {Y_fh.shape[1]} (one per asset)."
        )

    present = np.unique(economy_ids)
    panel = np.column_stack(
        [Y_fh[:, economy_ids == e].mean(axis=1) for e in present]
    )

    mu_economy, w = james_stein_mean(panel)

    return mu_economy[np.searchsorted(present, economy_ids)], w

def economy_ids_from_indicator(C: object, n_assets: int) -> np.ndarray:
    """Economy index of each asset from the asset-to-currency indicator matrix C, in R_c's column order."""
    if C is None:
        raise ValueError("bundle must contain 'C' (asset-to-currency matrix).")
    C_arr = np.asarray(C, dtype=float)
    if C_arr.ndim != 2 or C_arr.shape[0] != n_assets:
        raise ValueError(f"'C' has shape {C_arr.shape}, expected ({n_assets}, K).")
    if not np.allclose(C_arr.sum(axis=1), 1.0):
        bad = int((~np.isclose(C_arr.sum(axis=1), 1.0)).sum())
        raise ValueError(
            f"Every row of 'C' must assign exactly one economy; {bad} row(s) do not."
        )
    return C_arr.argmax(axis=1)


def conditional_currency_mean(F: object, Y_c: np.ndarray) -> np.ndarray:
    """Expected excess currency returns from the carry, mu_c = -f_t, using the last forward premium of the window.

    Under a random-walk spot rate E_t[e_{t+1} - f_t] = -f_t. The base currency stays at zero.
    """
    if F is None or getattr(F, "empty", False):
        raise ValueError(
            "bundle must contain a non-empty 'F' (daily forward premiums) for the "
            "conditional currency mean; check that the runner passes it."
        )
    f_last = np.asarray(getattr(F, "values", F), dtype=float)[-1]
    mu = -np.nan_to_num(f_last, nan=0.0)

    base = np.where(Y_c.var(axis=0) <= 0.0)[0]
    if base.size:
        mu[base] = 0.0
    return mu
