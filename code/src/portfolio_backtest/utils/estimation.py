"""
Statistical estimation utilities for portfolio optimization.
"""

from __future__ import annotations

import numpy as np


def bayes_stein(returns: np.ndarray) -> np.ndarray:
    """
    Bayes-Stein shrinkage estimator for the mean.

    Shrinks each asset's sample mean toward the minimum-variance portfolio mean
    with a data-driven shrinkage intensity. The precision matrix uses the
    Ledoit-Wolf covariance :func:`ledoit_wolf_cc` rather than the raw sample
    covariance, so the shrinkage target and intensity stay well-defined when the
    number of assets is large relative to the window — including ``N >= T``, where
    the sample covariance is singular and the raw mean would otherwise be returned
    unshrunk.

    Zero-variance columns (e.g. a base-currency excess return that is identically
    zero) carry no covariance information: their raw mean is returned unchanged and
    the remaining columns are shrunk on their own sub-block.

    Args:
        returns: (T, N) array of asset returns.

    Returns:
        1-D array of shape *(N,)* with shrunk mean estimates.
    """
    T, N = returns.shape
    mu = returns.mean(axis=0)

    # Shrink only the informative (non-degenerate) columns; keep raw means for any
    # zero-variance columns so the precision matrix below stays non-singular.
    nz = returns.var(axis=0) > 0.0
    if not nz.all():
        out = mu.copy()
        idx = np.where(nz)[0]
        if idx.size:
            out[idx] = bayes_stein(returns[:, idx])
        return out

    try:
        Sigma = ledoit_wolf_cc(returns)
        Sigma_inv = np.linalg.inv(Sigma)
    except np.linalg.LinAlgError:
        return mu

    ones = np.ones(N)
    denom = float(ones @ Sigma_inv @ ones)
    mu_g = float(ones @ Sigma_inv @ mu) / denom

    diff = mu - mu_g * ones
    delta = float(T * (diff @ Sigma_inv @ diff))
    phi = (N + 2) / ((N + 2) + delta)

    return (1.0 - phi) * mu + phi * mu_g * ones


def ledoit_wolf_cc(returns: np.ndarray) -> np.ndarray:
    """
    Ledoit-Wolf (2004) linear shrinkage covariance estimator toward the
    constant-correlation target.

    Returns the convex combination ``Sigma_LW = v * F + (1 - v) * S``, where
    ``S`` is the sample covariance, ``F`` is the constant-correlation target
    (``F_ii = S_ii``, ``F_ij = rho_bar * sqrt(S_ii S_jj)``), and ``v`` is the
    closed-form shrinkage intensity of Ledoit & Wolf (2004). Implements the
    canonical ``covCor`` routine.

    Args:
        returns: (T, N) array of asset returns.

    Returns:
        (N, N) shrunk covariance matrix.
    """
    t, n = returns.shape
    x = returns - returns.mean(axis=0)
    sample = (x.T @ x) / t  # MLE sample covariance (1 / T)

    # A single asset (or no cross terms) has no correlation to shrink.
    if n < 2:
        return sample

    var = np.diag(sample)

    # Zero-variance columns (e.g. the base-currency excess return, identically
    # zero) carry no correlation. Shrink only the informative sub-block and leave
    # their rows/columns zero, which is their true (zero) variance and covariance.
    nz = var > 0.0
    if not nz.all():
        out = np.zeros((n, n))
        idx = np.where(nz)[0]
        if idx.size:
            out[np.ix_(idx, idx)] = ledoit_wolf_cc(returns[:, idx])
        return out

    sqrtvar = np.sqrt(var)
    outer_sqrt = np.outer(sqrtvar, sqrtvar)

    # Constant-correlation prior F.
    rho_bar = (np.sum(sample / outer_sqrt) - n) / (n * (n - 1))
    prior = rho_bar * outer_sqrt
    np.fill_diagonal(prior, var)

    # pi-hat: asymptotic variance of the sample covariance entries.
    y = x ** 2
    phi_mat = (y.T @ y) / t - 2.0 * (x.T @ x) * sample / t + sample ** 2
    phi = phi_mat.sum()

    # rho-hat: asymptotic covariance of the target with the sample covariance.
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

    # gamma-hat: Frobenius distance between target and sample covariance.
    gamma = np.linalg.norm(sample - prior, "fro") ** 2
    if gamma == 0.0:
        return sample

    shrinkage = max(0.0, min(1.0, ((phi - rho) / gamma) / t))
    return shrinkage * prior + (1.0 - shrinkage) * sample


def sample_cross_cov(Y1: np.ndarray, Y2: np.ndarray) -> np.ndarray:
    """
    Sample cross-covariance block between two return matrices.

    Args:
        Y1: (T, N) array of returns.
        Y2: (T, M) array of returns, sharing the same T observations as ``Y1``.

    Returns:
        (N, M) sample cross-covariance matrix (ddof = 1, not shrunk).
    """
    t = Y1.shape[0]
    x1 = Y1 - Y1.mean(axis=0)
    x2 = Y2 - Y2.mean(axis=0)
    return (x1.T @ x2) / (t - 1)
