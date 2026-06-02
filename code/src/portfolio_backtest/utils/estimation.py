"""
Statistical estimation utilities for portfolio optimization.
"""

from __future__ import annotations

import numpy as np


def bayes_stein(returns: np.ndarray) -> np.ndarray:
    """
    Bayes-Stein shrinkage estimator for the mean.

    Shrinks each assets sample mean toward the minimum variance portfolio
    mean with a shrinkage intensity.

    Args:
        (T, N) array of asset returns.

    Returns:
        1-D array of shape *(N,)* with shrunk mean estimates.
    """
    T, N = returns.shape
    mu = returns.mean(axis=0)
    try:
        Sigma = np.cov(returns, rowvar=False, ddof=1)
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
