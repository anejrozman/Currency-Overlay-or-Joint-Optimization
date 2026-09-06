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


def james_stein_mean(Y: np.ndarray) -> tuple[np.ndarray, float]:
    """
    James-Stein shrinkage estimator for the mean, shrinking toward zero.

    Unlike :func:`bayes_stein`, which shrinks toward the minimum-variance
    portfolio mean, the target here is the origin: the estimate is
    ``(1 - w) * mu_hat`` with

        w = min(1, (d - 2) * sigma_bar_sq / (q * ||mu_hat||^2)),

    where ``sigma_bar_sq`` is the average per-column sample variance. The
    intensity is data-driven and grows with the noise level (large average
    variance, short window) and with the dimension, and shrinks as the sample
    mean vector moves away from zero. At ``w = 1`` the estimate is the zero
    vector, i.e. no return prediction at all.

    Shrinking toward zero rather than a grand mean is deliberate: a grand mean
    is itself an estimated quantity that inherits the noise the shrinkage is
    meant to suppress, whereas zero is the null hypothesis that the assets carry
    no predictable drift over the window. The estimator degenerates when the
    classical dominance condition cannot hold (``d <= 2``) or when the shrinkage
    factor is undefined (``||mu_hat|| = 0``); both return ``w = 1``, i.e. the
    zero vector.

    Args:
        Y: (q, d) array of returns.

    Returns:
        Two-tuple of the shrunk mean vector, shape *(d,)*, and the shrinkage
        weight ``w`` in *[0, 1]*.
    """
    mu_hat = Y.mean(axis=0)
    q, d = Y.shape

    # James-Stein dominates the sample mean only from d = 3 on, and the factor
    # below is undefined at mu_hat = 0. Both cases collapse to the target.
    ss = float(mu_hat @ mu_hat)
    if d <= 2 or ss == 0.0:
        return np.zeros_like(mu_hat), 1.0

    sigma_bar_sq = float(Y.var(axis=0, ddof=1).mean())
    w = min(1.0, (d - 2) * sigma_bar_sq / (q * ss))
    return (1.0 - w) * mu_hat, w


def pooled_asset_mean(
    Y_fh: np.ndarray, economy_ids: np.ndarray
) -> tuple[np.ndarray, float]:
    """
    Economy-pooled, James-Stein shrunk asset means.

    Estimating N individual asset means from a single window is hopeless at the
    sample sizes here (N ~ 100-130 against q ~ 2500 daily observations, with a
    signal-to-noise ratio per asset far below one). Pooling collapses the
    estimation problem onto the 14 economies: each economy's return series is
    the equal-weighted average of its constituents, that q x d panel is passed
    through :func:`james_stein_mean`, and every asset inherits its own economy's
    shrunk mean. The dimensionality of the mean estimate drops from N to at most
    14 while the number of observations is unchanged, and the cross-sectional
    spread of expected returns is restricted to the between-economy dimension.

    Economies with no asset in the current estimation window are dropped rather
    than carried as empty columns, so ``d`` is the number of economies actually
    represented, which is at most 14 and can be smaller for a window whose
    point-in-time universe does not span every economy.

    Args:
        Y_fh: (q, N) array of asset returns.
        economy_ids: Length-N integer array mapping each asset to its economy.

    Returns:
        Two-tuple of the per-asset mean vector, shape *(N,)*, in which all
        assets of one economy share a value, and the shrinkage weight ``w``
        applied at the economy level.
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

    # Expand back to asset level: position of each asset's economy in `present`.
    return mu_economy[np.searchsorted(present, economy_ids)], w

def economy_ids_from_indicator(C: object, n_assets: int) -> np.ndarray:
    """
    Map each asset to its economy from an asset-to-currency indicator matrix.

    The Backtester assembles this matrix (bundle key ``"C"``) and restricts it to
    the window's valid assets; its columns are R_c's columns in order, so economy
    k and currency k are the same object and no separate mapping is needed.

    Args:
        C: (N, K) asset-to-currency indicator matrix from the data bundle.
        n_assets: Expected number of assets, for validation.

    Returns:
        Length-N integer array of economy indices, in R_c's column order.

    Raises:
        ValueError: If ``C`` is missing, mis-shaped, or has a row that is not a
                    single indicator -- an asset with no economy, or more than
                    one, would otherwise be silently misattributed.
    """
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
    """
    Conditional currency mean from the forward premium: mu_c = -f_t.

    Under the random-walk hypothesis for spot (E_t[e_{t+1}] = 0), the definition
    R_c = e_{t+1} - f_t gives E_t[R_c] = -f_t directly, so the expected excess
    currency return is the negative forward premium -- the carry.  Nothing is
    estimated: f_t is observed at the end of the window, which is why this
    replaces the James-Stein shrunk sample mean the module used previously.  That
    sample mean was shrunk to exactly zero in most windows, since 13 currency
    means over a 2500-day window carry almost no signal, leaving step 2 with no
    mean term at all.

    The pooled OLS of R_c on f over the full sample gives a slope of -0.83,
    against -1 under random-walk spot and 0 under uncovered interest parity, so
    the -1 loading used here is close to what the data supports and far from the
    UIP null.  The remaining discrepancy does not matter: a constant rescaling of
    mu_c is absorbed exactly by gamma, since the objective is
    Risk - mu' w / gamma.  Only a time-varying or per-currency loading would add
    content beyond this, at the cost of parameters estimated from one window.

    The forward premium frame is already shifted by one day when it is built
    (:func:`~portfolio_backtest.utils.returns.compute_daily_forward_premium`), so
    the last row of the window is the premium known one day before the window
    ends.  That is stale by a day rather than forward-looking, which is the safe
    direction.

    The base currency is held at zero: its R_c column is identically zero by
    construction and its forward premium is likewise zero, so it carries no carry.

    Args:
        F: (q, K) frame of daily forward premiums for the window, aligned to R_c.
        Y_c: (q, K) array of excess currency returns, used only to locate the
             base-currency column.

    Returns:
        Length-K conditional mean vector.

    Raises:
        ValueError: If ``F`` is missing or empty -- this optimizer conditions on
                    it and must not silently fall back to a zero mean.
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
