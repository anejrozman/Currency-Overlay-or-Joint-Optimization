"""
Circular block bootstrap with an automatic block length.

Both diagnostic scripts form one value per quarterly rebalancing date and need a
confidence interval for its mean.  Those series are serially dependent — ten-year
estimation windows overlap by a single quarter — so an iid bootstrap understates
the spread.  The block length is chosen per series by the Politis-White rule
rather than by hand, and returned alongside the interval so it can be reported.
"""

from __future__ import annotations

import numpy as np

N_BOOT = 10000
SEED = 0


def politis_white(x: np.ndarray) -> int:
    """
    Optimal circular-block length, b* = (2 G^2 / D)^(1/3) n^(1/3).

    G and D are formed from the sample autocorrelations under a flat-top lag
    window; the bandwidth is twice the smallest lag beyond which the
    autocorrelations stay inside the white-noise band for several consecutive
    lags.  Politis & White (2004), with the correction in Patton, Politis &
    White (2009).
    """
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = x.size
    if n < 8 or np.allclose(x, 0.0):
        return 1
    k_n = max(5, int(np.sqrt(np.log10(n))))
    m_max = int(np.ceil(np.sqrt(n))) + k_n
    b_max = np.ceil(min(3.0 * np.sqrt(n), n / 3.0))
    acov = np.array([np.dot(x[: n - k], x[k:]) / n for k in range(m_max + 1)])
    rho = acov / acov[0]
    crit = 2.0 * np.sqrt(np.log10(n) / n)
    m = m_max
    for k in range(1, m_max - k_n + 1):
        if np.all(np.abs(rho[k : k + k_n]) < crit):
            m = k - 1
            break
    M = int(min(max(2 * max(m, 1), 2), m_max))
    lags = np.arange(-M, M + 1)
    r = rho[np.abs(lags)]
    flat_top = np.clip(2.0 * (1.0 - np.abs(lags) / M), 0.0, 1.0)
    G = np.sum(flat_top * np.abs(lags) * r)
    D = (4.0 / 3.0) * np.sum(flat_top * r) ** 2
    if D <= 0 or not np.isfinite(G):
        return 1
    b = (2.0 * G**2 / D) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
    return int(np.clip(round(b), 1, b_max))


def circular_index(n: int, block: int, n_boot: int, rng) -> np.ndarray:
    """
    Row of resampled positions per replication.

    Draws ceil(n / block) uniform start points, takes `block` consecutive
    positions from each while wrapping past the end of the series, and truncates
    the concatenation back to length n.  Wrapping keeps every observation equally
    likely to be drawn, which the non-circular block bootstrap does not.
    """
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    return ((starts[:, :, None] + np.arange(block)) % n).reshape(n_boot, -1)[:, :n]


def block_bootstrap_mean(
    x: np.ndarray, seed: int = SEED, n_boot: int = N_BOOT
) -> tuple[float, float, float, float, int]:
    """
    Mean of *x* with a percentile confidence interval from the circular block
    bootstrap.

    Returns:
        ``(mean, lo, hi, se, block)`` — the sample mean, the 2.5th and 97.5th
        percentiles of the resampled means, their standard deviation, and the
        block length the Politis-White rule selected.  Series shorter than eight
        observations return the mean with NaN interval.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    nan = float("nan")
    if x.size < 8:
        return (float(np.mean(x)) if x.size else nan), nan, nan, nan, 1
    block = politis_white(x)
    rng = np.random.default_rng(seed)
    means = x[circular_index(x.size, block, n_boot, rng)].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(x.mean()), float(lo), float(hi), float(means.std(ddof=1)), block
