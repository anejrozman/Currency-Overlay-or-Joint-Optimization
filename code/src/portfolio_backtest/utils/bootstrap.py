"""Circular block bootstrap of a mean, with the Politis-White block length."""

from __future__ import annotations

import numpy as np

N_BOOT = 10000
SEED = 0


def politis_white(x: np.ndarray) -> int:
    """Optimal circular block length, (2 G^2 / D)^(1/3) n^(1/3) (Politis & White 2004; Patton, Politis &
    White 2009).
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
    """Resampled positions per replication: blocks of consecutive positions from uniform starts, wrapping
    past the end of the series.
    """
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    return ((starts[:, :, None] + np.arange(block)) % n).reshape(n_boot, -1)[:, :n]


def block_bootstrap_mean(
    x: np.ndarray, seed: int = SEED, n_boot: int = N_BOOT
) -> tuple[float, float, float, float, int]:
    """(mean, 2.5th and 97.5th percentiles, standard error, block length) of the bootstrapped mean.

    With fewer than eight observations the interval is NaN.
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
