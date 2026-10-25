"""Correlation of equally weighted fully hedged economy returns with excess currency returns (Chapter 4).

One heatmap per universe size, 14 economies against the 13 non-USD currencies, plus a combined figure.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from common import universe_mask
from portfolio_backtest import config
from portfolio_backtest.utils.io import build_asset_to_currency, load_csv
from portfolio_backtest.utils.returns import (compute_daily_forward_premium, compute_excess_currency_returns,
                                              compute_fully_hedged_returns)

ECONOMIES = ["AUD", "CAD", "DKK", "EUR", "HKD", "JPY", "NZD", "NOK", "SGD", "ZAR", "SEK", "CHF", "GBP", "USD"]
CURRENCIES = ECONOMIES[:-1]
OUT_DIR = config.RESULTS_DIR / "figures"


def correlation(hedged: pd.DataFrame, excess_fx: pd.DataFrame, atc: dict[str, str], mask: pd.DataFrame):
    cols = [c for c in hedged.columns if c in atc]
    masked = hedged[cols].where(mask.reindex(index=hedged.index, columns=cols) == 1)
    economy = masked.T.groupby(pd.Series({c: atc[c] for c in cols})).mean().T
    corr = economy.add_prefix("EQ_").join(excess_fx.add_prefix("FX_"), how="outer").corr(method="pearson")
    corr = corr.reindex(index=[f"EQ_{e}" for e in ECONOMIES], columns=[f"FX_{c}" for c in CURRENCIES])
    return corr.set_axis(pd.Index(ECONOMIES, name="economy")).set_axis(pd.Index(CURRENCIES, name="currency"), axis=1)


def heatmap(ax, corr: pd.DataFrame, norm, vabs: float):
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="lightgrey")
    im = ax.imshow(np.ma.masked_invalid(corr.values), cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index, fontsize=8)
    ax.set_ylabel("Equally Weighted Economy Fully Hedged Return")
    ax.set_xlabel("Excess Currency Return")
    for (i, j), val in np.ndenumerate(corr.values):
        if not np.isnan(val):
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(val) > 0.5 * vabs else "black")
    return im


def symmetric_norm(*matrices: pd.DataFrame) -> tuple[TwoSlopeNorm, float]:
    vabs = max((np.nanmax(np.abs(m.values)) for m in matrices if np.isfinite(m.values).any()), default=0.0)
    if not np.isfinite(vabs) or vabs == 0.0:
        vabs = 0.1
    return TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs), vabs


def save(fig, stem: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
        print(f"Saved figure: {OUT_DIR / f'{stem}.{ext}'}")
    plt.close(fig)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    local = load_csv(config.PROCESSED_DIR / config.LOCAL_RETURNS_FILE)
    local = local.where(local.abs() < config.RETURN_FILTER_BACKTEST)
    spot = load_csv(config.PROCESSED_DIR / config.SPOT_RETURNS_FILE)
    premium = compute_daily_forward_premium(load_csv(config.PROCESSED_DIR / config.FWD_PREMIUMS_3M_FILE),
                                            config.TENOR_DAYS_3M)
    excess_fx = compute_excess_currency_returns(spot, premium).drop(columns=["USD"], errors="ignore")
    hedged = compute_fully_hedged_returns(
        local, spot, premium, build_asset_to_currency(config.RAW_DIR / config.UNIVERSE_FILE_100))

    corrs = {}
    for n, name in config.UNIVERSE_FILES.items():
        path = config.RAW_DIR / name
        corrs[n] = correlation(hedged, excess_fx, build_asset_to_currency(path), universe_mask(path, hedged.index))
        corrs[n].to_csv(OUT_DIR / f"asset_currency_corr_top{n}.csv")
        norm, vabs = symmetric_norm(corrs[n])
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.colorbar(heatmap(ax, corrs[n], norm, vabs), ax=ax, fraction=0.046, pad=0.04, label="Pearson correlation")
        fig.tight_layout()
        save(fig, f"asset_currency_corr_top{n}")

    norm, vabs = symmetric_norm(*corrs.values())
    fig, axes = plt.subplots(len(corrs), 1, figsize=(10, 22))
    for ax, (n, corr) in zip(axes, sorted(corrs.items())):
        im = heatmap(ax, corr, norm, vabs)
        ax.set_title(f"top-{n}")
        ax.set(xlabel="", ylabel="")
    fig.supxlabel("Excess Currency Return", y=0.075)
    fig.supylabel("Equally Weighted Economy Fully Hedged Return")
    fig.suptitle("Asset-Currency Correlation Across Increasing Universe Sizes", y=0.905)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="Pearson correlation")
    save(fig, "asset_currency_corr_combined")
