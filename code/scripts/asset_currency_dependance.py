"""
Pearson cross-correlations between fully hedged stock returns and excess currency returns,
for the top 10/50/100 universes.

Economies are currency blocks, so the combinations are 14 economies x 13 currencies. 
USD is dropped from the currency axis since its excess return is zero.
Returns are recomputed from the processed data with the
same functions the backtester uses. 

Author: Anej Rozman
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from portfolio_backtest import config
from portfolio_backtest.utils.io import build_asset_to_currency, load_csv
from portfolio_backtest.utils.returns import (
    compute_daily_forward_premium,
    compute_excess_currency_returns,
    compute_fully_hedged_returns,
)

ECONOMY_ORDER = ["AUD", "CAD", "DKK", "EUR", "HKD", "JPY", "NZD", "NOK", "SGD", "ZAR", "SEK", "CHF", "GBP", "USD"]
CURRENCY_ORDER = [c for c in ECONOMY_ORDER if c != "USD"]
UNIVERSE_SIZES = (10, 50, 100)


def build_universe_mask(universe_file: Path | str, index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Daily universe mask (1.0 where a stock is a constituent, NaN
    otherwise), forward-filled from the annual reconstitution dates.
    """
    universe_df = pd.read_csv(universe_file)
    universe_df["clean_ric"] = (
        universe_df["ric_code"]
        .astype(str)
        .str.replace("<", "", regex=False)
        .str.replace(">", "", regex=False)
    )
    universe_df["date"] = pd.to_datetime(universe_df["date"])

    constituents = universe_df.groupby("date")["clean_ric"].agg(set)
    all_rics = sorted(universe_df["clean_ric"].unique())
    mask = pd.DataFrame(np.nan, index=constituents.index, columns=all_rics)
    for recon_date, rics in constituents.items():
        mask.loc[recon_date, list(rics)] = 1.0

    return mask.reindex(index, method="ffill")


def aggregate_economy_equity(
    R_fh: pd.DataFrame,
    asset_to_currency: dict[str, str],
    mask: pd.DataFrame,
) -> pd.DataFrame:
    """
    Put per-stock fully hedged returns into one equal-weighted daily series
    per economy, using the universe mask.
    """
    cols = [c for c in R_fh.columns if c in asset_to_currency]
    mask_aligned = mask.reindex(index=R_fh.index, columns=cols)
    masked = R_fh[cols].where(mask_aligned == 1)
    currency_of = pd.Series({c: asset_to_currency[c] for c in cols})
    economy_equity = masked.T.groupby(currency_of).mean().T
    return economy_equity


def cross_correlation(
    economy_equity: pd.DataFrame,
    excess_currency_returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Pearson cross-correlation of every economy return series against every currency series
    """
    eq = economy_equity.add_prefix("EQ_")
    fx = excess_currency_returns.add_prefix("FX_")
    combined = eq.join(fx, how="outer")

    full_corr = combined.corr(method="pearson")

    corr = full_corr.reindex(
        index=[f"EQ_{e}" for e in ECONOMY_ORDER],
        columns=[f"FX_{c}" for c in CURRENCY_ORDER],
    )
    corr.index = ECONOMY_ORDER
    corr.columns = CURRENCY_ORDER
    corr.index.name = "economy"
    corr.columns.name = "currency"
    return corr


def _annotate(ax, corr: pd.DataFrame, vabs: float) -> None:
    values = corr.values
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            if np.isnan(val):
                continue
            color = "white" if abs(val) > 0.5 * vabs else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=7)


def _draw_heatmap(ax, corr: pd.DataFrame, norm, vabs: float, *, show_ylabels: bool):
    masked = np.ma.masked_invalid(corr.values)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="lightgrey")
    im = ax.imshow(masked, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr.index)))
    if show_ylabels:
        ax.set_yticklabels(corr.index, fontsize=8)
        ax.set_ylabel("Equally Weighted Economy Fully Hedged Return")
    else:
        ax.set_yticklabels([])
    ax.set_xlabel("Excess Currency Return")
    _annotate(ax, corr, vabs)
    return im


def _symmetric_norm(*matrices: pd.DataFrame) -> tuple[TwoSlopeNorm, float]:
    vabs = max(
        (np.nanmax(np.abs(m.values)) for m in matrices if np.isfinite(m.values).any()),
        default=0.0,
    )
    if not np.isfinite(vabs) or vabs == 0.0:
        vabs = 0.1
    return TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs), vabs


def _save(fig: plt.Figure, stem: Path) -> None:
    for ext in ("pdf", "png"):
        path = stem.with_suffix(f".{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved figure: {path}")
    plt.close(fig)


def plot_heatmap(corr: pd.DataFrame, title: str, stem: Path) -> None:
    norm, vabs = _symmetric_norm(corr)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = _draw_heatmap(ax, corr, norm, vabs, show_ylabels=True)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson correlation")
    fig.tight_layout()
    _save(fig, stem)


def plot_combined(corrs: dict[int, pd.DataFrame], stem: Path) -> None:
    norm, vabs = _symmetric_norm(*corrs.values())
    fig, axes = plt.subplots(len(corrs), 1, figsize=(10, 22))
    im = None
    for ax, (n, corr) in zip(axes, sorted(corrs.items())):
        im = _draw_heatmap(ax, corr, norm, vabs, show_ylabels=True)
        ax.set_title(f"top-{n}")
        # Drop the per-axis labels; the figure carries a single shared label each.
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.supxlabel("Excess Currency Return", y=0.075)
    fig.supylabel("Equally Weighted Economy Fully Hedged Return")
    fig.suptitle(
        "Asset-Currency Correlation Across Increasing Universe Sizes", y=0.905
    )
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="Pearson correlation")
    _save(fig, stem)


def main() -> None:
    figures_dir = config.RESULTS_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    R_local = load_csv(config.PROCESSED_DIR / config.LOCAL_RETURNS_FILE)
    e = load_csv(config.PROCESSED_DIR / config.SPOT_RETURNS_FILE)
    f_raw = load_csv(config.PROCESSED_DIR / config.FWD_PREMIUMS_3M_FILE)
    R_local = R_local.where(R_local.abs() < config.RETURN_FILTER_BACKTEST)

    # Compute fully hedged returns and excess currency returns
    atc_full = build_asset_to_currency(config.RAW_DIR / config.UNIVERSE_FILE_100)
    f_daily = compute_daily_forward_premium(f_raw, config.TENOR_DAYS_3M)
    R_c = compute_excess_currency_returns(e, f_daily).drop(columns=["USD"], errors="ignore")
    R_fh_full = compute_fully_hedged_returns(R_local, e, f_daily, atc_full)

    # Compute correlations for each universe size
    corrs: dict[int, pd.DataFrame] = {}
    for n in UNIVERSE_SIZES:
        universe_file = config.RAW_DIR / config.UNIVERSE_FILES[n]
        atc_n = build_asset_to_currency(universe_file)
        mask_n = build_universe_mask(universe_file, R_fh_full.index)
        economy_equity = aggregate_economy_equity(R_fh_full, atc_n, mask_n)
        corr = cross_correlation(economy_equity, R_c)
        corrs[n] = corr

        csv_path = figures_dir / f"asset_currency_corr_top{n}.csv"
        corr.to_csv(csv_path)
        plot_heatmap(
            corr,
            f"Equally weighted economy fully hedged return vs. currency excess return (top-{n})",
            figures_dir / f"asset_currency_corr_top{n}",
        )

    plot_combined(corrs, figures_dir / "asset_currency_corr_combined")

if __name__ == "__main__":
    main()
