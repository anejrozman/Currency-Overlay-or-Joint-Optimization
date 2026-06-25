"""
Walk-forward backtester implementing Ulrych & Lucescu (2026), Section 4 and Appendix B.

Author: Anej Rozman
"""

from __future__ import annotations

import logging
import math
import pandas as pd
import numpy as np
from itertools import product
from typing import Optional, Type

from portfolio_backtest.config import ASSET_COST_BPS, FX_COST_BPS
from portfolio_backtest.optimizers.base_optimizer import BaseOptimizer

logger = logging.getLogger(__name__)


class Backtester:

    def __init__(
        self,
        optimizer: Type[BaseOptimizer],
        param_grid: Optional[dict[str, list]] = None,
        window_years: int = 10,
        val_years: int = 1,
        rebalancing_frequency: str = "quarterly",
        asset_return_key: str = "R_fh",
        logging: bool = False,
    ) -> None:
        """
        Args:
            optimizer: Optimizer class to instantiate at each step.
            param_grid: Grid of hyperparameters for model selection.  Each key
                        maps to a list of candidate values.  Pass ``None`` or
                        ``{}`` to skip grid search (use optimizer defaults).
            window_years: Total rolling lookback window length in years used for model
                          estimation, validation and selection at each rebalancing date.
            val_years: Validation window length in years.  Must be less than ``window_years``. 
            rebalancing_frequency: One of ``"daily"``, ``"weekly"``,
                                   ``"monthly"``, ``"quarterly"``,
                                   ``"semi-annually"`` or ``"annually"``.
            asset_return_key: Key used to look up the asset return DataFrame
                              inside ``data_bundle``.  Use ``"R_fh"`` for
                              fully-hedged returns  or ``"R_u"`` for unhedged returns.
            logging: Emit progress messages via the logging module. 
        """
        self.optimizer = optimizer
        self.param_grid = param_grid if param_grid else {}
        self.window_years = window_years
        self.val_years = val_years
        self.rebalancing_frequency = rebalancing_frequency
        self.asset_return_key = asset_return_key
        self._logging = logging


    # Helper functions

    def _get_rebalance_dates(self, dates: pd.DatetimeIndex) -> list[pd.Timestamp]:
        """
        Build the rebalancing dates from a trading day index.
        For non daily frequencies, the rebalance date is the last available
        trading day in each calendar period.

        Args:
            dates: Trading day index of the return dataframe.

        Returns:
            List of rebalancing timestamps.
        """
        if dates is None or len(dates) == 0:
            return []

        trading_dates = pd.DatetimeIndex(dates).sort_values().unique()
        freq = self.rebalancing_frequency.strip().lower()

        if freq == "daily":
            return list(trading_dates)
        if freq == "weekly":
            period_key = trading_dates.to_period("W-FRI")
        elif freq == "monthly":
            period_key = trading_dates.to_period("M")
        elif freq == "quarterly":
            period_key = trading_dates.to_period("Q")
        elif freq == "semi-annually":
            period_key = pd.Index(
                [(d.year, 1 if d.month <= 6 else 2) for d in trading_dates],
                dtype="object",
            )
        elif freq == "annually":
            period_key = trading_dates.to_period("Y")
        else:
            raise ValueError(
                "Unsupported rebalancing_frequency. "
            )

        helper = pd.DataFrame({"date": trading_dates, "period": period_key})
        return (
            helper.groupby("period", sort=False)["date"]
            .max()
            .sort_values()
            .tolist()
        )

    def _get_estimation_and_validation_splits(
        self,
        dates: pd.DatetimeIndex,
        rebalancing_date: pd.Timestamp,
        next_rebalancing_date: pd.Timestamp,
    ) -> dict[str, pd.Timestamp]:
        """
        Compute period boundaries for one rebalancing cycle.

        Args:
            dates: Trading day index of the return dataframe.
            rebalancing_date: Current rebalancing date *t*.
            next_rebalancing_date: Next rebalancing date *t_next*.

        Returns:
            Dictionary with keys:
            ``train_start``, ``train_end``,
            ``validation_start``, ``validation_end``,
            ``full_estimation_start``, ``full_estimation_end``,
            ``holding_start``, ``holding_end``.

        Raises:
            ValueError: If either date is absent from *dates*, dates are
                        out of order, or the holding window falls outside
                        the available index.
        """
        trading_dates = pd.DatetimeIndex(dates).sort_values().unique()
        t = pd.Timestamp(rebalancing_date)
        t_next = pd.Timestamp(next_rebalancing_date)
        estimation_anchor = t - pd.DateOffset(years=self.window_years)
        validation_anchor = t - pd.DateOffset(years=self.val_years)

        def first_on_or_after(ts: pd.Timestamp) -> pd.Timestamp:
            idx = trading_dates.searchsorted(ts, side="left")
            if idx >= len(trading_dates):
                raise ValueError("Cannot align start boundary: anchor is after available dates.")
            return pd.Timestamp(trading_dates[idx])

        def last_on_or_before(ts: pd.Timestamp) -> pd.Timestamp:
            idx = trading_dates.searchsorted(ts, side="right") - 1
            if idx < 0:
                raise ValueError("Cannot align end boundary: anchor is before available dates.")
            return pd.Timestamp(trading_dates[idx])

        train_start = first_on_or_after(estimation_anchor)
        train_end = last_on_or_before(validation_anchor)
        train_end_idx = trading_dates.get_loc(train_end)
        validation_start = pd.Timestamp(trading_dates[train_end_idx + 1])
        validation_end = t

        full_estimation_start = train_start
        full_estimation_end = t

        # Execution lag assumptions
        # - Weights computed at close of t
        # - Orders executed at close of t+1
        # - OOS returns start at t+2 and end at t_next+1
        t_idx = trading_dates.get_loc(t)
        t_next_idx = trading_dates.get_loc(t_next)

        if isinstance(t_idx, slice) or isinstance(t_next_idx, slice):
            raise ValueError("Duplicate rebalancing dates in trading index.")

        holding_start_idx = int(t_idx) + 2
        holding_end_idx = int(t_next_idx) + 1

        if holding_start_idx >= len(trading_dates):
            raise ValueError("Cannot compute holding start (t+2): not enough future dates.")
        if holding_end_idx >= len(trading_dates):
            raise ValueError("Cannot compute holding end (t_next+1): not enough future dates.")

        return {
            "train_start": train_start,
            "train_end": train_end,
            "validation_start": validation_start,
            "validation_end": validation_end,
            "full_estimation_start": full_estimation_start,
            "full_estimation_end": full_estimation_end,
            "holding_start": pd.Timestamp(trading_dates[holding_start_idx]),
            "holding_end": pd.Timestamp(trading_dates[holding_end_idx]),
        }

    def _filter_universe(self, data_slice: pd.DataFrame) -> list[str]:
        """
        Args:
            data_slice: Sliced return DataFrame for the estimation window.

        Returns:
            List of column names (asset RICs) with a complete price history.

        Raises:
            ValueError: If no assets have a complete history.
        """
        valid_assets = data_slice.columns[data_slice.notna().all()].tolist()
        if not valid_assets:
            raise ValueError("No assets have a complete price history for this window.")
        return valid_assets

    def _generate_grid(self) -> list[dict]:
        """
        Generate all hyperparameter combinations from ``self.param_grid``.

        Returns:
            List of dictionaries. Returns ``[{}]`` if no grid is defined.
        """
        if not self.param_grid:
            return [{}]
        keys, values = zip(*self.param_grid.items())
        return [dict(zip(keys, v)) for v in product(*values)]

    def _compute_asset_transaction_costs(
        self,
        drifted_weights: pd.Series,
        new_weights: pd.Series,
        asset_cost_bps: float = ASSET_COST_BPS,
    ) -> float:
        """
        Calculate transaction costs from rebalancing individual asset holdings.

        Uses the log-linear cost model from Appendix C of Ulrych & Lucescu (2026):
        ``tau = log((1 + chi) / (1 - chi))`` where ``chi = cost_bps / 10_000``.

        Args:
            drifted_weights: Drifted asset weights carried over from last period.
            new_weights: Target asset weights after rebalancing.
            asset_cost_bps: Cost in basis points per unit traded. ``ASSET_COST_BPS`` from ``config.py``.

        Returns:
            Total asset transaction cost as a fraction of portfolio value.
        """
        chi = asset_cost_bps / 10_000.0
        tau = math.log((1 + chi) / (1 - chi))
        return float((tau * (new_weights - drifted_weights).abs()).sum())

    def _compute_currency_transaction_costs(
        self,
        drifted_forward_weights: pd.Series,
        new_forward_weights: pd.Series,
        drifted_currency_weights: pd.Series,
        new_currency_weights: pd.Series,
        full_forward_turnover: bool,
        base_currency: str = "USD",
        fx_cost_bps: float = FX_COST_BPS,
    ) -> float:
        """
        Calculate transaction costs from currency spot conversions and forward hedges.

        Args:
            drifted_forward_weights: Drifted forward hedge weights from last period.
            new_forward_weights: Target forward hedge weights after rebalancing.
            drifted_currency_weights: Drifted implied currency weights.
            new_currency_weights: New implied currency weights.
            full_forward_turnover: If ``True``, assume contracts expire and the
                                   full new position is traded (no daily rebalancing).
                                   If ``False``, only incremental adjustments are traded.
            base_currency: Base currency (no transaction costs for spot conversions in this currency)
            fx_cost_bps: Cost in basis points per unit traded. ``FX_COST_BPS`` from ``config.py``.

        Returns:
            Total currency transaction cost as a fraction of portfolio value.
        """
        chi = fx_cost_bps / 10_000.0
        tau = math.log((1 + chi) / (1 - chi))

        def _exclude_base(s: pd.Series) -> pd.Series:
            s_out = s.copy()
            if base_currency in s_out.index:
                s_out[base_currency] = 0.0
            return s_out

        new_fwds = _exclude_base(new_forward_weights)
        drifted_fwds = _exclude_base(drifted_forward_weights)
        new_spot = _exclude_base(new_currency_weights)
        drifted_spot = _exclude_base(drifted_currency_weights)

        if full_forward_turnover:
            fwd_volume = new_fwds.abs()
        else:
            fwd_volume = (new_fwds - drifted_fwds).abs()

        fwd_costs = (tau * fwd_volume).sum()
        spot_costs = (tau * (new_spot - drifted_spot).abs()).sum()
        return float(fwd_costs + spot_costs)

    def compute_performance_metrics(
        self,
        oos_returns: pd.Series,
        rf_series: pd.Series,
        label: str = "Strategy",
        trading_days: int = 252,
    ) -> dict:
        """
        Calculate and print the backtest performace metrics.

        * **ES_80** — Daily Expected Shortfall at 80 % confidence level
        * **ES_95** — Daily Expected Shortfall at 95 % confidence level
        * **VOL**   — Annualized volatility
        * **MD**    — Maximum drawdown
        * **SR**    — Sharpe Ratio = annualized excess mean / annualized volatility
        * **SOR**   — Sortino Ratio = annualized excess mean / downside deviation
        * **ESR**   — Expected Shortfall Ratio = annualized excess mean / annualized ES_80

        Args:
            oos_returns: Daily out-of-sample portfolio returns.
            label: Strategy name printed in the output table.
            trading_days: Number of trading days per year for annualisation.
            rf_series: Daily risk-free rate seriesaligned to *oos_returns*.

        Returns:
            Dictionary of metrics.
        """
        r = oos_returns.dropna()
        rf_daily = rf_series.reindex(r.index).ffill()
        r_excess = r - rf_daily

        # Annualized excess mean 
        cagr = (1.0 + r).prod() ** (trading_days / len(r)) - 1.0
        rf_ann = float(rf_daily.mean() if hasattr(rf_daily, "mean") else rf_daily) * trading_days
        ann_excess_mean = cagr - rf_ann

        # Annualized volatility
        ann_vol = r.std(ddof=1) * math.sqrt(trading_days)

        # Daily Expected Shortfall: mean of the worst tail returns.
        # ES_80 averages the worst 20% of days, ES_95 the worst 5%.
        es_threshold_80 = r.quantile(0.20)
        daily_es_80 = -r[r <= es_threshold_80].mean()
        ann_es_80 = daily_es_80 * math.sqrt(trading_days)

        es_threshold_95 = r.quantile(0.05)
        daily_es_95 = -r[r <= es_threshold_95].mean()

        # Maximum Drawdown
        cum = (1.0 + r).cumprod()
        drawdown = (cum - cum.cummax()) / cum.cummax()
        max_drawdown = float(abs(drawdown.min()))

        # Sharpe Ratio
        sr = ann_excess_mean / ann_vol if ann_vol > 0 else float("nan")

        # Sortino Ratio
        downside_sq = (r_excess.clip(upper=0.0) ** 2).mean()
        if downside_sq > 0:
            downside_vol = math.sqrt(downside_sq) * math.sqrt(trading_days)
            sor = ann_excess_mean / downside_vol if downside_vol > 0 else float("nan")
        else:
            sor = float("nan")

        # Expected Shortfall Ratio (based on ES_80)
        esr = ann_excess_mean / ann_es_80 if ann_es_80 > 0 else float("nan")

        asset_to = (
            round(float(np.mean(self._asset_turnovers)), 4)
            if getattr(self, "_asset_turnovers", None)
            else float("nan")
        )
        currency_to = (
            round(float(np.mean(self._currency_turnovers)), 4)
            if getattr(self, "_currency_turnovers", None)
            else float("nan")
        )

        metrics = {
            "ES_80 (daily, %)":              round(daily_es_80 * 100, 4),
            "ES_95 (daily, %)":              round(daily_es_95 * 100, 4),
            "Volatility (ann., %)":          round(ann_vol * 100, 4),
            "Max Drawdown (%)":              round(max_drawdown * 100, 4),
            "Sharpe Ratio":                  round(sr, 4),
            "Sortino Ratio":                 round(sor, 4),
            "ESR":                           round(esr, 4),
            "Avg Asset Turnover (per rebal.)":    asset_to,
            "Avg Currency Turnover (per rebal.)": currency_to,
        }

        # Print metrics in a table at the end of the backtest
        width = max(len(k) for k in metrics) + 2
        print(f"\n{'─' * (width + 12)}")
        print(f"  {label}")
        print(f"{'─' * (width + 12)}")
        for k, v in metrics.items():
            print(f"  {k:<{width}} {v:>8}")
        print(f"{'─' * (width + 12)}\n")

        return metrics

    def run(self, data_bundle: dict) -> pd.Series:
        """
        Execute the backtest.

        Algorithm following Ulrych & Lucescu 2026, Section 4 and Appendix B.

        1. Determine all rebalancing dates from the data's trading-day index.
        2. For each rebalancing date *t* with a full ``window_years`` of history:

           a. Grid-search hyperparameters on the validation window.
           b. Refit the best optimizer on the full estimation window.
           c. Compute asset weights (and currency overlay).
           d. Apply transaction costs at rebalancing.
           e. Accumulate returns over the holding period unitl next rebalance date.

        Args:
            data_bundle: Dictionary with keys:

                * ``asset_return_key`` (pd.DataFrame) daily asset returns
                  (``R_fh`` or ``R_u`` depending on ``self.asset_return_key``).
                * ``"R_c"`` (pd.DataFrame, optional) excess currency returns.
                * ``"asset_to_currency"`` (dict) RIC to currency mapping.
                * ``"universe_mask"`` (pd.DataFrame, optional) universe mask.

        Returns:
            pd.Series of daily out of sample portfolio returns.
        """

        R_fh: pd.DataFrame = data_bundle[self.asset_return_key]
        R_c: Optional[pd.DataFrame] = data_bundle.get("R_c")
        asset_to_currency: dict = data_bundle["asset_to_currency"]
        universe_mask: Optional[pd.DataFrame] = data_bundle.get("universe_mask", None)

        all_assets = R_fh.columns.tolist()
        all_currencies = R_c.columns.tolist() if R_c is not None else []
        n_currencies = len(all_currencies)

        # Build asset-to-currency indicator matrix C
        C = pd.DataFrame(0.0, index=all_assets, columns=all_currencies)
        for asset, ccy in asset_to_currency.items():
            if asset in C.index and ccy in C.columns:
                C.loc[asset, ccy] = 1.0

        full_date_index = R_fh.index
        rebalance_dates = self._get_rebalance_dates(full_date_index)
        full_forward_turnover = self.rebalancing_frequency.strip().lower() != "daily"

        # State carried across periods
        drifted_x = pd.Series(0.0, index=all_assets)
        drifted_w = pd.Series(0.0, index=all_currencies)
        drifted_phi = pd.Series(0.0, index=all_currencies)
        is_first_period = True

        period_returns: list[pd.Series] = []
        self._asset_turnovers: list[float] = []
        self._currency_turnovers: list[float] = []
        data_start = full_date_index[0]
        n_periods = len(rebalance_dates) - 1

        for i in range(n_periods):
            if self._logging:
                done = int(40 * (i + 1) / n_periods)
                logger.info(
                    "Backtest progress: |%s%s| %.1f%% (%d/%d)",
                    "█" * done,
                    "-" * (40 - done),
                    (i + 1) / n_periods * 100,
                    i + 1,
                    n_periods,
                )

            t = rebalance_dates[i]
            t_next = rebalance_dates[i + 1]

            # Require a full window_years of history
            if pd.Timestamp(t) - pd.DateOffset(years=self.window_years) < data_start:
                continue

            try:
                splits = self._get_estimation_and_validation_splits(full_date_index, t, t_next)
            except ValueError:
                continue

            train_start = splits["train_start"]
            train_end = splits["train_end"]
            validation_start = splits["validation_start"]
            validation_end = splits["validation_end"]
            full_est_start = splits["full_estimation_start"]
            full_est_end = splits["full_estimation_end"]
            holding_start = splits["holding_start"]
            holding_end = splits["holding_end"]

            # Filter investable universe. First mask then completeness check
            if universe_mask is not None:
                mask_row = (
                    universe_mask.loc[t]
                    if t in universe_mask.index
                    else pd.Series(dtype=float)
                )
                current_universe = [a for a in mask_row[mask_row.notna()].index if a in R_fh.columns]
                if not current_universe:
                    continue
                full_est_slice = R_fh.loc[full_est_start:full_est_end, current_universe]
            else:
                full_est_slice = R_fh.loc[full_est_start:full_est_end]

            try:
                valid_assets = self._filter_universe(full_est_slice)
            except ValueError:
                continue

            R_c_train = R_c.loc[train_start:train_end] if R_c is not None else pd.DataFrame()
            R_c_val = R_c.loc[validation_start:validation_end] if R_c is not None else pd.DataFrame()
            R_c_full = R_c.loc[full_est_start:full_est_end] if R_c is not None else pd.DataFrame()
            C_sub = C.loc[valid_assets].values 

            # Hyperparameter grid search 
            param_combinations = self._generate_grid()
            best_params = param_combinations[0]
            best_score = float("inf")

            if self.param_grid:
                for params in param_combinations:
                    opt = self.optimizer(hyperparams=params)
                    train_bundle_inner = {
                        self.asset_return_key: R_fh.loc[train_start:train_end, valid_assets],
                        "R_c": R_c_train,
                        "C": C_sub,
                    }
                    opt.fit(train_bundle_inner)
                    candidate_weights = opt.optimize()

                    val_bundle_inner = {
                        self.asset_return_key: R_fh.loc[validation_start:validation_end, valid_assets],
                        "R_c": R_c_val,
                        "C": C_sub,
                    }
                    s = opt.score(candidate_weights, val_bundle_inner)
                    if s < best_score:
                        best_score = s
                        best_params = params

            # Refit on full estimation window with best hyperparameters
            opt_final = self.optimizer(hyperparams=best_params)
            full_train_bundle = {
                self.asset_return_key: R_fh.loc[full_est_start:full_est_end, valid_assets],
                "R_c": R_c_full,
                "C": C_sub,
            }
            opt_final.fit(full_train_bundle)
            raw_weights = opt_final.optimize()

            # Parse optimizer output into asset weights x_t and forward hedge weights psi_t
            if isinstance(raw_weights, dict):
                x_sub = np.asarray(raw_weights["x"], dtype=float)
                psi_sub = np.asarray(raw_weights["psi"], dtype=float)
            else:
                x_sub = np.asarray(raw_weights, dtype=float)
                psi_sub = np.zeros(n_currencies, dtype=float)

            # Align to full universe
            x_t = pd.Series(0.0, index=all_assets)
            x_t[valid_assets] = x_sub
            psi_t = pd.Series(psi_sub, index=all_currencies)

            # Implied currency exposure and forward hedge positions
            w_t = C.T @ x_t          
            phi_t = w_t - psi_t     

            #Transaction costs
            if is_first_period:
                asset_tc = 0.0
                currency_tc = 0.0
                is_first_period = False
            else:
                asset_tc = self._compute_asset_transaction_costs(drifted_x, x_t)
                currency_tc = self._compute_currency_transaction_costs(
                    drifted_forward_weights=drifted_phi,
                    new_forward_weights=phi_t,
                    drifted_currency_weights=drifted_w,
                    new_currency_weights=w_t,
                    full_forward_turnover=full_forward_turnover,
                )
                # Track raw turnover (no cost scaling) for performance metrics
                self._asset_turnovers.append(float((x_t - drifted_x).abs().sum()))
                phi_t_non_base = phi_t.drop("USD", errors="ignore")
                self._currency_turnovers.append(float(phi_t_non_base.abs().sum()))
            tc_total = asset_tc + currency_tc

            # Returns during holding period 
            R_fh_hold = R_fh.loc[holding_start:holding_end, valid_assets].fillna(0.0)
            R_c_hold = (
                R_c.loc[holding_start:holding_end].fillna(0.0)
                if R_c is not None
                else pd.DataFrame(index=R_fh_hold.index, columns=[])
            )

            x_vec = x_t[valid_assets].values     
            psi_vec = psi_t.values         

            asset_cum_growth = (1.0 + R_fh_hold).cumprod(axis=0)
            asset_wealth = asset_cum_growth.to_numpy() @ x_vec  
            currency_cum_pnl = (R_c_hold.to_numpy() @ psi_vec).cumsum() 
            total_wealth = asset_wealth + currency_cum_pnl 

            period_ret = pd.Series(0.0, index=R_fh_hold.index, dtype=float)
            period_ret.iloc[0] = total_wealth[0] - 1.0 - tc_total
            if len(period_ret) > 1:
                period_ret.iloc[1:] = total_wealth[1:] / total_wealth[:-1] - 1.0

            # Update drifted weights for next period
            end_growth = asset_cum_growth.iloc[-1].to_numpy()
            unnorm_x = end_growth * x_vec
            total_asset_value = unnorm_x.sum()
            drifted_x_values = unnorm_x / total_asset_value if total_asset_value > 0 else x_vec

            drifted_x = pd.Series(0.0, index=all_assets)
            drifted_x[valid_assets] = drifted_x_values
            drifted_w = C.T @ drifted_x
            drifted_phi = drifted_w - psi_t  #psi_t does not drift. #TODO: Reset it each period

            period_returns.append(period_ret)

        if not period_returns:
            return pd.Series(dtype=float)

        return pd.concat(period_returns).sort_index()
