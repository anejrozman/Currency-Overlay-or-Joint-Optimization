"""

Description: Backtester implementation based on "Regularized Multi-Currency Expected Shortfall Portfolios" by Ulrych and Lucescu (2026).  

Author: Anej Rozman

"""

import pandas as pd
import os
import sys
import math
from itertools import product
from typing import Type, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.optimizers.BaseOptimizer import BaseOptimizer

class Backtester:

    def __init__(self, 
                 optimizer: Type[BaseOptimizer], 
                 param_grid: Optional[dict] = None,         # Grid of hyperparameters to search over during model selection. Depends on optimizer type. 
                 window_years: int = 10,                    # Total length of rolling lookcback window used for model estimaton, validation and selection at each rebalancing date.
                 val_years: int = 1,                        # Length of validation window used for model selection at each rebalancing date. Must be less than window_years.
                 rebalancing_frequency: str = "quarterly",  # Options: daily, weekly, monthly, quarterly, semi-annually, annually
                 asset_return_key: str = 'R_fh',            # Data bundle key for the asset return series used in estimation and realization.
                 logging: bool = False
                 ):

            self.optimizer = optimizer
            self.param_grid = param_grid if param_grid else {}
            self.window_years = window_years
            self.val_years = val_years
            self.rebalancing_frequency = rebalancing_frequency
            self.asset_return_key = asset_return_key
            self.logging = logging
    
    def _get_rebalance_dates(self, dates: pd.DatetimeIndex) -> list[pd.Timestamp]:
        """
        Build rebalancing dates from a trading-day index.

        For non-daily frequencies, the rebalance date is the last available
        trading day in each period.
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
                dtype="object"
            )
        elif freq == "annually":
            period_key = trading_dates.to_period("Y")
        else:
            raise ValueError(
                "Unsupported rebalancing_frequency. "
                "Use one of: daily, weekly, monthly, quarterly, semi-annually, annually."
            )

        helper = pd.DataFrame({"date": trading_dates, "period": period_key})
        rebalance_dates = (
            helper.groupby("period", sort=False)["date"]
            .max()
            .sort_values()
            .tolist()
        )
        return rebalance_dates
    
    def _get_estimation_and_validation_splits(self, 
                                              dates: pd.DatetimeIndex,              # Full index of trading dates.
                                              rebalancing_date: pd.Timestamp,       # Current rebalancing date for which we want to compute the splits.
                                              next_rebalancing_date: pd.Timestamp   # Next rebalancing date, needed to compute the holding period end boundary with execution lag.
                                              ) -> dict[str, pd.Timestamp]:
        """
        Compute period boundaries for one rebalancing cycle.

        Returns a dictionary with:
        - train_start, train_end
        - validation_start, validation_end
        - full_estimation_start, full_estimation_end
        - holding_start, holding_end
        """

        trading_dates = pd.DatetimeIndex(dates).sort_values().unique()
        t = pd.Timestamp(rebalancing_date)
        t_next = pd.Timestamp(next_rebalancing_date)

        if t not in trading_dates:
            raise ValueError("'rebalancing_date' must be present in 'dates'.")
        if t_next not in trading_dates:
            raise ValueError("'next_rebalancing_date' must be present in 'dates'.")
        if t_next <= t:
            raise ValueError("'next_rebalancing_date' must be strictly after 'rebalancing_date'.")

        # Calendar anchors based on year offsets.
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

        # Estimation/validation boundaries.
        train_start = first_on_or_after(estimation_anchor)
        train_end = last_on_or_before(validation_anchor)
        train_end_idx = trading_dates.get_loc(train_end)
        validation_start = pd.Timestamp(trading_dates[train_end_idx + 1])
        validation_end = t

        if train_start > train_end:
            raise ValueError("Invalid training window: start is after end.")
        if validation_start > validation_end:
            raise ValueError("Invalid validation window: start is after end.")

        # Full estimation window for final fit after model selection.
        full_estimation_start = train_start
        full_estimation_end = t

        # Execution lag:
        # - Weights computed at close of t
        # - Orders executed at close of t+1
        # - OOS returns start at t+2 and end at t_next+1
        t_idx = trading_dates.get_loc(t)
        t_next_idx = trading_dates.get_loc(t_next)

        if isinstance(t_idx, slice) or isinstance(t_next_idx, slice):
            raise ValueError("Duplicate rebalancing dates detected in trading index.")

        holding_start_idx = int(t_idx) + 2
        holding_end_idx = int(t_next_idx) + 1

        if holding_start_idx >= len(trading_dates):
            raise ValueError("Cannot compute holding start (t+2): insufficient future dates.")
        if holding_end_idx >= len(trading_dates):
            raise ValueError("Cannot compute holding end (t_next+1): insufficient future dates.")
        if holding_start_idx > holding_end_idx:
            raise ValueError("Invalid holding window: start is after end.")

        holding_start = pd.Timestamp(trading_dates[holding_start_idx])
        holding_end = pd.Timestamp(trading_dates[holding_end_idx])

        return {
            "train_start": train_start,
            "train_end": train_end,
            "validation_start": validation_start,
            "validation_end": validation_end,
            "full_estimation_start": full_estimation_start,
            "full_estimation_end": full_estimation_end,
            "holding_start": holding_start,
            "holding_end": holding_end,
        }
    
    def _filter_universe(self, data_slice: pd.DataFrame) -> list[str]:
        """
        Takes a sliced dataframe of returns.
        Returns a list of column names that have no missing values 
        in this specific time window.
        """
        valid_assets = data_slice.columns[data_slice.notna().all()].tolist()
        if len(valid_assets) == 0:
            raise ValueError("No assets have a complete price history for this window.")
        return valid_assets

    
    def _generate_grid(self) -> list:
        """
        Generates a list of hyperparameter combinations from the provided parameter grid.
        """
        if not self.param_grid: 
             return [{}]
        keys, values = zip(*self.param_grid.items())
        return [dict(zip(keys, v)) for v in product(*values)]


    def _compute_asset_transaction_costs(self, 
                                         drifted_weights: pd.Series, 
                                         new_weights: pd.Series, 
                                         asset_cost_bps: float = 10.0   # Default to 10 bps per unit traded, assumption in Appendix C of the thesis.
                                         ) -> float:
        """
        Calculate the transaction costs incurred from rebalancing individual asset holdings.
        """
        chi = asset_cost_bps / 10000.0
        tau = math.log((1 + chi) / (1 - chi))
        trade_volume = (new_weights - drifted_weights).abs()
        
        return float((tau * trade_volume).sum())

    def _compute_currency_transaction_costs(self,
                                            drifted_forward_weights: pd.Series,
                                            new_forward_weights: pd.Series,
                                            drifted_currency_weights: pd.Series,
                                            new_currency_weights: pd.Series,
                                            full_forward_turnover: bool,
                                            base_currency: str = "USD",
                                            fx_cost_bps: float = 1.0        # Default to 1 bp per unit traded, assumption in Appendix C of the thesis.
                                            ) -> float:
        """
        Calculate the transaction costs incurred from currency spot conversions and forward hedges.
        """

        chi = fx_cost_bps / 10000.0
        tau = math.log((1 + chi) / (1 - chi))
        
        # Helper to zero out the base currency costs
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
            # Old contracts expire, new contracts are fully traded
            fwd_volume = new_fwds.abs()
        else:
            # Contracts are kept open, only incremental adjustments are traded
            fwd_volume = (new_fwds - drifted_fwds).abs()
            
        fwd_costs = (tau * fwd_volume).sum()
        
        # Exchanges required to fund purchases of assets denominated in different currencies
        spot_volume = (new_spot - drifted_spot).abs()
        spot_costs = (tau * spot_volume).sum()
        
        return float(fwd_costs + spot_costs)

    def compute_performance_metrics(self,
                                    oos_returns: pd.Series,
                                    label: str = "Strategy",
                                    trading_days: int = 252,
                                    rf_annual: float = 0.0,
                                    rf_series: Optional[pd.Series] = None
                                    ) -> dict:
        """
        Compute and print the performance metrics reported in Ulrych & Lucescu (2026):
          - Daily Expected Shortfall at the 80% confidence level (ES_80)
          - Annualized Volatility (VOL)
          - Maximum Drawdown (MD)
          - Sharpe Ratio (SR)    = annualized excess mean / annualized volatility
          - Sortino Ratio (SOR)  = annualized excess mean / annualized downside deviation
          - ESR                  = annualized excess mean / annualized ES_80

        SR, SOR and ESR are computed on excess returns (r - rf_daily) to match the
        paper's convention. ES, Vol and MD use gross returns.
        rf_series: optional pd.Series of daily RF rates in decimal form aligned to oos_returns.
                   If provided, takes precedence over rf_annual.
        rf_annual is the annualized risk-free rate (default 0.0 for backward compatibility).

        ES and volatility are annualized by multiplying the daily figures by sqrt(trading_days).
        Mean is annualized by multiplying by trading_days.
        Sortino uses the standard T-denominator (all observations, negatives contribute r^2).
        """
        r = oos_returns.dropna()
        if rf_series is not None:
            rf_daily = rf_series.reindex(r.index).fillna(rf_annual / trading_days)
        else:
            rf_daily = rf_annual / trading_days
        r_excess = r - rf_daily

        # Annualized excess mean via CAGR (numerator for SR, SOR, ESR):
        # gross CAGR minus annualised RF avoids arithmetic-mean upward bias
        cagr = (1.0 + r).prod() ** (trading_days / len(r)) - 1.0
        rf_ann = float(rf_daily.mean() if hasattr(rf_daily, 'mean') else rf_daily) * trading_days
        ann_excess_mean = cagr - rf_ann

        # Annualized volatility (gross returns, for Vol and SR denominator)
        ann_vol = r.std(ddof=1) * math.sqrt(trading_days)

        # Daily ES at 80% confidence — average of worst 20% of gross daily returns
        es_threshold = r.quantile(0.20)
        daily_es = -r[r <= es_threshold].mean()
        ann_es   = daily_es * math.sqrt(trading_days)

        # Maximum Drawdown (gross returns)
        cum = (1.0 + r).cumprod()
        running_peak = cum.cummax()
        drawdown = (cum - running_peak) / running_peak
        max_drawdown = float(abs(drawdown.min()))

        # Sharpe Ratio: excess mean / vol
        sr = ann_excess_mean / ann_vol if ann_vol > 0 else float("nan")

        # Sortino Ratio — standard T-denominator: all observations, negative excess contribute r²
        downside_sq = (r_excess.clip(upper=0.0) ** 2).mean()
        if downside_sq > 0:
            downside_vol = math.sqrt(downside_sq) * math.sqrt(trading_days)
            sor = ann_excess_mean / downside_vol if downside_vol > 0 else float("nan")
        else:
            sor = float("nan")

        # Expected Shortfall Ratio: excess mean / annualized ES
        esr = ann_excess_mean / ann_es if ann_es > 0 else float("nan")

        metrics = {
            "ES_80 (daily, %)":      round(daily_es * 100, 4),
            "Volatility (ann., %)":  round(ann_vol  * 100, 4),
            "Max Drawdown (%)":      round(max_drawdown * 100, 4),
            "Sharpe Ratio":          round(sr,  4),
            "Sortino Ratio":         round(sor, 4),
            "ESR":                   round(esr, 4),
        }

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
        Walk-forward backtest following Ulrych & Lucescu (2026), Section 4 / Appendix B.

        data_bundle keys:
            'R_fh'             : pd.DataFrame  — fully hedged asset returns (daily index)
            'R_c'              : pd.DataFrame  — excess currency returns (daily index)
            'asset_to_currency': dict          — RIC -> currency string mapping

        Returns:
            pd.Series of daily out-of-sample portfolio returns indexed by date.
        """
        import numpy as np

        R_fh: pd.DataFrame = data_bundle[self.asset_return_key]
        R_c: Optional[pd.DataFrame] = data_bundle.get('R_c')
        asset_to_currency: dict = data_bundle['asset_to_currency']
        universe_mask: Optional[pd.DataFrame] = data_bundle.get('universe_mask', None)

        all_assets = R_fh.columns.tolist()
        all_currencies = R_c.columns.tolist() if R_c is not None else []
        n_currencies = len(all_currencies)

        # Build asset-to-currency indicator matrix C (N_assets x N_currencies)
        C = pd.DataFrame(0.0, index=all_assets, columns=all_currencies)
        for asset, ccy in asset_to_currency.items():
            if asset in C.index and ccy in C.columns:
                C.loc[asset, ccy] = 1.0

        full_date_index = R_fh.index
        rebalance_dates = self._get_rebalance_dates(full_date_index)

        if len(rebalance_dates) < 2:
            raise ValueError("Need at least two rebalancing dates to run the backtest.")

        full_forward_turnover = self.rebalancing_frequency.strip().lower() != "daily"

        # State carried across periods
        drifted_x   = pd.Series(0.0, index=all_assets)
        drifted_w   = pd.Series(0.0, index=all_currencies)
        drifted_phi = pd.Series(0.0, index=all_currencies)
        is_first_period = True

        period_returns: list[pd.Series] = []

        data_start = full_date_index[0]

        n_periods = len(rebalance_dates) - 1
        for i in range(n_periods):
            if self.logging:
                sys.stdout.write(f'\rBacktest Progress: |{"█" * int(40 * (i + 1) / n_periods) + "-" * (40 - int(40 * (i + 1) / n_periods))}| {(i + 1) / n_periods:.1%} ({i + 1}/{n_periods})')
                sys.stdout.flush()

            t      = rebalance_dates[i]
            t_next = rebalance_dates[i + 1]

            # Require a full window_years of history before using this rebalancing date.
            # If the 10-year anchor falls before the data start, the estimation window
            # is shorter than required and the period is skipped.
            estimation_anchor = pd.Timestamp(t) - pd.DateOffset(years=self.window_years)
            if estimation_anchor < data_start:
                continue

            # Skip periods for which t_next+1 would fall outside the date index
            try:
                splits = self._get_estimation_and_validation_splits(full_date_index, t, t_next)
            except ValueError:
                continue

            train_start       = splits["train_start"]
            train_end         = splits["train_end"]
            validation_start  = splits["validation_start"]
            validation_end    = splits["validation_end"]
            full_est_start    = splits["full_estimation_start"]
            full_est_end      = splits["full_estimation_end"]
            holding_start     = splits["holding_start"]
            holding_end       = splits["holding_end"]

            # Filter investable universe (no NaN in full estimation window).
            # If a point-in-time universe mask is provided, first restrict to the
            # assets that are in the universe AS OF the current rebalancing date,
            # then check completeness in the (unmasked) estimation window.
            # This mirrors the reference code: current top-50 + full price history.
            if universe_mask is not None:
                mask_row = universe_mask.loc[t] if t in universe_mask.index else pd.Series(dtype=float)
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

            R_c_train = R_c.loc[train_start:train_end]           if R_c is not None else pd.DataFrame()
            R_c_val   = R_c.loc[validation_start:validation_end] if R_c is not None else pd.DataFrame()
            R_c_full  = R_c.loc[full_est_start:full_est_end]     if R_c is not None else pd.DataFrame()
            C_sub     = C.loc[valid_assets].values  # shape (n_valid, n_currencies)

            # --- Hyperparameter grid search ---
            param_combinations = self._generate_grid()
            best_params = param_combinations[0]
            best_score  = float("inf")

            if self.param_grid:
                for params in param_combinations:
                    opt = self.optimizer(hyperparams=params)
                    train_bundle_inner = {
                        self.asset_return_key: R_fh.loc[train_start:train_end, valid_assets],
                        "R_c":  R_c_train,
                        "C":    C_sub,
                    }
                    opt.fit(train_bundle_inner)
                    candidate_weights = opt.optimize()

                    val_bundle_inner = {
                        self.asset_return_key: R_fh.loc[validation_start:validation_end, valid_assets],
                        "R_c":  R_c_val,
                        "C":    C_sub,
                    }
                    s = opt.score(candidate_weights, val_bundle_inner)
                    if s < best_score:
                        best_score  = s
                        best_params = params

            # --- Refit on full estimation window with best hyperparams ---
            opt_final = self.optimizer(hyperparams=best_params)
            full_train_bundle = {
                self.asset_return_key: R_fh.loc[full_est_start:full_est_end, valid_assets],
                "R_c":  R_c_full,
                "C":    C_sub,
            }
            opt_final.fit(full_train_bundle)
            raw_weights = opt_final.optimize()

            # --- Parse optimizer output ---
            if isinstance(raw_weights, dict):
                x_sub   = np.asarray(raw_weights["x"],   dtype=float)   # asset weights (n_valid,)
                psi_sub = np.asarray(raw_weights["psi"], dtype=float)    # currency overlay (n_currencies,)
            else:
                x_sub   = np.asarray(raw_weights, dtype=float)
                psi_sub = np.zeros(n_currencies, dtype=float)

            # Align to full universe
            x_t   = pd.Series(0.0, index=all_assets)
            x_t[valid_assets] = x_sub
            psi_t = pd.Series(psi_sub, index=all_currencies)

            # Implied currency exposure and forward hedge positions
            w_t   = C.T @ x_t          # (n_currencies,)
            phi_t = w_t - psi_t        # (n_currencies,) — total forward position

            # --- Transaction costs ---
            if is_first_period:
                asset_tc    = 0.0
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
            tc_total = asset_tc + currency_tc

            # --- Holding-period returns (wealth-index approach for true buy-and-hold) ---
            R_fh_hold = R_fh.loc[holding_start:holding_end, valid_assets].fillna(0.0)
            R_c_hold  = (R_c.loc[holding_start:holding_end].fillna(0.0)
                         if R_c is not None
                         else pd.DataFrame(index=R_fh_hold.index, columns=[]))

            x_vec = x_t[valid_assets].values          # (n_valid,) target asset weights
            psi_vec = psi_t.values                    # (n_currencies,) currency overlay

            # Cumulative asset portfolio value (starts at 1.0)
            asset_cum_growth = (1.0 + R_fh_hold).cumprod(axis=0)
            asset_wealth = asset_cum_growth.to_numpy() @ x_vec   # (T,)

            # Cumulative zero-investment currency overlay P&L
            currency_cum_pnl = (R_c_hold.to_numpy() @ psi_vec).cumsum()  # (T,)

            total_wealth = asset_wealth + currency_cum_pnl       # (T,)

            period_ret = pd.Series(0.0, index=R_fh_hold.index, dtype=float)
            period_ret.iloc[0] = total_wealth[0] - 1.0 - tc_total
            if len(period_ret) > 1:
                period_ret.iloc[1:] = total_wealth[1:] / total_wealth[:-1] - 1.0

            # --- Update drifted weights for next period ---
            # Use the true end-of-period cumulative growth to get the actual drifted weights
            end_growth = asset_cum_growth.iloc[-1].to_numpy()    # cumulative factor per asset
            unnorm_x   = end_growth * x_vec
            total_asset_value = unnorm_x.sum()
            drifted_x_values  = unnorm_x / total_asset_value if total_asset_value > 0 else x_vec

            drifted_x = pd.Series(0.0, index=all_assets)
            drifted_x[valid_assets] = drifted_x_values

            drifted_w   = C.T @ drifted_x
            drifted_phi = drifted_w - psi_t   # psi_t does not drift; it is reset each period

            period_returns.append(period_ret)

        if self.logging:
            print()

        if not period_returns:
            return pd.Series(dtype=float)

        return pd.concat(period_returns).sort_index()
