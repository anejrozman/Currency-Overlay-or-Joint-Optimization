"""Walk-forward backtester of Ulrych & Lucescu (2026), Section 4 and Appendix B."""

from __future__ import annotations

import logging
import math
from itertools import product

import numpy as np
import pandas as pd

from portfolio_backtest.config import ASSET_COST_BPS, FX_COST_BPS
from portfolio_backtest.optimizers.base_optimizer import BaseOptimizer

logger = logging.getLogger(__name__)

PERIODS = {"weekly": "W-FRI", "monthly": "M", "quarterly": "Q", "annually": "Y"}


def cost_rate(bps: float) -> float:
    """Log-linear cost per unit traded, tau = log((1 + chi) / (1 - chi)) (Appendix C)."""
    chi = bps / 10_000.0
    return math.log((1 + chi) / (1 - chi))


class Backtester:
    """Each rebalancing date: grid search on the validation window, refit on the full estimation window, then
    hold the weights, net of transaction costs, until the next rebalancing date.

    asset_return_key "R_u" marks an unhedged strategy, which holds no forwards. training_return_key, by default
    the asset returns, is the frame used for estimation; ASSETS-ZERO trains on "R_fh" and realizes "R_u".
    """

    def __init__(
        self,
        optimizer: type[BaseOptimizer],
        param_grid: dict[str, list] | None = None,
        window_years: int = 10,
        val_years: int = 1,
        rebalancing_frequency: str = "quarterly",
        asset_return_key: str = "R_fh",
        training_return_key: str | None = None,
        logging: bool = False,
    ) -> None:
        self.optimizer = optimizer
        self.param_grid = param_grid if param_grid else {}
        self.window_years = window_years
        self.val_years = val_years
        self.rebalancing_frequency = rebalancing_frequency
        self.asset_return_key = asset_return_key
        self.training_return_key = training_return_key or asset_return_key
        self._is_unhedged = asset_return_key == "R_u"
        self._logging = logging

    def _get_rebalance_dates(self, dates: pd.DatetimeIndex) -> list[pd.Timestamp]:
        """The last trading day of each period of the rebalancing frequency."""
        if dates is None or len(dates) == 0:
            return []
        trading_dates = pd.DatetimeIndex(dates).sort_values().unique()
        freq = self.rebalancing_frequency.strip().lower()
        if freq == "daily":
            return list(trading_dates)
        if freq == "semi-annually":
            period = pd.Index([(d.year, 1 if d.month <= 6 else 2) for d in trading_dates], dtype="object")
        elif freq in PERIODS:
            period = trading_dates.to_period(PERIODS[freq])
        else:
            raise ValueError("Unsupported rebalancing_frequency. ")
        helper = pd.DataFrame({"date": trading_dates, "period": period})
        return helper.groupby("period", sort=False)["date"].max().sort_values().tolist()

    def _get_estimation_and_validation_splits(
        self, dates: pd.DatetimeIndex, rebalancing_date: pd.Timestamp, next_rebalancing_date: pd.Timestamp
    ) -> dict[str, pd.Timestamp]:
        """Train, validation, full estimation and holding window boundaries of one rebalancing date.

        Weights are computed at the close of t and traded at the close of t+1, so the holding window runs
        from t+2 to t_next+1. Raises ValueError where a boundary falls outside the dates.
        """
        trading_dates = pd.DatetimeIndex(dates).sort_values().unique()
        t = pd.Timestamp(rebalancing_date)
        t_next = pd.Timestamp(next_rebalancing_date)

        start = trading_dates.searchsorted(t - pd.DateOffset(years=self.window_years), side="left")
        if start >= len(trading_dates):
            raise ValueError("Cannot align start boundary: anchor is after available dates.")
        end = trading_dates.searchsorted(t - pd.DateOffset(years=self.val_years), side="right") - 1
        if end < 0:
            raise ValueError("Cannot align end boundary: anchor is before available dates.")
        train_start = pd.Timestamp(trading_dates[start])
        train_end = pd.Timestamp(trading_dates[end])
        validation_start = pd.Timestamp(trading_dates[trading_dates.get_loc(train_end) + 1])

        t_idx, t_next_idx = trading_dates.get_loc(t), trading_dates.get_loc(t_next)
        if isinstance(t_idx, slice) or isinstance(t_next_idx, slice):
            raise ValueError("Duplicate rebalancing dates in trading index.")
        holding_start_idx, holding_end_idx = int(t_idx) + 2, int(t_next_idx) + 1
        if holding_start_idx >= len(trading_dates):
            raise ValueError("Cannot compute holding start (t+2): not enough future dates.")
        if holding_end_idx >= len(trading_dates):
            raise ValueError("Cannot compute holding end (t_next+1): not enough future dates.")

        return {
            "train_start": train_start, "train_end": train_end,
            "validation_start": validation_start, "validation_end": t,
            "full_estimation_start": train_start, "full_estimation_end": t,
            "holding_start": pd.Timestamp(trading_dates[holding_start_idx]),
            "holding_end": pd.Timestamp(trading_dates[holding_end_idx]),
        }

    def _filter_universe(self, data_slice: pd.DataFrame) -> list[str]:
        """Assets with a complete return history in the window."""
        valid_assets = data_slice.columns[data_slice.notna().all()].tolist()
        if not valid_assets:
            raise ValueError("No assets have a complete price history for this window.")
        return valid_assets

    def _generate_grid(self) -> list[dict]:
        if not self.param_grid:
            return [{}]
        keys, values = zip(*self.param_grid.items())
        return [dict(zip(keys, v)) for v in product(*values)]

    def _compute_asset_transaction_costs(
        self, drifted_weights: pd.Series, new_weights: pd.Series, asset_cost_bps: float = ASSET_COST_BPS
    ) -> float:
        return float((cost_rate(asset_cost_bps) * (new_weights - drifted_weights).abs()).sum())

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
        """Cost of the forward hedges and the spot conversions, the base currency excluded.

        With full_forward_turnover the forwards expire each period and the whole new position is traded.
        """
        tau = cost_rate(fx_cost_bps)

        def exclude_base(s: pd.Series) -> pd.Series:
            s = s.copy()
            if base_currency in s.index:
                s[base_currency] = 0.0
            return s

        new_fwds, drifted_fwds = exclude_base(new_forward_weights), exclude_base(drifted_forward_weights)
        new_spot, drifted_spot = exclude_base(new_currency_weights), exclude_base(drifted_currency_weights)
        fwd_volume = new_fwds.abs() if full_forward_turnover else (new_fwds - drifted_fwds).abs()
        return float((tau * fwd_volume).sum() + (tau * (new_spot - drifted_spot).abs()).sum())

    def compute_performance_metrics(
        self, oos_returns: pd.Series, rf_series: pd.Series, label: str = "Strategy", trading_days: int = 252
    ) -> dict:
        """Daily ES_80 and ES_95, annualized volatility and MAD, maximum drawdown, Sharpe, Sortino and ES ratios
        and average turnovers; printed as a table and returned.

        The excess return is the CAGR minus the annualized mean risk-free rate. MAD is taken about the
        out-of-sample mean and annualized with sqrt(trading_days), like the volatility.
        """
        r = oos_returns.dropna()
        rf_daily = rf_series.reindex(r.index).ffill()
        r_excess = r - rf_daily
        cagr = (1.0 + r).prod() ** (trading_days / len(r)) - 1.0
        ann_excess_mean = cagr - float(rf_daily.mean()) * trading_days
        ann_vol = r.std(ddof=1) * math.sqrt(trading_days)

        daily_es_80 = -r[r <= r.quantile(0.20)].mean()
        daily_es_95 = -r[r <= r.quantile(0.05)].mean()
        ann_es_80 = daily_es_80 * math.sqrt(trading_days)
        ann_mad = float((r - r.mean()).abs().mean()) * math.sqrt(trading_days)
        cum = (1.0 + r).cumprod()
        max_drawdown = float(abs(((cum - cum.cummax()) / cum.cummax()).min()))

        sr = ann_excess_mean / ann_vol if ann_vol > 0 else float("nan")
        downside_sq = (r_excess.clip(upper=0.0) ** 2).mean()
        sor = float("nan")
        if downside_sq > 0:
            downside_vol = math.sqrt(downside_sq) * math.sqrt(trading_days)
            sor = ann_excess_mean / downside_vol if downside_vol > 0 else float("nan")
        esr = ann_excess_mean / ann_es_80 if ann_es_80 > 0 else float("nan")

        def average(turnovers):
            return round(float(np.mean(turnovers)), 4) if turnovers else float("nan")

        metrics = {
            "ES_80 (daily, %)": round(daily_es_80 * 100, 4),
            "ES_95 (daily, %)": round(daily_es_95 * 100, 4),
            "Volatility (ann., %)": round(ann_vol * 100, 4),
            "MAD (ann., %)": round(ann_mad * 100, 4),
            "Max Drawdown (%)": round(max_drawdown * 100, 4),
            "Sharpe Ratio": round(sr, 4),
            "Sortino Ratio": round(sor, 4),
            "ESR": round(esr, 4),
            "Avg Asset Turnover (per rebal.)": average(getattr(self, "_asset_turnovers", None)),
            "Avg Currency Turnover (per rebal.)": average(getattr(self, "_currency_turnovers", None)),
        }
        width = max(len(k) for k in metrics) + 2
        rule = "─" * (width + 12)
        print(f"\n{rule}\n  {label}\n{rule}")
        for k, v in metrics.items():
            print(f"  {k:<{width}} {v:>8}")
        print(f"{rule}\n")
        return metrics

    def get_weight_records(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Asset and currency overlay weights as solved at each rebalancing date."""
        return (pd.DataFrame(self._asset_weight_records, columns=["date", "asset", "weight"]),
                pd.DataFrame(self._currency_weight_records, columns=["date", "currency", "psi"]))

    def _bundle(self, R_train: pd.DataFrame, R_c, F, assets: list[str], C: np.ndarray, start, end) -> dict:
        def window(frame):
            return frame.loc[start:end] if frame is not None else pd.DataFrame()
        return {self.training_return_key: R_train.loc[start:end, assets], "R_c": window(R_c), "F": window(F), "C": C}

    def run(self, data_bundle: dict) -> pd.Series:
        """Daily out-of-sample portfolio returns.

        data_bundle holds the asset returns under asset_return_key (and the training returns under
        training_return_key, if different), "asset_to_currency", and optionally "R_c" (excess currency
        returns), "F" (forward premiums, used by the mean-risk optimizers) and "universe_mask".
        """
        R_fh = data_bundle[self.asset_return_key]
        R_train = data_bundle[self.training_return_key] if self.training_return_key != self.asset_return_key else R_fh
        R_c = data_bundle.get("R_c")
        F = data_bundle.get("F")
        asset_to_currency = data_bundle["asset_to_currency"]
        universe_mask = data_bundle.get("universe_mask", None)

        all_assets = R_fh.columns.tolist()
        # unhedged strategies have no currency returns, so the currencies come from the asset mapping
        all_currencies = R_c.columns.tolist() if R_c is not None else sorted(set(asset_to_currency.values()))
        C = pd.DataFrame(0.0, index=all_assets, columns=all_currencies)
        for asset, ccy in asset_to_currency.items():
            if asset in C.index and ccy in C.columns:
                C.loc[asset, ccy] = 1.0

        dates = R_fh.index
        rebalance_dates = self._get_rebalance_dates(dates)
        full_forward_turnover = self.rebalancing_frequency.strip().lower() != "daily"
        drifted_x = pd.Series(0.0, index=all_assets)
        drifted_w = pd.Series(0.0, index=all_currencies)
        drifted_phi = pd.Series(0.0, index=all_currencies)
        is_first_period = True
        period_returns = []
        self._asset_turnovers, self._currency_turnovers = [], []
        self._selected_params, self._asset_weight_records, self._currency_weight_records = [], [], []
        n_periods = len(rebalance_dates) - 1

        for i in range(n_periods):
            if self._logging:
                done = int(40 * (i + 1) / n_periods)
                logger.info("Backtest progress: |%s%s| %.1f%% (%d/%d)", "█" * done, "-" * (40 - done),
                            (i + 1) / n_periods * 100, i + 1, n_periods)
            t, t_next = rebalance_dates[i], rebalance_dates[i + 1]
            if pd.Timestamp(t) - pd.DateOffset(years=self.window_years) < dates[0]:
                continue
            try:
                s = self._get_estimation_and_validation_splits(dates, t, t_next)
            except ValueError:
                continue

            # the investable universe: in the mask at t, with a complete history over the estimation window
            full_window = slice(s["full_estimation_start"], s["full_estimation_end"])
            if universe_mask is not None:
                mask_row = universe_mask.loc[t] if t in universe_mask.index else pd.Series(dtype=float)
                current = [a for a in mask_row[mask_row.notna()].index if a in R_train.columns]
                if not current:
                    continue
                full_slice = R_train.loc[full_window, current]
            else:
                full_slice = R_train.loc[full_window]
            try:
                assets = self._filter_universe(full_slice)
            except ValueError:
                continue
            C_sub = C.loc[assets].values

            grid = self._generate_grid()
            best_params, best_score = grid[0], float("inf")
            if self.param_grid:
                for params in grid:
                    opt = self.optimizer(hyperparams=params)
                    opt.fit(self._bundle(R_train, R_c, F, assets, C_sub, s["train_start"], s["train_end"]))
                    candidate = opt.optimize()
                    score = opt.score(candidate, self._bundle(R_train, R_c, F, assets, C_sub,
                                                              s["validation_start"], s["validation_end"]))
                    if score < best_score:
                        best_score, best_params = score, params
            self._selected_params.append({"date": t, "n_assets": len(assets), "val_score": best_score, **best_params})

            opt = self.optimizer(hyperparams=best_params)
            opt.fit(self._bundle(R_train, R_c, F, assets, C_sub, s["full_estimation_start"], s["full_estimation_end"]))
            raw = opt.optimize()
            if isinstance(raw, dict):
                x_sub, psi_sub = np.asarray(raw["x"], dtype=float), np.asarray(raw["psi"], dtype=float)
            else:
                x_sub, psi_sub = np.asarray(raw, dtype=float), np.zeros(len(all_currencies), dtype=float)
            x_t = pd.Series(0.0, index=all_assets)
            x_t[assets] = x_sub
            psi_t = pd.Series(psi_sub, index=all_currencies)
            self._asset_weight_records += [{"date": t, "asset": a, "weight": float(w)} for a, w in zip(assets, x_sub)]
            self._currency_weight_records += [{"date": t, "currency": c, "psi": float(p)}
                                              for c, p in zip(all_currencies, psi_sub)]

            # currency exposure w and forward positions phi; unhedged strategies hold no forwards
            w_t = C.T @ x_t
            phi_t = pd.Series(0.0, index=all_currencies) if self._is_unhedged else w_t - psi_t

            if is_first_period:
                tc_total = 0.0
                is_first_period = False
            else:
                asset_tc = self._compute_asset_transaction_costs(drifted_x, x_t)
                currency_tc = self._compute_currency_transaction_costs(
                    drifted_forward_weights=drifted_phi, new_forward_weights=phi_t,
                    drifted_currency_weights=drifted_w, new_currency_weights=w_t,
                    full_forward_turnover=full_forward_turnover)
                tc_total = asset_tc + currency_tc
                # turnover: forward notional when hedged, spot conversions when unhedged
                self._asset_turnovers.append(float((x_t - drifted_x).abs().sum()))
                traded = (w_t - drifted_w) if self._is_unhedged else phi_t
                self._currency_turnovers.append(float(traded.drop("USD", errors="ignore").abs().sum()))

            R_fh_hold = R_fh.loc[s["holding_start"]:s["holding_end"], assets].fillna(0.0)
            R_c_hold = (R_c.loc[s["holding_start"]:s["holding_end"]].fillna(0.0) if R_c is not None
                        else pd.DataFrame(0.0, index=R_fh_hold.index, columns=all_currencies))
            x_vec, psi_vec = x_t[assets].values, psi_t.values
            asset_growth = (1.0 + R_fh_hold).cumprod(axis=0)
            wealth = asset_growth.to_numpy() @ x_vec + (R_c_hold.to_numpy() @ psi_vec).cumsum()
            period_ret = pd.Series(0.0, index=R_fh_hold.index, dtype=float)
            period_ret.iloc[0] = wealth[0] - 1.0 - tc_total
            if len(period_ret) > 1:
                period_ret.iloc[1:] = wealth[1:] / wealth[:-1] - 1.0
            period_returns.append(period_ret)

            # the assets drift with their returns over the holding period, the overlay psi does not
            unnorm_x = asset_growth.iloc[-1].to_numpy() * x_vec
            total = unnorm_x.sum()
            drifted_x = pd.Series(0.0, index=all_assets)
            drifted_x[assets] = unnorm_x / total if total > 0 else x_vec
            drifted_w = C.T @ drifted_x
            drifted_phi = pd.Series(0.0, index=all_currencies) if self._is_unhedged else drifted_w - psi_t

        if not period_returns:
            return pd.Series(dtype=float)
        return pd.concat(period_returns).sort_index()
