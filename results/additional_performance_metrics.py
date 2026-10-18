"""
Additional out-of-sample performance metrics (Appendix G.5).

For every optimizer of the baseline, regularized and mean-risk families, and for the
equally weighted portfolio, the realized daily out-of-sample returns are summarized by
the risk measures the optimizers minimize and by the Expected Shortfall ratio:

    ES      Expected Shortfall at beta = 0.8, the negative mean of the returns at or below
            their (1 - beta)-quantile, as the Backtester computes ES_80 (Appendix A)
    SOES    second-order Expected Shortfall at beta = 0.8 (Appendix A)
    SD      semi-deviation, the square root of the semi-variance below the mean (Appendix B)
    MAD     mean absolute deviation around the mean (Appendix C)
    ESR     Expected Shortfall ratio, ER / (sqrt(252) ES) with ER the annualized excess
            return of Ch. 4, as the Backtester computes it

ES, SOES, SD and MAD are daily values in percent, like the ES_80 of the Backtester
metrics. Each measure is evaluated on the empirical distribution of the full
out-of-sample series, the sample mean taking the place of the expected return.

Usage:
    python results/additional_performance_metrics.py

Author: Anej Rozman
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

RESULTS_DIR: Path = Path(__file__).resolve().parent
RUNS_DIR: Path = RESULTS_DIR / "runs"
RF_PATH: Path = RESULTS_DIR.parent / "data" / "raw" / "us_risk_free_rates.csv"
OUT_DIR: Path = RESULTS_DIR / "additional_performance_metrics"

TRADING_DAYS: int = 252
BETA: float = 0.8
MEASURES: tuple[str, ...] = ("mv", "smv", "mad", "es", "soes")
UNIVERSES: tuple[int, ...] = (10, 50, 100)

# (runs table, programs as (strategy template, label)) per family. EW is added to every family.
FAMILIES: dict[str, tuple[str, tuple[tuple[str, str], ...]]] = {
    "baseline": ("top_runs.csv", (("{}_curr_overlay", "Overlay"), ("{}_joint", "Joint"))),
    "regularized": ("top_runs.csv", (("reg_{}_curr_overlay", "Overlay"), ("reg_{}_joint", "Joint"),
                                     ("reg_{}_asset_full", "Full"), ("reg_{}_asset_zero", "Zero"))),
    "mean-risk": ("top_runs_w_mean.csv", (("{}_curr_overlay_w_mean", "Overlay"), ("{}_joint_w_mean", "Joint"))),
}
BENCHMARK: str = "equal_weight_optimizer"


def _table(name: str) -> pd.DataFrame:
    table = pd.read_csv(RUNS_DIR / name, skipinitialspace=True)
    table.columns = [c.strip() for c in table.columns]
    table["strategy_name"] = table["strategy_name"].astype(str).str.strip()
    return table.set_index("strategy_name")


def _returns(table: pd.DataFrame, strategy: str, universe: int) -> pd.Series:
    run = int(table.loc[strategy, f"top_{universe}_run"])
    path = RUNS_DIR / strategy / f"run_{run}" / "oos_returns.csv"
    return pd.read_csv(path, index_col=0, parse_dates=True)["return"].dropna()


def _risk_free() -> pd.Series:
    raw = pd.read_csv(RF_PATH, index_col=0)
    raw.index = pd.to_datetime(raw.index.astype(str), format="%Y%m%d")
    return raw["RF"] / 100.0


def expected_shortfall(returns: np.ndarray, beta: float = BETA) -> float:
    """Negative mean of the returns at or below their (1 - beta)-quantile, as in the Backtester."""
    threshold = np.quantile(returns, 1.0 - beta)
    return float(-returns[returns <= threshold].mean())


def second_order_expected_shortfall(losses: np.ndarray, beta: float = BETA) -> float:
    """min over alpha of alpha + ||(L - alpha)^+||_2 / (1 - beta), with ||Z||_2 = E[Z^2]^(1/2)."""
    def objective(alpha: float) -> float:
        return alpha + math.sqrt(np.mean(np.maximum(losses - alpha, 0.0) ** 2)) / (1.0 - beta)
    # The objective is convex in alpha and its minimizer lies within the range of the losses.
    result = minimize_scalar(objective, bounds=(float(losses.min()), float(losses.max())),
                             method="bounded", options={"xatol": 1e-12})
    return float(result.fun)


def metrics(returns: pd.Series, risk_free: pd.Series) -> dict[str, float]:
    r = returns.to_numpy()
    deviation = r - r.mean()
    # Annualized excess return as computed by the Backtester for the Ch. 5 tables: the CAGR of the
    # portfolio minus 252 times the mean daily risk-free rate (the first day has no rate and is skipped).
    rf = risk_free.reindex(returns.index).ffill()
    cagr = np.prod(1.0 + r) ** (TRADING_DAYS / len(r)) - 1.0
    excess = cagr - TRADING_DAYS * float(rf.mean())
    es = expected_shortfall(r)
    return {
        "ES": 100.0 * es,
        "SOES": 100.0 * second_order_expected_shortfall(-r),
        "SD": 100.0 * math.sqrt(np.mean(np.minimum(deviation, 0.0) ** 2)),
        "MAD": 100.0 * np.mean(np.abs(deviation)),
        # Like the Backtester, the ratio uses the ES annualized with sqrt(252), as the volatility of the Sharpe ratio.
        "ESR": excess / (es * math.sqrt(TRADING_DAYS)),
        "ER": 100.0 * excess,
    }


ROWS: tuple[tuple[str, str], ...] = (
    ("ES", r"ES (\%)"), ("SOES", r"SOES (\%)"), ("SD", r"SD (\%)"), ("MAD", r"MAD (\%)"), ("ESR", "ESR"),
)
MEASURE_HEADERS: tuple[str, ...] = ("Variance", "Semi-Variance", "MAD", "ES", "SOES")
UNIVERSE_PANELS: dict[int, str] = {
    10: "Small Investable Universe (top 10 stocks per economy)",
    50: "Medium Investable Universe (top 50 stocks per economy)",
    100: "Large Investable Universe (top 100 stocks per economy)",
}


def _number(value: float) -> str:
    text = f"{value:.2f}"
    if text == "-0.00":
        return "0.00"
    return f"${text}$" if text.startswith("-") else text


def _block(frame: pd.DataFrame, universe: int, programs: tuple[str, str], with_ew: bool,
           measure_header: bool = True) -> list[str]:
    """Header rows and metric rows of one block: EW (or an empty column) and two programs per measure."""
    cells = frame[frame["universe"] == universe]
    ew = cells[cells["program"] == "EW"].iloc[0]
    header = r"    \textbf{Metric} & " + (r"\textbf{EW}" if with_ew else "") + " & " + " & ".join(
        rf"\textbf{{{p}}}" for _ in MEASURES for p in programs) + r" \\"
    lines = []
    if measure_header:
        lines += [
            "    & & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{\textbf{{{m}}}}}" for m in MEASURE_HEADERS) + r" \\",
            r"    \cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8} \cmidrule(lr){9-10} \cmidrule(lr){11-12}",
        ]
    lines += [header, r"    \midrule"]
    for key, label in ROWS:
        values = [_number(ew[key]) if with_ew else ""]
        for measure in MEASURES:
            for program in programs:
                row = cells[(cells["measure"] == measure) & (cells["program"] == program)].iloc[0]
                values.append(_number(row[key]))
        lines.append(f"    {label} & " + " & ".join(values) + r" \\")
    return lines


def latex_table(frame: pd.DataFrame, family: str) -> str:
    """The tabular of one family, with a panel per universe."""
    lines = [r"  \begin{tabular}{l*{11}{c}}", r"    \toprule"]
    for index, universe in enumerate(UNIVERSES):
        letter = "ABC"[index]
        if index:
            lines.append(r"    \midrule")
        if family == "regularized":
            lines.append(rf"    \multicolumn{{12}}{{c}}{{\textbf{{Panel {letter}: {UNIVERSE_PANELS[universe]}}}}} \\")
            lines.append(r"    \midrule")
            lines += _block(frame, universe, ("Overlay", "Joint"), with_ew=False)
            # The benchmarks share the measure columns of the block above, so their header is not repeated.
            lines.append(r"    \midrule")
            lines += _block(frame, universe, ("Full", "Zero"), with_ew=True, measure_header=False)
        else:
            lines.append(rf"    \multicolumn{{12}}{{c}}{{\textbf{{Panel {letter}: {UNIVERSE_PANELS[universe]}}}}} \\")
            lines.append(r"    \midrule")
            lines += _block(frame, universe, ("Overlay", "Joint"), with_ew=True)
    lines += [r"    \bottomrule", r"  \end{tabular}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    risk_free = _risk_free()
    rows = []
    for family, (runs_file, programs) in FAMILIES.items():
        table = _table(runs_file)
        for universe in UNIVERSES:
            ew = metrics(_returns(table, BENCHMARK, universe), risk_free)
            rows.append({"family": family, "universe": universe, "measure": "-", "program": "EW", **ew})
            for measure in MEASURES:
                for template, label in programs:
                    values = metrics(_returns(table, template.format(measure), universe), risk_free)
                    rows.append({"family": family, "universe": universe, "measure": measure,
                                 "program": label, **values})
    frame = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT_DIR / "metrics.csv", index=False)
    for family in FAMILIES:
        (OUT_DIR / f"table_{family}.tex").write_text(latex_table(frame[frame["family"] == family], family))
    print(frame.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
