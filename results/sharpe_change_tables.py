"""
Changes in the out-of-sample Sharpe ratio for Tables 5.2 and 5.6 of the thesis.

Table 5.2 compares each regularized overlay and joint optimizer with its unregularized
version, and Table 5.6 each mean-risk optimizer with the same unregularized optimizer
without a mean term. Panel A reports the absolute change of the Sharpe ratio and Panel B
the relative change in percent of the unregularized Sharpe ratio. The summary rows report
the change of the average Sharpe ratio of a group of risk measures.

The Sharpe ratios are computed from the saved out-of-sample returns as in the Backtester,
the CAGR of the portfolio minus 252 times the mean daily risk-free rate over the annualized
volatility, and are not rounded before taking the differences.

Usage:
    python results/sharpe_change_tables.py

Author: Anej Rozman
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

RESULTS_DIR: Path = Path(__file__).resolve().parent
RUNS_DIR: Path = RESULTS_DIR / "runs"
RF_PATH: Path = RESULTS_DIR.parent / "data" / "raw" / "us_risk_free_rates.csv"
OUT_DIR: Path = RESULTS_DIR / "sharpe_change"

TRADING_DAYS: int = 252
UNIVERSES: tuple[int, ...] = (10, 50, 100)
PROGRAMS: tuple[str, ...] = ("curr_overlay", "joint")
MEASURES: tuple[tuple[str, str], ...] = (
    ("mv", "Variance"), ("smv", "Semi-Variance"), ("mad", "MAD"), ("es", "ES"), ("soes", "SOES"),
)
# (runs table, strategy template, summary rows as (label, measures)) of the compared optimizers.
COMPARISONS: dict[str, tuple[str, str, tuple[tuple[str, tuple[str, ...]], ...]]] = {
    "regularization": ("top_runs.csv", "reg_{}_{}", (("Average", ("mv", "smv", "mad", "es", "soes")),)),
    "mean_term": ("top_runs_w_mean.csv", "{}_{}_w_mean", (("Deviation measures", ("mv", "smv", "mad")),
                                                          ("Tail measures", ("es", "soes")))),
}


def _table(name: str) -> pd.DataFrame:
    table = pd.read_csv(RUNS_DIR / name, skipinitialspace=True)
    table.columns = [c.strip() for c in table.columns]
    table["strategy_name"] = table["strategy_name"].astype(str).str.strip()
    return table.set_index("strategy_name")


def _risk_free() -> pd.Series:
    raw = pd.read_csv(RF_PATH, index_col=0)
    raw.index = pd.to_datetime(raw.index.astype(str), format="%Y%m%d")
    return raw["RF"] / 100.0


def sharpe_ratio(table: pd.DataFrame, strategy: str, universe: int, risk_free: pd.Series) -> float:
    run = int(table.loc[strategy, f"top_{universe}_run"])
    r = pd.read_csv(RUNS_DIR / strategy / f"run_{run}" / "oos_returns.csv", index_col=0,
                    parse_dates=True)["return"].dropna()
    rf = risk_free.reindex(r.index).ffill()
    cagr = (1.0 + r).prod() ** (TRADING_DAYS / len(r)) - 1.0
    return (cagr - TRADING_DAYS * float(rf.mean())) / (r.std(ddof=1) * math.sqrt(TRADING_DAYS))


def _absolute(value: float) -> str:
    text = f"{value:+.2f}"
    return "$0.00$" if text in ("+0.00", "-0.00") else f"${text}$"


def _relative(value: float) -> str:
    text = f"{value:+.0f}"
    return "$0$" if text in ("+0", "-0") else f"${text}$"


def _rows(base: dict, new: dict, groups, fmt, relative: bool) -> list[str]:
    """Rows of one panel: one per risk measure, then the summary rows."""
    def change(measures, program, universe):
        b = sum(base[(m, program, universe)] for m in measures) / len(measures)
        n = sum(new[(m, program, universe)] for m in measures) / len(measures)
        return 100.0 * (n - b) / abs(b) if relative else n - b

    lines = []
    for key, label in MEASURES:
        cells = [fmt(change((key,), p, u)) for p in PROGRAMS for u in UNIVERSES]
        lines.append(f"    {label:<21} & " + " & ".join(cells) + r" \\")
    lines.append(r"    \midrule")
    for label, measures in groups:
        cells = [fmt(change(measures, p, u)) for p in PROGRAMS for u in UNIVERSES]
        lines.append(f"    {label:<21} & " + " & ".join(cells) + r" \\")
    return lines


def latex_table(base: dict, new: dict, groups) -> str:
    lines = [
        r"  \begin{tabular}{l*{6}{c}}",
        r"    \toprule",
        r"                          & \multicolumn{3}{c}{\textbf{Overlay}}             & \multicolumn{3}{c}{\textbf{Joint}} \\",
        r"    \cmidrule(lr){2-4} \cmidrule(lr){5-7}",
        r"    \textbf{Risk Measure} & \textbf{Small} & \textbf{Medium} & \textbf{Large} & \textbf{Small} & \textbf{Medium} & \textbf{Large} \\",
        r"    \midrule",
        r"    \multicolumn{7}{c}{\textbf{Panel A: Absolute Change}} \\",
        r"    \midrule",
        *_rows(base, new, groups, _absolute, relative=False),
        r"    \midrule",
        r"    \multicolumn{7}{c}{\textbf{Panel B: Relative Change (\%)}} \\",
        r"    \midrule",
        *_rows(base, new, groups, _relative, relative=True),
        r"    \bottomrule",
        r"  \end{tabular}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    risk_free = _risk_free()
    unregularized = _table("top_runs.csv")
    base = {(m, p, u): sharpe_ratio(unregularized, f"{m}_{p}", u, risk_free)
            for m, _ in MEASURES for p in PROGRAMS for u in UNIVERSES}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, (runs_file, template, groups) in COMPARISONS.items():
        table = _table(runs_file)
        new = {(m, p, u): sharpe_ratio(table, template.format(m, p), u, risk_free)
               for m, _ in MEASURES for p in PROGRAMS for u in UNIVERSES}
        tex = latex_table(base, new, groups)
        (OUT_DIR / f"table_{name}.tex").write_text(tex)
        print(f"== {name}\n{tex}")


if __name__ == "__main__":
    main()
