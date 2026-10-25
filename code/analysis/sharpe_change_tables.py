"""Change in out-of-sample Sharpe ratio from regularization and from a mean term (Tables 5.2 and 5.6)."""

from __future__ import annotations

import math

from common import MEASURES, RESULTS_DIR, TRADING_DAYS, UNIVERSES, excess_return, oos_returns, risk_free, top_runs

OUT_DIR = RESULTS_DIR / "sharpe_change"
PROGRAMS = ("curr_overlay", "joint")
# runs table, strategy template and summary rows (label, measures) per comparison
COMPARISONS = {
    "regularization": ("top_runs.csv", "reg_{}_{}", (("Average", tuple(MEASURES)),)),
    "mean_term": ("top_runs_w_mean.csv", "{}_{}_w_mean", (("Deviation measures", ("mv", "smv", "mad")),
                                                          ("Tail measures", ("es", "soes")))),
}
HEADER = r"""  \begin{tabular}{l*{6}{c}}
    \toprule
                          & \multicolumn{3}{c}{\textbf{Overlay}}             & \multicolumn{3}{c}{\textbf{Joint}} \\
    \cmidrule(lr){2-4} \cmidrule(lr){5-7}
    \textbf{Risk Measure} & \textbf{Small} & \textbf{Medium} & \textbf{Large} & \textbf{Small} & \textbf{Medium} & \textbf{Large} \\
    \midrule"""


def sharpe_ratios(runs_file: str, template: str, rf) -> dict:
    table = top_runs(runs_file)
    ratios = {}
    for m in MEASURES:
        for p in PROGRAMS:
            for u in UNIVERSES:
                r = oos_returns(table, template.format(m, p), u)
                ratios[m, p, u] = excess_return(r, rf) / (r.std(ddof=1) * math.sqrt(TRADING_DAYS))
    return ratios


def signed(value: float, decimals: int) -> str:
    text = f"{value:+.{decimals}f}"
    return f"${text[1:]}$" if float(text) == 0 else f"${text}$"


def panel(base: dict, new: dict, groups, relative: bool) -> list[str]:
    def change(measures, p, u):
        b = sum(base[m, p, u] for m in measures) / len(measures)
        n = sum(new[m, p, u] for m in measures) / len(measures)
        return signed(100.0 * (n - b) / abs(b), 0) if relative else signed(n - b, 2)

    rows = [(label, (key,)) for key, label in MEASURES.items()] + [None] + list(groups)
    return [r"    \midrule" if row is None else f"    {row[0]:<21} & "
            + " & ".join(change(row[1], p, u) for p in PROGRAMS for u in UNIVERSES) + r" \\" for row in rows]


def latex_table(base: dict, new: dict, groups) -> str:
    lines = [HEADER, r"    \multicolumn{7}{c}{\textbf{Panel A: Absolute Change}} \\", r"    \midrule",
             *panel(base, new, groups, relative=False), r"    \midrule",
             r"    \multicolumn{7}{c}{\textbf{Panel B: Relative Change (\%)}} \\", r"    \midrule",
             *panel(base, new, groups, relative=True), r"    \bottomrule", r"  \end{tabular}"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    rf = risk_free()
    base = sharpe_ratios("top_runs.csv", "{}_{}", rf)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, (runs_file, template, groups) in COMPARISONS.items():
        tex = latex_table(base, sharpe_ratios(runs_file, template, rf), groups)
        (OUT_DIR / f"table_{name}.tex").write_text(tex)
        print(f"== {name}\n{tex}")
