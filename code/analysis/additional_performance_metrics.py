"""Realized ES, SOES, semi-deviation, MAD and ES ratio of the out-of-sample returns (Appendix G.5)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from common import MEASURES, RESULTS_DIR, TRADING_DAYS, UNIVERSES, excess_return, oos_returns, risk_free, top_runs

OUT_DIR = RESULTS_DIR / "additional_performance_metrics"
BETA = 0.8
# runs table and (strategy template, label) per family; EW is added to every family
FAMILIES = {
    "baseline": ("top_runs.csv", (("{}_curr_overlay", "Overlay"), ("{}_joint", "Joint"))),
    "regularized": ("top_runs.csv", (("reg_{}_curr_overlay", "Overlay"), ("reg_{}_joint", "Joint"),
                                     ("reg_{}_asset_full", "Full"), ("reg_{}_asset_zero", "Zero"))),
    "mean-risk": ("top_runs_w_mean.csv", (("{}_curr_overlay_w_mean", "Overlay"), ("{}_joint_w_mean", "Joint"))),
}
ROWS = (("ES", r"ES (\%)"), ("SOES", r"SOES (\%)"), ("SD", r"SD (\%)"), ("MAD", r"MAD (\%)"), ("ESR", "ESR"))
PANELS = {10: "Small Investable Universe (top 10 stocks per economy)",
          50: "Medium Investable Universe (top 50 stocks per economy)",
          100: "Large Investable Universe (top 100 stocks per economy)"}


def expected_shortfall(r: np.ndarray) -> float:
    return float(-r[r <= np.quantile(r, 1.0 - BETA)].mean())


def second_order_expected_shortfall(losses: np.ndarray) -> float:
    """min over alpha of alpha + E[((L - alpha)^+)^2]^(1/2) / (1 - beta)."""
    result = minimize_scalar(
        lambda a: a + math.sqrt(np.mean(np.maximum(losses - a, 0.0) ** 2)) / (1.0 - BETA),
        bounds=(float(losses.min()), float(losses.max())), method="bounded", options={"xatol": 1e-12})
    return float(result.fun)


def metrics(returns: pd.Series, rf: pd.Series) -> dict[str, float]:
    r = returns.to_numpy()
    deviation = r - r.mean()
    excess = excess_return(returns, rf)
    es = expected_shortfall(r)
    return {"ES": 100.0 * es, "SOES": 100.0 * second_order_expected_shortfall(-r),
            "SD": 100.0 * math.sqrt(np.mean(np.minimum(deviation, 0.0) ** 2)),
            "MAD": 100.0 * np.mean(np.abs(deviation)),
            "ESR": excess / (es * math.sqrt(TRADING_DAYS)), "ER": 100.0 * excess}


def number(value: float) -> str:
    text = f"{value:.2f}"
    return "0.00" if text == "-0.00" else f"${text}$" if text.startswith("-") else text


def block(cells: pd.DataFrame, programs: tuple[str, str], with_ew: bool, measure_header: bool = True) -> list[str]:
    """EW (or an empty column) and two programs per risk measure."""
    lines = ["    & & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{\textbf{{{m}}}}}" for m in MEASURES.values()) + r" \\",
             r"    \cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8} \cmidrule(lr){9-10} \cmidrule(lr){11-12}",
             ] if measure_header else []
    lines += [r"    \textbf{Metric} & " + (r"\textbf{EW}" if with_ew else "") + " & "
              + " & ".join(rf"\textbf{{{p}}}" for _ in MEASURES for p in programs) + r" \\", r"    \midrule"]
    cell = cells.set_index(["measure", "program"])
    for key, label in ROWS:
        values = [number(cell.at[("-", "EW"), key]) if with_ew else ""]
        values += [number(cell.at[(m, p), key]) for m in MEASURES for p in programs]
        lines.append(f"    {label} & " + " & ".join(values) + r" \\")
    return lines


def latex_table(frame: pd.DataFrame, family: str) -> str:
    lines = [r"  \begin{tabular}{l*{11}{c}}", r"    \toprule"]
    for i, universe in enumerate(UNIVERSES):
        cells = frame[frame["universe"] == universe]
        lines += [r"    \midrule"] * bool(i) + [
            rf"    \multicolumn{{12}}{{c}}{{\textbf{{Panel {'ABC'[i]}: {PANELS[universe]}}}}} \\", r"    \midrule"]
        if family == "regularized":
            # the benchmark rows share the measure columns above
            lines += block(cells, ("Overlay", "Joint"), with_ew=False) + [r"    \midrule"]
            lines += block(cells, ("Full", "Zero"), with_ew=True, measure_header=False)
        else:
            lines += block(cells, ("Overlay", "Joint"), with_ew=True)
    return "\n".join(lines + [r"    \bottomrule", r"  \end{tabular}"]) + "\n"


if __name__ == "__main__":
    rf = risk_free()
    rows = []
    for family, (runs_file, programs) in FAMILIES.items():
        table = top_runs(runs_file)
        for u in UNIVERSES:
            rows.append({"family": family, "universe": u, "measure": "-", "program": "EW",
                         **metrics(oos_returns(table, "equal_weight_optimizer", u), rf)})
            rows += [{"family": family, "universe": u, "measure": m, "program": label,
                      **metrics(oos_returns(table, template.format(m), u), rf)}
                     for m in MEASURES for template, label in programs]
    frame = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT_DIR / "metrics.csv", index=False)
    for family in FAMILIES:
        (OUT_DIR / f"table_{family}.tex").write_text(latex_table(frame[frame["family"] == family], family))
    print(frame.round(3).to_string(index=False))
