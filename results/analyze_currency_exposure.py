"""
Interactive dashboard of the currency exposure to one currency through the backtests.

The radio buttons on the left choose the group of optimizers and a foreign currency.
The grid has a row for the joint and the overlay optimizers and a column per
universe, and each panel draws the currency exposure psi of every risk measure,
set at each rebalancing date and held until the next one, as in Backtester.run.
The runs are those recorded in runs/top_runs.csv for the unregularized and
regularized optimizers, and in runs/top_runs_w_mean.csv for the unregularized
mean-risk optimizers, as in the drawdown and turnover figures. USD, the base
currency, is left out: its column of R_c is identically zero, so psi_USD is
unidentified and saved as zero.

The check buttons hide risk measures, and "Save PDF" writes the current view in
the thesis figure style to figures/currency_exposure/<group>/<currency>.pdf, with
the shown measures appended to the name when some are hidden.

Usage:
    python results/analyze_currency_exposure.py

Author: Anej Rozman
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.widgets import Button, CheckButtons, RadioButtons

RESULTS_DIR: Path = Path(__file__).resolve().parent
RUNS_DIR: Path = RESULTS_DIR / "runs"
OUT_DIR: Path = RESULTS_DIR / "figures" / "currency_exposure"

# Thesis names, table order and colours of the risk measures, as in the Chapter 5
# cumulative figures.
MEASURES: dict[str, tuple[str, str]] = {
    "mv": ("Variance", "tab:blue"),
    "smv": ("Semi-Variance", "tab:green"),
    "mad": ("MAD", "tab:orange"),
    "es": ("ES", "tab:red"),
    "soes": ("SOES", "tab:purple"),
}

# Button label: (runs table, strategy template filled with measure and program, output folder).
GROUPS: dict[str, tuple[str, str, str]] = {
    "Unregularized": ("top_runs.csv", "{}_{}", "unregularized"),
    "Regularized": ("top_runs.csv", "reg_{}_{}", "regularized"),
    "Mean-risk": ("top_runs_w_mean.csv", "{}_{}_w_mean", "with_mean_term"),
}

PROGRAMS: dict[str, str] = {"joint": "Joint", "curr_overlay": "Overlay"}
UNIVERSES: tuple[int, ...] = (10, 50, 100)
BASE_CURRENCY: str = "USD"

plt.rcParams["pdf.fonttype"] = 42

# The saved exposures of one run in long format, and the last date of its out-of-sample returns.
Run = tuple[pd.DataFrame, pd.Timestamp]


def _table(name: str) -> pd.DataFrame:
    table = pd.read_csv(RUNS_DIR / name, skipinitialspace=True)
    table.columns = [c.strip() for c in table.columns]
    table["strategy_name"] = table["strategy_name"].astype(str).str.strip()
    return table.set_index("strategy_name")


def _load_run(run_dir: Path) -> Run | None:
    if not (run_dir / "weights_currencies.csv").exists():
        return None
    psi = pd.read_csv(run_dir / "weights_currencies.csv", parse_dates=["date"])
    end = pd.read_csv(run_dir / "oos_returns.csv", usecols=["date"], parse_dates=["date"])["date"].iloc[-1]
    return psi[psi["currency"] != BASE_CURRENCY], end


def _load_runs() -> dict[tuple[str, str, str, int], Run | None]:
    """The run of every group, program, measure and universe; None where no weights were saved."""
    tables = {name: _table(name) for name in {runs_file for runs_file, _, _ in GROUPS.values()}}
    runs: dict[tuple[str, str, str, int], Run | None] = {}
    for group, (runs_file, template, _) in GROUPS.items():
        table = tables[runs_file]
        for program in PROGRAMS:
            for measure in MEASURES:
                strategy = template.format(measure, program)
                for universe in UNIVERSES:
                    run = table.at[strategy, f"top_{universe}_run"] if strategy in table.index else None
                    runs[(group, program, measure, universe)] = (
                        None if pd.isna(run) else _load_run(RUNS_DIR / strategy / f"run_{int(run)}")
                    )
    return runs


def _steps(run: Run, currency: str) -> pd.Series:
    """The exposure to one currency per rebalancing date, with the last one held to the end of the sample."""
    psi, end = run
    series = psi.loc[psi["currency"] == currency].set_index("date")["psi"].sort_index()
    return pd.concat([series, pd.Series([series.iloc[-1]], index=[end])])


def _grid(fig: Figure, text: float, **layout: float) -> np.ndarray:
    """A row per program and a column per universe, all on one date and one exposure axis."""
    axes = fig.subplots(len(PROGRAMS), len(UNIVERSES), sharex=True, sharey=True, squeeze=False,
                        gridspec_kw=layout)
    for row, label in enumerate(PROGRAMS.values()):
        for column, universe in enumerate(UNIVERSES):
            axis = axes[row, column]
            axis.grid(alpha=0.3, linewidth=0.5)
            axis.set_axisbelow(True)
            for side in ("top", "right"):
                axis.spines[side].set_visible(False)
            axis.tick_params(labelsize=text, length=2)
            axis.xaxis.set_major_locator(mdates.YearLocator(base=5))
            axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            if row == 0:
                axis.set_title(f"top-{universe}", fontsize=text + 1.5)
            if column == 0:
                axis.set_ylabel(label, fontsize=text + 1.5)
    return axes


def _draw(axes: np.ndarray, runs: dict[tuple[str, str, str, int], Run | None], group: str,
          currency: str, measures: list[str], linewidth: float) -> dict[str, object]:
    """Redraw every panel with a line per risk measure and return the legend handles."""
    handles: dict[str, object] = {}
    drawn: list[pd.Series] = []
    for row, program in enumerate(PROGRAMS):
        for column, universe in enumerate(UNIVERSES):
            axis = axes[row, column]
            for artist in [*axis.lines, *axis.texts]:
                artist.remove()
            axis.axhline(0.0, color="black", linewidth=0.6)
            missing = []
            for measure in measures:
                label, colour = MEASURES[measure]
                run = runs[(group, program, measure, universe)]
                if run is None:
                    missing.append(label)
                    continue
                series = _steps(run, currency)
                line, = axis.plot(series.index, series.to_numpy(), drawstyle="steps-post", color=colour,
                                  linewidth=linewidth)
                handles.setdefault(label, line)
                drawn.append(series)
            if missing:
                axis.text(0.02, 0.97, "No saved weights: " + ", ".join(missing), transform=axis.transAxes,
                          fontsize=7, va="top", color="0.4")
    if drawn:
        # One exposure scale for all panels, always including zero.
        low = min(0.0, min(float(s.min()) for s in drawn))
        high = max(0.0, max(float(s.max()) for s in drawn))
        pad = 0.05 * (high - low) or 0.05
        axes[0, 0].set_ylim(low - pad, high + pad)
        axes[0, 0].set_xlim(min(s.index[0] for s in drawn), max(s.index[-1] for s in drawn))
    return {MEASURES[m][0]: handles[MEASURES[m][0]] for m in measures if MEASURES[m][0] in handles}


def _save_pdf(runs: dict[tuple[str, str, str, int], Run | None], group: str, currency: str,
              measures: list[str], out_dir: Path) -> Path:
    """The current view in the thesis figure style, without the controls."""
    fig = Figure(figsize=(6.2, 3.4))
    axes = _grid(fig, text=7)
    handles = _draw(axes, runs, group, currency, measures, linewidth=0.8)
    fig.legend(list(handles.values()), list(handles), loc="lower center", ncol=len(handles),
               fontsize=7.5, frameon=False, handlelength=2.4)
    fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=0.6, w_pad=0.6)
    name = "_".join([currency.lower()] + ([] if len(measures) == len(MEASURES) else measures))
    path = out_dir / GROUPS[group][2] / f"{name}.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    return path


def _dashboard(runs: dict[tuple[str, str, str, int], Run | None]) -> list[object]:
    """Open the window and return its widgets, which must stay referenced while it is shown."""
    currencies = sorted({c for run in runs.values() if run is not None for c in run[0]["currency"]})
    fig = plt.figure(figsize=(14, 7.5))
    fig.canvas.manager.set_window_title("Currency exposure")
    axes = _grid(fig, text=9, left=0.2, right=0.985, bottom=0.08, top=0.9, wspace=0.08, hspace=0.15)

    def control(rect: tuple[float, float, float, float], title: str) -> plt.Axes:
        axis = fig.add_axes(rect)
        axis.set_title(title, loc="left", fontsize=10)
        return axis

    groups = RadioButtons(control((0.015, 0.78, 0.12, 0.12), "Group"), list(GROUPS),
                          label_props={"fontsize": [10]})
    currency = RadioButtons(control((0.015, 0.31, 0.12, 0.42), "Currency"), currencies,
                            label_props={"fontsize": [10]})
    measures = CheckButtons(control((0.015, 0.09, 0.12, 0.17), "Risk measures"),
                            [label for label, _ in MEASURES.values()], actives=[True] * len(MEASURES),
                            label_props={"color": [colour for _, colour in MEASURES.values()],
                                         "fontsize": [10] * len(MEASURES)})
    save = Button(fig.add_axes((0.015, 0.02, 0.12, 0.045)), "Save PDF")
    status = fig.text(0.2, 0.015, "", fontsize=8.5, color="0.35")

    def shown() -> list[str]:
        return [m for m, active in zip(MEASURES, measures.get_status()) if active]

    def redraw(_: object = None) -> None:
        _draw(axes, runs, groups.value_selected, currency.value_selected, shown(), linewidth=1.1)
        fig.suptitle(f"Currency exposure to {currency.value_selected}, "
                     f"{groups.value_selected.lower()} optimizers", fontsize=12)
        status.set_text("")
        fig.canvas.draw_idle()

    def export(_: object) -> None:
        if shown():
            path = _save_pdf(runs, groups.value_selected, currency.value_selected, shown(), OUT_DIR)
            status.set_text(f"Saved {path.relative_to(RESULTS_DIR.parent)}")
        else:
            status.set_text("Select at least one risk measure to save.")
        fig.canvas.draw_idle()

    for widget in (groups, currency, measures):
        widget.on_clicked(redraw)
    save.on_clicked(export)
    redraw()
    return [groups, currency, measures, save]


def main() -> None:
    runs = _load_runs()
    if all(run is None for run in runs.values()):
        raise SystemExit("No runs with saved weights recorded.")
    widgets = _dashboard(runs)  # noqa: F841 -- matplotlib holds widget callbacks weakly
    plt.show()


if __name__ == "__main__":
    main()
