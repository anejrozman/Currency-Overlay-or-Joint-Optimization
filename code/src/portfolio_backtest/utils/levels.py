"""
The confidence levels swept by the level experiment, and how they are spelled.

One source of truth for four consumers that must agree exactly: the cluster
submitter (code/scripts/euler/run_levels.sh), the run collector
(code/scripts/collect_level_runs.py), the gap analysis
(code/scripts/in_sample_vs_out_of_sample_at_diff_levels.py) and its own
submitter.  A level that is spelled one way in a CSV row and another in a shard
filename silently splits one cell into two, so the spelling lives here rather
than in an f-string in each caller.

The same four consumers must also agree on what a measure is called, which pair
of strategies it sweeps, and where its table and results land.  SWEEP_MEASURES
holds that too, for the same reason: a strategy named one way in a submitter and
another in a collector produces an empty table rather than an error.

Levels are held in percent as floats.  ``format_level`` renders a whole level
without a decimal point, so the six original levels keep the names they were
written under -- ``es_joint_level_70``, ``gaps_unregularized_level70_top10.csv``
-- while the half levels added later read as ``72.5``.  Nothing already on disk
has to be renamed.

Author: Anej Rozman
"""

from __future__ import annotations

from dataclasses import dataclass

# The sweep, ascending. 50-95 were run first; the 2.5-point grid between 70 and
# 97.5 was added afterwards to resolve the top end, where the in-sample gap rises
# fastest and the out-of-sample tail thins fastest.
ES_LEVELS: tuple[float, ...] = (
    50.0, 60.0, 70.0,
    72.5, 75.0, 77.5,
    80.0,
    82.5, 85.0, 87.5,
    90.0,
    92.5, 95.0, 97.5,
)

# Levels drawn one-panel-each on the by-universe figure. The full sweep is 14
# levels, which at one panel apiece is a figure no page can hold; the round ones
# carry the shape and the dense grid is what the by-level figure is for.
ROUND_LEVELS: tuple[float, ...] = (60.0, 70.0, 80.0, 90.0, 95.0)

TOLERANCE: float = 1e-9

# The levels are a property of the experiment, not of the measure: ES and SOES are
# swept over the same grid so their panels can be read side by side. ES_LEVELS is
# kept as the name the sweep was first written under; SWEEP_LEVELS is what the
# measure-neutral callers use.
SWEEP_LEVELS: tuple[float, ...] = ES_LEVELS


@dataclass(frozen=True)
class SweepMeasure:
    """What one measure of the level sweep is called and where its artifacts go."""

    key: str            # the risk-measure key in_sample_analysis.evaluate_risk takes
    label: str          # for log lines and figure titles
    joint: str          # strategy_registry key of the joint program
    overlay: str        # strategy_registry key of the sequential overlay
    runs_file: str      # table under results/runs/ recording which run is which cell
    out_dir: str        # directory under results/ holding the shards and figures

    @property
    def strategies(self) -> dict[str, str]:
        """The swept pair, keyed by the role each plays in the gap."""
        return {"joint": self.joint, "overlay": self.overlay}


# Only the UNREGULARIZED programs are swept: the regularized variants exist at
# alpha = 0.80 alone, and their penalties would confound the level with the
# shrinkage the validation search picks per level.
SWEEP_MEASURES: dict[str, SweepMeasure] = {
    "es": SweepMeasure(
        key="es",
        label="Expected Shortfall",
        joint="es_joint",
        overlay="es_curr_overlay",
        runs_file="top_runs_es_at_different_levels.csv",
        out_dir="in_sample_vs_out_of_sample_es_at_diff_levels",
    ),
    "soes": SweepMeasure(
        key="soes",
        label="Second-Order Expected Shortfall",
        joint="soes_joint",
        overlay="soes_curr_overlay",
        runs_file="top_runs_soes_at_different_levels.csv",
        out_dir="in_sample_vs_out_of_sample_soes_at_diff_levels",
    ),
}


def measure_of(name: str) -> SweepMeasure:
    """
    Look up a sweep measure by key, rejecting anything unswept.

    Raises:
        ValueError: If *name* is not a measure the sweep covers.
    """
    try:
        return SWEEP_MEASURES[name]
    except KeyError:
        raise ValueError(
            f"'{name}' is not a swept measure "
            f"({', '.join(sorted(SWEEP_MEASURES))})."
        ) from None


def format_level(level: float) -> str:
    """
    Canonical spelling of *level* for row names, filenames and job tags.

    Whole levels lose the decimal point (70.0 -> "70") so names written before
    the half levels existed still resolve; half levels keep one decimal
    (72.5 -> "72.5").
    """
    return f"{level:g}"


def alpha_of(level: float) -> float:
    """Confidence level in percent to the alpha the optimizers take."""
    return level / 100.0


def level_of_alpha(alpha: float, levels: tuple[float, ...] = ES_LEVELS) -> float | None:
    """
    The sweep level *alpha* belongs to, or None if it is not one of them.

    Matched by tolerance rather than by rounding: round() would map 0.725 and
    0.72 onto the same level, and 0.725 is a level while 0.72 is a mistake worth
    surfacing.
    """
    for level in levels:
        if abs(alpha * 100.0 - level) < TOLERANCE:
            return level
    return None


def parse_level(text: str, levels: tuple[float, ...] = ES_LEVELS) -> float:
    """
    Parse a level written as a string, requiring it to be one of *levels*.

    Raises:
        ValueError: If *text* is not a number or is not a level in the sweep.
    """
    try:
        value = float(text)
    except ValueError:
        raise ValueError(f"'{text}' is not a number.") from None
    for level in levels:
        if abs(value - level) < TOLERANCE:
            return level
    raise ValueError(
        f"{text} is not one of the sweep levels "
        f"({', '.join(format_level(x) for x in levels)})."
    )
