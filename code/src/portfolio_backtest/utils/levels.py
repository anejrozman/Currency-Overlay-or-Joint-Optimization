"""
The ES confidence levels swept by the level experiment, and how they are spelled.

One source of truth for four consumers that must agree exactly: the cluster
submitter (code/scripts/euler/run_es_levels.sh), the run collector
(code/scripts/collect_es_level_runs.py), the gap analysis
(code/scripts/in_sample_vs_out_of_sample_es_at_diff_levels.py) and its own
submitter.  A level that is spelled one way in a CSV row and another in a shard
filename silently splits one cell into two, so the spelling lives here rather
than in an f-string in each caller.

Levels are held in percent as floats.  ``format_level`` renders a whole level
without a decimal point, so the six original levels keep the names they were
written under -- ``es_joint_level_70``, ``gaps_unregularized_level70_top10.csv``
-- while the half levels added later read as ``72.5``.  Nothing already on disk
has to be renamed.

Author: Anej Rozman
"""

from __future__ import annotations

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
