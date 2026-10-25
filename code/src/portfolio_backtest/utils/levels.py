"""The confidence levels of the level sweep and the measures it covers, shared by its submitter, collector
and gap analysis.

Levels are in percent; format_level writes whole levels without a decimal point (70) and half levels with one
(72.5).
"""

from __future__ import annotations

from dataclasses import dataclass

SWEEP_LEVELS: tuple[float, ...] = (50.0, 60.0, 70.0, 72.5, 75.0, 77.5, 80.0, 82.5, 85.0, 87.5, 90.0, 92.5, 95.0, 97.5)

# the levels drawn one panel each in the by-universe figure
ROUND_LEVELS: tuple[float, ...] = (60.0, 70.0, 80.0, 90.0, 95.0)

TOLERANCE: float = 1e-9


@dataclass(frozen=True)
class SweepMeasure:
    """Names, strategies and output locations of one measure of the level sweep."""

    key: str            # the risk-measure key in_sample_analysis.evaluate_risk takes
    label: str          # for log lines and figure titles
    short: str          # for axis labels and panel titles, where label is too long
    joint: str          # strategy_registry key of the joint program
    overlay: str        # strategy_registry key of the sequential overlay
    runs_file: str      # table under results/runs/ recording which run is which cell
    out_dir: str        # directory under results/ holding the shards and figures

    @property
    def strategies(self) -> dict[str, str]:
        """The swept pair, keyed by the role each plays in the gap."""
        return {"joint": self.joint, "overlay": self.overlay}


# only the unregularized programs are swept
SWEEP_MEASURES: dict[str, SweepMeasure] = {
    "es": SweepMeasure(
        key="es",
        label="Expected Shortfall",
        short="ES",
        joint="es_joint",
        overlay="es_curr_overlay",
        runs_file="top_runs_es_at_different_levels.csv",
        out_dir="in_sample_vs_out_of_sample_es_at_diff_levels",
    ),
    "soes": SweepMeasure(
        key="soes",
        label="Second-Order Expected Shortfall",
        short="SOES",
        joint="soes_joint",
        overlay="soes_curr_overlay",
        runs_file="top_runs_soes_at_different_levels.csv",
        out_dir="in_sample_vs_out_of_sample_soes_at_diff_levels",
    ),
}


def format_level(level: float) -> str:
    """Spelling of a level in row names, filenames and job tags."""
    return f"{level:g}"


def alpha_of(level: float) -> float:
    """Confidence level in percent to the alpha the optimizers take."""
    return level / 100.0


def level_of_alpha(alpha: float, levels: tuple[float, ...] = SWEEP_LEVELS) -> float | None:
    """The sweep level of alpha, or None; matched within a tolerance, since 0.725 is a level and 0.72 is
    not.
    """
    for level in levels:
        if abs(alpha * 100.0 - level) < TOLERANCE:
            return level
    return None


def parse_level(text: str, levels: tuple[float, ...] = SWEEP_LEVELS) -> float:
    """Parse a level from text, requiring it to be one of the sweep levels."""
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
