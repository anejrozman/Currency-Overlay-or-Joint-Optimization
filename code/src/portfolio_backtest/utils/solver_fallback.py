"""
Gurobi solve with a two-stage fallback, for the conic optimizers whose production
Gurobi settings do not always converge.

The production settings -- FeasibilityTol and OptimalityTol at Gurobi's floor of
1e-9, BarQCPConvTol at 1e-8 and scaling switched off -- ask for more accuracy than
Gurobi's barrier can always deliver on the mean-bearing SOES programs.  When it
stalls short of them it stops with SUBOPTIMAL or ITERATION_LIMIT, which cvxpy 1.5
reports as ``user_limit`` rather than ``optimal_inaccurate``, so a single stalled
solve used to abort the whole backtest.

:func:`solve_with_fallback` keeps the production solve as the first attempt,
unchanged, so every solve that converges gives exactly the weights it gave before.
Only when that attempt fails does it retry Gurobi with :data:`GUROBI_ROBUST_PARAMS`,
and then Clarabel.  Every fallback is logged as a warning and recorded in the
caller's event list, which ``run_strategy.py`` writes to ``solver_fallbacks.csv`` in
the run directory.

Author: Anej Rozman
"""

from __future__ import annotations

import logging

import cvxpy as cp
import numpy as np

logger = logging.getLogger(__name__)

# Second attempt.  Gurobi's default tolerances (1e-6) and automatic scaling replace
# the production settings, and the homogeneous barrier is switched on: it is built
# for programs close to infeasible or unbounded, and the mean-bearing SOES programs
# are reported unbounded without their overlay cap.
GUROBI_ROBUST_PARAMS = {
    "Method": 2,
    "BarHomogeneous": 1,
    "NumericFocus": 2,
    "Threads": 3,
}

# The attempts in order, as recorded in solver_fallbacks.csv.
ATTEMPTS = ("gurobi", "gurobi_robust", "clarabel")

_ACCEPTED_STATUSES = ("optimal", "optimal_inaccurate")

# Gurobi's own termination codes, for the log.  cvxpy folds most of these into
# ``user_limit``, which hides why the barrier stopped.
_GUROBI_STATUS = {
    2: "OPTIMAL",
    3: "INFEASIBLE",
    4: "INF_OR_UNBD",
    5: "UNBOUNDED",
    7: "ITERATION_LIMIT",
    9: "TIME_LIMIT",
    11: "INTERRUPTED",
    12: "NUMERIC",
    13: "SUBOPTIMAL",
    16: "WORK_LIMIT",
    17: "MEM_LIMIT",
}


def _describe(problem: cp.Problem, solver: str) -> str:
    """Status of the solve just finished, with Gurobi's own code where available."""
    text = str(problem.status)
    if solver == cp.GUROBI:
        model = getattr(problem.solver_stats, "extra_stats", None)
        try:
            code, iters = model.Status, model.BarIterCount
        except Exception:  # no Gurobi model attached, or the attribute is unavailable
            pass
        else:
            text += f" (Gurobi {_GUROBI_STATUS.get(code, code)}, {iters} barrier iterations)"
    value = problem.value
    if value is not None and np.isfinite(value):
        text += f", objective {value:.6e}"
    return text


def _attempt(problem: cp.Problem, solver: str, params: dict) -> tuple[bool, str]:
    """Run one solve.  Returns whether it was accepted and a description of it."""
    try:
        problem.solve(solver=solver, verbose=False, **params)
    except cp.error.SolverError as exc:
        return False, f"solver error ({exc})"
    return problem.status in _ACCEPTED_STATUSES, _describe(problem, solver)


def solve_with_fallback(
    problem: cp.Problem,
    gurobi_params: dict,
    *,
    context: dict,
    events: list[dict],
    log: logging.Logger = logger,
) -> str:
    """
    Solve with Gurobi at ``gurobi_params``; if that does not end ``optimal`` or
    ``optimal_inaccurate``, retry Gurobi with :data:`GUROBI_ROBUST_PARAMS`, then
    Clarabel.

    Args:
        problem: The cvxpy problem.  Its variables hold the accepted solution on return.
        gurobi_params: The caller's production Gurobi parameters, tried first and
            unchanged.
        context: Identifies the solve in the log and in ``events``, e.g. the window
            dates and hyperparameters.
        events: Receives one record per solve that needed a fallback, including the
            one that failed outright.
        log: Logger to report through, so the warning carries the caller's module.

    Returns:
        The attempt that was accepted: ``"gurobi"``, ``"gurobi_robust"`` or
        ``"clarabel"``.

    Raises:
        RuntimeError: If all three attempts fail.
    """
    plan = {
        "gurobi": (cp.GUROBI, gurobi_params),
        "gurobi_robust": (cp.GUROBI, GUROBI_ROBUST_PARAMS),
        "clarabel": (cp.CLARABEL, {}),
    }
    outcomes = {name: "not run" for name in ATTEMPTS}
    accepted = None
    for name in ATTEMPTS:
        ok, outcomes[name] = _attempt(problem, *plan[name])
        if ok:
            accepted = name
            break

    if accepted == "gurobi":
        return accepted

    events.append({**context, **outcomes, "accepted": accepted or "none"})
    where = ", ".join(f"{key}={value}" for key, value in context.items())
    tried = "; ".join(f"{name}: {outcomes[name]}" for name in ATTEMPTS)
    if accepted is None:
        log.error("Solver fallback FAILED [%s]. %s", where, tried)
        raise RuntimeError(f"Optimization failed with every solver: {tried}")
    log.warning("Solver fallback [%s]. %s. Accepted %s.", where, tried, accepted)
    return accepted
