"""Gurobi solve with fallbacks to robust Gurobi settings and then Clarabel, for the conic optimizers.

The first attempt uses the caller's settings unchanged, so a solve that converges gives the same weights as
without the fallback.
"""

from __future__ import annotations

import logging

import cvxpy as cp
import numpy as np

logger = logging.getLogger(__name__)

# default tolerances and scaling, with the homogeneous barrier for nearly unbounded programs
GUROBI_ROBUST_PARAMS = {
    "Method": 2,
    "BarHomogeneous": 1,
    "NumericFocus": 2,
    "Threads": 3,
}

ATTEMPTS = ("gurobi", "gurobi_robust", "clarabel")

_ACCEPTED_STATUSES = ("optimal", "optimal_inaccurate")

# Gurobi termination codes, which cvxpy reports as user_limit
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
    """Solve status, with Gurobi's own code and the objective where available."""
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
    """Run one solve; returns whether it was accepted and a description."""
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
    """Solve with gurobi_params, then GUROBI_ROBUST_PARAMS, then Clarabel; returns the accepted attempt.

    Every solve that needed a fallback is logged and appended to events. Raises RuntimeError if all fail.
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
