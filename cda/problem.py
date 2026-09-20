"""Problem wrapper: normalised search space, batch evaluation and budget accounting.

Every optimiser in this package searches the unit hypercube ``[0, 1]^n`` and never
sees the real bounds of the objective function.  That is not an implementation
convenience: the Contagious Diseases Algorithm explicitly requires it.  The paper
fixes the maximum contact radius at ``1/6`` of a normalised dimension, so the step
sizes in :mod:`cda.core` are only meaningful once every axis has the same extent.

The wrapper also owns the evaluation counter.  CDA produces a different number of
children every iteration, so comparing it to a fixed-population algorithm is only
fair on a per-evaluation basis, which is exactly the x-axis used in the figures of
the original manuscript.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

__all__ = ["Problem", "BudgetExhausted"]


class BudgetExhausted(RuntimeError):
    """Raised internally when an optimiser asks for more evaluations than it has."""


@dataclass
class Problem:
    """A box-constrained minimisation problem defined on normalised coordinates.

    Parameters
    ----------
    func:
        Vectorised objective.  Receives an array of shape ``(k, n)`` in *real*
        coordinates and returns ``(k,)`` objective values.
    lower, upper:
        Real-space bounds, each of length ``n``.
    name:
        Human-readable identifier used in result tables.
    optimum_value:
        Known global minimum, when it is known.  Used only for reporting the
        distance to the optimum and the success rate; never used by an optimiser.
    optimum_position:
        A known global minimiser in real coordinates, when one is known.
    budget:
        Maximum number of objective evaluations.  ``None`` means unlimited.
    tags:
        Free-form labels such as ``"multimodal"`` or ``"separable"``.
    """

    func: Callable[[np.ndarray], np.ndarray]
    lower: np.ndarray
    upper: np.ndarray
    name: str = "problem"
    optimum_value: float | None = None
    optimum_position: np.ndarray | None = None
    budget: int | None = None
    tags: tuple[str, ...] = ()

    # --- mutable run state -------------------------------------------------
    evaluations: int = field(default=0, init=False)
    best_value: float = field(default=np.inf, init=False)
    best_position: np.ndarray | None = field(default=None, init=False)
    history_evals: list[int] = field(default_factory=list, init=False)
    history_best: list[float] = field(default_factory=list, init=False)
    history_batch_mean: list[float] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.lower = np.asarray(self.lower, dtype=float).ravel()
        self.upper = np.asarray(self.upper, dtype=float).ravel()
        if self.lower.shape != self.upper.shape:
            raise ValueError("lower and upper bounds must have the same length")
        if np.any(self.upper <= self.lower):
            raise ValueError("every upper bound must exceed its lower bound")
        if self.optimum_position is not None:
            self.optimum_position = np.asarray(self.optimum_position, dtype=float).ravel()

    # --- geometry ----------------------------------------------------------
    @property
    def dim(self) -> int:
        return int(self.lower.size)

    @property
    def span(self) -> np.ndarray:
        return self.upper - self.lower

    def denormalise(self, x_unit: np.ndarray) -> np.ndarray:
        """Map ``[0, 1]^n`` coordinates to real coordinates."""
        x_unit = np.atleast_2d(np.asarray(x_unit, dtype=float))
        return self.lower + x_unit * self.span

    def normalise(self, x_real: np.ndarray) -> np.ndarray:
        """Map real coordinates back to ``[0, 1]^n``."""
        x_real = np.atleast_2d(np.asarray(x_real, dtype=float))
        return (x_real - self.lower) / self.span

    # --- evaluation --------------------------------------------------------
    @property
    def remaining(self) -> int:
        if self.budget is None:
            return np.iinfo(np.int64).max
        return max(0, self.budget - self.evaluations)

    def exhausted(self) -> bool:
        return self.remaining <= 0

    def evaluate(self, x_unit: np.ndarray, *, record: bool = True) -> np.ndarray:
        """Evaluate a batch of normalised points, honouring the budget.

        If the batch is larger than the remaining budget it is truncated, so an
        optimiser can never overspend.  The returned array is then shorter than
        the requested batch and the caller is expected to stop.
        """
        x_unit = np.atleast_2d(np.asarray(x_unit, dtype=float))
        if x_unit.shape[1] != self.dim:
            raise ValueError(f"expected {self.dim} dimensions, got {x_unit.shape[1]}")

        allowed = min(len(x_unit), self.remaining)
        if allowed <= 0:
            return np.empty(0, dtype=float)
        x_unit = x_unit[:allowed]

        values = np.asarray(self.func(self.denormalise(x_unit)), dtype=float).ravel()
        if values.size != allowed:
            raise ValueError("objective returned the wrong number of values")
        # A non-finite objective must never win; push it to +inf so that the
        # "catches the disease" comparison in CDA rejects it deterministically.
        values = np.where(np.isfinite(values), values, np.inf)

        self.evaluations += allowed
        local_best = int(np.argmin(values))
        if values[local_best] < self.best_value:
            self.best_value = float(values[local_best])
            self.best_position = x_unit[local_best].copy()

        if record:
            self.history_evals.append(self.evaluations)
            self.history_best.append(self.best_value)
            finite = values[np.isfinite(values)]
            self.history_batch_mean.append(float(finite.mean()) if finite.size else np.inf)
        return values

    # --- reporting ---------------------------------------------------------
    def best_real(self) -> np.ndarray | None:
        """The best position found so far, in real coordinates."""
        if self.best_position is None:
            return None
        return self.denormalise(self.best_position)[0]

    def error(self) -> float:
        """Absolute gap between the best value found and the known optimum."""
        if self.optimum_value is None:
            return float("nan")
        return abs(self.best_value - self.optimum_value)

    def evals_to_target(self, tolerance: float) -> int | None:
        """Evaluations needed to first reach ``optimum_value + tolerance``."""
        if self.optimum_value is None:
            return None
        target = self.optimum_value + tolerance
        for evals, best in zip(self.history_evals, self.history_best):
            if best <= target:
                return evals
        return None

    def reset(self) -> None:
        self.evaluations = 0
        self.best_value = np.inf
        self.best_position = None
        self.history_evals = []
        self.history_best = []
        self.history_batch_mean = []

    def copy_fresh(self, budget: int | None = None) -> "Problem":
        """A clean instance of the same problem, optionally with a new budget."""
        return Problem(
            func=self.func,
            lower=self.lower.copy(),
            upper=self.upper.copy(),
            name=self.name,
            optimum_value=self.optimum_value,
            optimum_position=None if self.optimum_position is None else self.optimum_position.copy(),
            budget=self.budget if budget is None else budget,
            tags=self.tags,
        )


def make_problem(
    func: Callable[[np.ndarray], np.ndarray],
    bounds: Sequence[tuple[float, float]],
    **kwargs,
) -> Problem:
    """Build a :class:`Problem` from a list of ``(low, high)`` pairs."""
    lower = np.array([b[0] for b in bounds], dtype=float)
    upper = np.array([b[1] for b in bounds], dtype=float)
    return Problem(func=func, lower=lower, upper=upper, **kwargs)
