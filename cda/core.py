"""The Contagious Diseases Algorithm (CDA).

Reference implementation of:

    A. Mohammadi, M. R. Farmani and C. Lucas,
    "Development of a New Evolutionary Algorithm Inspired by Outbreak of
    Contagious Diseases" (2010).

The metaphor
------------
The search space is a *community*.  Every point in it is an *individual*, and an
individual's objective value is its *health care factor* (HCF) -- a low HCF means
poor hygiene and high susceptibility.  Individuals that currently carry the
disease are *transmitters*.  Each iteration is one day of the outbreak:

1. Every transmitter makes some high-risk contacts.  A transmitter with a *lower*
   HCF (a better solution) makes *more* contacts but over *shorter* distances; a
   transmitter with a high HCF makes a single long-range contact.
2. A contacted individual catches the disease only if its HCF is no worse than the
   transmitter's.  Otherwise it is healthy enough to shrug the infection off.
3. Yesterday's transmitters enter quarantine and stop spreading.  The newly
   infected individuals are tomorrow's transmitters.

Minimising the objective is therefore the same as letting the disease find the
least healthy person in the community.

Equation 2 and its typesetting error
------------------------------------
Equation 2 of the manuscript is typeset as

    sigma = r * (1/6 - 1/(6 * alpha**(m-1))) + 1/6

which awards the *best* transmitter a standard deviation approaching ``1/3`` at
late iterations.  That grows the step size of the best solution and directly
contradicts the surrounding prose, which states that ``1/6`` is the *maximum*
standard deviation and that the minimum "will be decreased by increase in
iteration number".  The default here is the reading that matches the prose, a
linear interpolation between the shrinking minimum and the fixed maximum:

    sigma = r * (1/(6 * alpha**(m-1)) - 1/6) + 1/6

Pass ``sigma_formula="literal"`` to get the equation exactly as printed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .problem import Problem

__all__ = ["CDAConfig", "CDAResult", "CDA", "minimise"]

SigmaFormula = Literal["corrected", "literal"]
BoundsMode = Literal["clip", "reflect", "resample"]
Extinction = Literal["stop", "restart"]


@dataclass
class CDAConfig:
    """Tuning parameters for :class:`CDA`.

    The manuscript pins down ``n_transmitters`` (50) and ``alpha`` (2) in Table 1
    and declares ``contact_num_max`` an input parameter without ever printing its
    value.  The remaining fields cover behaviour the paper leaves unspecified;
    each default is documented in the README.
    """

    # --- parameters taken from the paper ----------------------------------
    n_transmitters: int = 50
    """Number of initial transmitters (the paper's initial population)."""

    contact_num_max: int = 5
    """Contacts made by the best transmitter.  The worst always makes exactly 1."""

    alpha: float = 2.0
    """High-risk contacts' limitation.  Larger values shrink the step faster."""

    sigma_max: float = 1.0 / 6.0
    """Maximum contact radius standard deviation, fixed at 1/6 by the paper."""

    # --- termination -------------------------------------------------------
    max_iterations: int = 500
    """Backstop iteration cap (the paper's third criterion)."""

    mean_gap_tol: float = 1e-10
    """Second criterion: stop once mean(|HCF_i - HCF_best|) falls below this."""

    target_value: float | None = None
    """Optional early stop once this objective value is reached."""

    # --- choices the paper leaves open ------------------------------------
    sigma_formula: SigmaFormula = "corrected"
    """``"corrected"`` follows the prose, ``"literal"`` the typeset Equation 2."""

    init: Literal["normal", "uniform"] = "normal"
    """The paper asks for mean 0.5 and std 0.288; ``"uniform"`` is the alternative."""

    init_std: float = 0.288
    """Standard deviation of the initial normal spread (the paper's value)."""

    bounds_mode: BoundsMode = "clip"
    """How contacts that leave the community are handled."""

    max_transmitters: int | None = 250
    """Cap on the population; the best transmitters survive truncation."""

    on_extinction: Extinction = "stop"
    """``"stop"`` is faithful to the paper; ``"restart"`` begins a new outbreak."""

    on_convergence: Extinction = "stop"
    """Same choice for the mean-gap criterion.  ``"restart"`` keeps searching."""

    restart_from_best: bool = True
    """When restarting, keep the incumbent best as one of the new transmitters."""

    track_positions: bool = False
    """Record every generation's transmitter positions, for the spread figures."""

    seed: int | None = None

    @classmethod
    def paper(cls, **overrides) -> "CDAConfig":
        """The configuration described in the manuscript (Table 1, alpha = 2).

        One outbreak, stopping as soon as it dies out or the community collapses
        onto a single objective value.
        """
        return cls(**{"n_transmitters": 50, "alpha": 2.0, **overrides})

    @classmethod
    def budgeted(cls, **overrides) -> "CDAConfig":
        """A variant that spends a full evaluation budget.

        Equation 2 shrinks the contact radius by a factor of ``alpha`` every day,
        so with ``alpha = 2`` the outbreak freezes after roughly twenty
        iterations and stops with most of its budget unspent.  This preset slows
        the decay and starts a fresh outbreak whenever the previous one dies out
        or converges, which is what makes CDA comparable to a fixed-population
        algorithm over a fixed number of evaluations.

        The defaults come from the sweep in ``experiments/run_sensitivity.py``.
        """
        defaults = {
            "n_transmitters": 20,
            "alpha": 1.15,
            "contact_num_max": 10,
            "on_extinction": "restart",
            "on_convergence": "restart",
            "max_iterations": 100_000,
            "mean_gap_tol": 1e-12,
        }
        return cls(**{**defaults, **overrides})

    def __post_init__(self) -> None:
        if self.n_transmitters < 1:
            raise ValueError("n_transmitters must be at least 1")
        if self.contact_num_max < 1:
            raise ValueError("contact_num_max must be at least 1")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if not 0 < self.sigma_max:
            raise ValueError("sigma_max must be positive")


@dataclass
class CDAResult:
    """Outcome of a CDA run."""

    best_value: float
    best_position: np.ndarray
    """Best position in *real* coordinates."""
    best_position_unit: np.ndarray
    """Best position in normalised coordinates."""
    evaluations: int
    iterations: int
    stop_reason: str
    history: dict[str, list] = field(default_factory=dict)
    """Per-iteration diagnostics: population size, best/mean HCF, sigma range,
    acceptance rate and cumulative evaluations."""


def contact_numbers(values: np.ndarray, contact_num_max: int) -> np.ndarray:
    """Equation 1: how many high-risk contacts each transmitter makes.

    ``r`` runs from 1 for the best (lowest) HCF to 0 for the worst, so the best
    transmitter makes ``contact_num_max`` contacts and the worst makes one.  When
    every transmitter shares the same HCF the ratio is degenerate; the paper does
    not cover that case, and we treat them all as equally best.
    """
    values = np.asarray(values, dtype=float)
    hcf_min = float(values.min())
    hcf_max = float(values.max())
    spread = hcf_min - hcf_max
    if not np.isfinite(spread) or spread == 0.0:
        ratio = np.ones_like(values)
    else:
        ratio = (values - hcf_max) / spread
    ratio = np.clip(ratio, 0.0, 1.0)
    return np.floor(ratio * (contact_num_max - 1)).astype(int) + 1


def contact_sigmas(
    values: np.ndarray,
    iteration: int,
    alpha: float,
    sigma_max: float = 1.0 / 6.0,
    formula: SigmaFormula = "corrected",
) -> np.ndarray:
    """Equation 2: the contact radius standard deviation of each transmitter.

    ``iteration`` is 1-based, matching the manuscript's ``m``.
    """
    values = np.asarray(values, dtype=float)
    hcf_min = float(values.min())
    hcf_max = float(values.max())
    spread = hcf_min - hcf_max
    if not np.isfinite(spread) or spread == 0.0:
        ratio = np.ones_like(values)
    else:
        ratio = (values - hcf_max) / spread
    ratio = np.clip(ratio, 0.0, 1.0)

    sigma_min = sigma_max / (alpha ** (iteration - 1))
    if formula == "corrected":
        return ratio * (sigma_min - sigma_max) + sigma_max
    if formula == "literal":
        return ratio * (sigma_max - sigma_min) + sigma_max
    raise ValueError(f"unknown sigma formula: {formula!r}")


class CDA:
    """The Contagious Diseases Algorithm.

    Example
    -------
    >>> import numpy as np
    >>> from cda.problem import make_problem
    >>> from cda.core import CDA, CDAConfig
    >>> problem = make_problem(lambda x: (x ** 2).sum(1), [(-5, 5)] * 2,
    ...                        optimum_value=0.0, budget=5000)
    >>> result = CDA(CDAConfig(seed=0)).run(problem)
    >>> bool(result.best_value < 1e-6)
    True
    """

    def __init__(self, config: CDAConfig | None = None) -> None:
        self.config = config or CDAConfig()

    # --- internals ---------------------------------------------------------
    def _initial_transmitters(self, rng: np.random.Generator, dim: int) -> np.ndarray:
        cfg = self.config
        if cfg.init == "uniform":
            return rng.random((cfg.n_transmitters, dim))
        # The paper's generator: mean 0.5, std 0.288 (the std of Uniform[0,1]),
        # clipped back into the community.
        points = rng.normal(0.5, cfg.init_std, size=(cfg.n_transmitters, dim))
        return np.clip(points, 0.0, 1.0)

    def _apply_bounds(self, points: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        mode = self.config.bounds_mode
        if mode == "clip":
            return np.clip(points, 0.0, 1.0)
        if mode == "reflect":
            # Fold repeatedly so that even far-out points land inside.
            folded = np.abs(points) % 2.0
            return np.where(folded > 1.0, 2.0 - folded, folded)
        if mode == "resample":
            outside = (points < 0.0) | (points > 1.0)
            if outside.any():
                points = points.copy()
                points[outside] = rng.random(int(outside.sum()))
            return points
        raise ValueError(f"unknown bounds mode: {mode!r}")

    def _spread(
        self,
        transmitters: np.ndarray,
        values: np.ndarray,
        iteration: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate every high-risk contact of the current transmitters.

        Returns the contacted individuals, the index of the transmitter that
        contacted each of them, and the sigma used per transmitter.
        """
        cfg = self.config
        counts = contact_numbers(values, cfg.contact_num_max)
        sigmas = contact_sigmas(
            values, iteration, cfg.alpha, cfg.sigma_max, cfg.sigma_formula
        )

        parent_index = np.repeat(np.arange(len(transmitters)), counts)
        total = int(parent_index.size)
        dim = transmitters.shape[1]

        # A high-risk contact is a vector with a uniformly random direction and a
        # normally distributed radius.  Normalising a Gaussian sample gives a
        # direction uniform on the unit sphere; the signed radius then carries the
        # length, which is equivalent to an unsigned radius with a random side.
        directions = rng.normal(size=(total, dim))
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        directions /= norms
        radii = rng.normal(0.0, sigmas[parent_index])[:, None]

        children = transmitters[parent_index] + radii * directions
        children = self._apply_bounds(children, rng)
        return children, parent_index, sigmas

    # --- driver ------------------------------------------------------------
    def run(self, problem: Problem) -> CDAResult:
        """Run the outbreak on ``problem`` until a termination criterion fires."""
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)
        problem.reset()

        history: dict[str, list] = {
            "iteration": [],
            "day": [],
            "outbreak": [],
            "evaluations": [],
            "population": [],
            "best": [],
            "mean": [],
            "sigma_min": [],
            "sigma_max": [],
            "acceptance_rate": [],
        }
        if cfg.track_positions:
            history["positions"] = []

        transmitters = self._initial_transmitters(rng, problem.dim)
        values = problem.evaluate(transmitters)
        transmitters = transmitters[: len(values)]
        if len(values) == 0:
            raise ValueError("the budget is too small to evaluate the initial population")

        stop_reason = "max_iterations"
        iteration = 0
        # ``day`` is the manuscript's ``m`` in Equation 2.  It counts days within
        # the current outbreak, so a restart brings the contact radius back to
        # its full 1/6 and exploration resumes.
        day = 0
        outbreak = 1

        for iteration in range(1, cfg.max_iterations + 1):
            if problem.exhausted():
                stop_reason = "budget_exhausted"
                break
            if cfg.target_value is not None and problem.best_value <= cfg.target_value:
                stop_reason = "target_reached"
                break

            day += 1
            children, parent_index, sigmas = self._spread(
                transmitters, values, day, rng
            )
            child_values = problem.evaluate(children)
            evaluated = len(child_values)
            children = children[:evaluated]
            parent_index = parent_index[:evaluated]

            # The infection rule: catch the disease iff no healthier than the
            # transmitter.  Ties count as infections, as the paper specifies.
            caught = child_values <= values[parent_index]
            survivors = children[caught]
            survivor_values = child_values[caught]

            finite = values[np.isfinite(values)]
            history["iteration"].append(iteration)
            history["day"].append(day)
            history["outbreak"].append(outbreak)
            history["evaluations"].append(problem.evaluations)
            history["population"].append(int(len(transmitters)))
            history["best"].append(float(values.min()))
            history["mean"].append(float(finite.mean()) if finite.size else float("inf"))
            history["sigma_min"].append(float(sigmas.min()))
            history["sigma_max"].append(float(sigmas.max()))
            history["acceptance_rate"].append(float(caught.mean()) if evaluated else 0.0)
            if cfg.track_positions:
                history["positions"].append(problem.denormalise(transmitters).copy())

            if evaluated == 0:
                stop_reason = "budget_exhausted"
                break

            # Criterion 1: the outbreak dies out because nobody was infected.
            if len(survivors) == 0:
                if cfg.on_extinction == "stop":
                    stop_reason = "extinction"
                    break
                transmitters, values, day, outbreak = self._restart(
                    problem, rng, outbreak
                )
                if len(values) == 0:
                    stop_reason = "budget_exhausted"
                    break
                continue

            # Keep the population bounded; the paper does not discuss growth.
            if cfg.max_transmitters is not None and len(survivors) > cfg.max_transmitters:
                keep = np.argsort(survivor_values, kind="stable")[: cfg.max_transmitters]
                survivors = survivors[keep]
                survivor_values = survivor_values[keep]

            transmitters, values = survivors, survivor_values

            # Criterion 2: the community has collapsed onto one objective value.
            if len(values) > 1:
                gap = float(np.mean(np.abs(values - values.min())))
                if gap < cfg.mean_gap_tol:
                    if cfg.on_convergence == "stop":
                        stop_reason = "mean_gap"
                        break
                    transmitters, values, day, outbreak = self._restart(
                        problem, rng, outbreak
                    )
                    if len(values) == 0:
                        stop_reason = "budget_exhausted"
                        break
        else:
            stop_reason = "max_iterations"

        best_unit = (
            problem.best_position
            if problem.best_position is not None
            else transmitters[int(np.argmin(values))]
        )
        return CDAResult(
            best_value=float(problem.best_value),
            best_position=problem.denormalise(best_unit)[0],
            best_position_unit=np.asarray(best_unit, dtype=float).copy(),
            evaluations=int(problem.evaluations),
            iterations=int(iteration),
            stop_reason=stop_reason,
            history=history,
        )

    def _restart(
        self, problem: Problem, rng: np.random.Generator, outbreak: int
    ) -> tuple[np.ndarray, np.ndarray, int, int]:
        """Seed a fresh outbreak after the previous one died out or converged.

        The day counter returns to zero, which restores the full 1/6 contact
        radius of Equation 2 and lets the search explore again.
        """
        transmitters = self._initial_transmitters(rng, problem.dim)
        if self.config.restart_from_best and problem.best_position is not None:
            transmitters[0] = problem.best_position
        values = problem.evaluate(transmitters)
        return transmitters[: len(values)], values, 0, outbreak + 1


def minimise(
    problem: Problem,
    config: CDAConfig | None = None,
) -> CDAResult:
    """Convenience wrapper: run CDA on ``problem``."""
    return CDA(config).run(problem)
