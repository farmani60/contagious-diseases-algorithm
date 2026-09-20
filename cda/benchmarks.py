"""Benchmark objective functions.

Two groups of functions live here:

* ``paper_f1`` and ``paper_f2`` are the two test functions defined in the
  original manuscript, transcribed exactly as printed.  Their global optima
  recorded here were computed directly, by dense grid search plus local
  refinement, not taken from the paper.
* A standard suite of unimodal, valley-shaped and multimodal functions, used to
  measure the algorithm on ground the paper never covered, including higher
  dimensions.

Every function is vectorised: it takes an array of shape ``(k, n)`` and returns
``(k,)``.  Every entry records its own known optimum, which the test suite checks
by evaluating the function at the documented minimiser.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from .problem import Problem

__all__ = [
    "BenchmarkFunction",
    "FUNCTIONS",
    "PAPER_FUNCTIONS",
    "get",
    "make_suite",
    "suite_names",
]


@dataclass(frozen=True)
class BenchmarkFunction:
    """A named objective with known bounds and a known global minimum."""

    name: str
    func: Callable[[np.ndarray], np.ndarray]
    lower: tuple[float, ...] | float
    upper: tuple[float, ...] | float
    optimum_value: float
    optimum_position: tuple[float, ...] | float | None = None
    fixed_dim: int | None = None
    """Set when the function only exists in one dimension."""
    tags: tuple[str, ...] = ()
    description: str = ""

    def bounds(self, dim: int) -> tuple[np.ndarray, np.ndarray]:
        low = np.full(dim, self.lower, dtype=float) if np.isscalar(self.lower) else np.asarray(self.lower, dtype=float)
        high = np.full(dim, self.upper, dtype=float) if np.isscalar(self.upper) else np.asarray(self.upper, dtype=float)
        return low, high

    def optimum(self, dim: int) -> np.ndarray | None:
        if self.optimum_position is None:
            return None
        if np.isscalar(self.optimum_position):
            return np.full(dim, float(self.optimum_position))
        return np.asarray(self.optimum_position, dtype=float)

    def to_problem(self, dim: int | None = None, budget: int | None = None) -> Problem:
        if self.fixed_dim is not None:
            if dim is not None and dim != self.fixed_dim:
                raise ValueError(
                    f"{self.name} is only defined in {self.fixed_dim} dimensions"
                )
            dim = self.fixed_dim
        elif dim is None:
            dim = 2
        low, high = self.bounds(dim)
        name = self.name if self.fixed_dim is not None else f"{self.name}_{dim}d"
        return Problem(
            func=self.func,
            lower=low,
            upper=high,
            name=name,
            optimum_value=self.optimum_value_for(dim),
            optimum_position=self.optimum(dim),
            budget=budget,
            tags=self.tags,
        )

    def optimum_value_for(self, dim: int) -> float:
        # Michalewicz is the only entry whose optimum depends on the dimension.
        if self.name == "michalewicz":
            return _MICHALEWICZ_OPTIMA.get(dim, float("nan"))
        return self.optimum_value


# --------------------------------------------------------------------------
# The two test functions of the manuscript
# --------------------------------------------------------------------------
def paper_f1(x: np.ndarray) -> np.ndarray:
    """Test function 1: a single sharp global basin at (4, 4) ringed by ripples."""
    x = np.atleast_2d(x)
    r = np.sqrt((x[:, 0] - 4.0) ** 2 + (x[:, 1] - 4.0) ** 2)
    return -20.0 * np.sin(0.1 + r) / (0.1 + r)


def paper_f2(x: np.ndarray) -> np.ndarray:
    """Test function 2: many local minima crowded around the global one."""
    x = np.atleast_2d(x)
    x1, x2 = x[:, 0], x[:, 1]
    radial = (x1 ** 2 + x2 ** 2) ** 0.25
    oscillation = np.sin(30.0 * ((x1 + 0.5) ** 2 + x2 ** 2) ** 0.1)
    return radial * oscillation + np.abs(x1) + np.abs(x2)


# --------------------------------------------------------------------------
# Standard suite
# --------------------------------------------------------------------------
def sphere(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(x)
    return np.sum(x ** 2, axis=1)


def step(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(x)
    return np.sum(np.floor(x + 0.5) ** 2, axis=1)


def zakharov(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(x)
    idx = np.arange(1, x.shape[1] + 1)
    partial = np.sum(0.5 * idx * x, axis=1)
    return np.sum(x ** 2, axis=1) + partial ** 2 + partial ** 4


def rosenbrock(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(x)
    head, tail = x[:, :-1], x[:, 1:]
    return np.sum(100.0 * (tail - head ** 2) ** 2 + (head - 1.0) ** 2, axis=1)


def rastrigin(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(x)
    return 10.0 * x.shape[1] + np.sum(x ** 2 - 10.0 * np.cos(2.0 * np.pi * x), axis=1)


def ackley(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(x)
    n = x.shape[1]
    term1 = -20.0 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2, axis=1) / n))
    term2 = -np.exp(np.sum(np.cos(2.0 * np.pi * x), axis=1) / n)
    return term1 + term2 + 20.0 + math.e


def griewank(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(x)
    idx = np.arange(1, x.shape[1] + 1)
    return (
        np.sum(x ** 2, axis=1) / 4000.0
        - np.prod(np.cos(x / np.sqrt(idx)), axis=1)
        + 1.0
    )


def schwefel(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(x)
    return 418.982887272433 * x.shape[1] - np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=1)


def levy(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(x)
    w = 1.0 + (x - 1.0) / 4.0
    first = np.sin(np.pi * w[:, 0]) ** 2
    middle = np.sum(
        (w[:, :-1] - 1.0) ** 2 * (1.0 + 10.0 * np.sin(np.pi * w[:, :-1] + 1.0) ** 2),
        axis=1,
    )
    last = (w[:, -1] - 1.0) ** 2 * (1.0 + np.sin(2.0 * np.pi * w[:, -1]) ** 2)
    return first + middle + last


def michalewicz(x: np.ndarray, steepness: int = 10) -> np.ndarray:
    x = np.atleast_2d(x)
    idx = np.arange(1, x.shape[1] + 1)
    return -np.sum(
        np.sin(x) * np.sin(idx * x ** 2 / np.pi) ** (2 * steepness), axis=1
    )


# Literature reference values (Molga & Smutnicki).  A long differential-evolution
# search of the 10-D case reaches -9.6184, so treat the 10-D success rate as
# "distance to the standard reference value".
_MICHALEWICZ_OPTIMA = {2: -1.8013034101, 5: -4.687658, 10: -9.6601517}


def easom(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(x)
    x1, x2 = x[:, 0], x[:, 1]
    return -np.cos(x1) * np.cos(x2) * np.exp(-((x1 - np.pi) ** 2 + (x2 - np.pi) ** 2))


def six_hump_camel(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(x)
    x1, x2 = x[:, 0], x[:, 1]
    return (
        (4.0 - 2.1 * x1 ** 2 + x1 ** 4 / 3.0) * x1 ** 2
        + x1 * x2
        + (-4.0 + 4.0 * x2 ** 2) * x2 ** 2
    )


def branin(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(x)
    x1, x2 = x[:, 0], x[:, 1]
    a, b = 1.0, 5.1 / (4.0 * np.pi ** 2)
    c, r, s, t = 5.0 / np.pi, 6.0, 10.0, 1.0 / (8.0 * np.pi)
    return a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1.0 - t) * np.cos(x1) + s


def booth(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(x)
    x1, x2 = x[:, 0], x[:, 1]
    return (x1 + 2.0 * x2 - 7.0) ** 2 + (2.0 * x1 + x2 - 5.0) ** 2


def himmelblau(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(x)
    x1, x2 = x[:, 0], x[:, 1]
    return (x1 ** 2 + x2 - 11.0) ** 2 + (x1 + x2 ** 2 - 7.0) ** 2


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------
_ALL: tuple[BenchmarkFunction, ...] = (
    BenchmarkFunction(
        "paper_f1", paper_f1, -5.0, 5.0, -19.96668332936563, (4.0, 4.0), 2,
        ("paper", "multimodal"),
        "Test function 1 of the manuscript: a sharp global basin at (4, 4).",
    ),
    BenchmarkFunction(
        "paper_f2", paper_f2, -5.0, 5.0, -0.24740519403861622,
        (-0.202149941, 0.0), 2, ("paper", "multimodal"),
        "Test function 2 of the manuscript: local minima crowd the global one.",
    ),
    BenchmarkFunction(
        "sphere", sphere, -5.12, 5.12, 0.0, 0.0, None,
        ("unimodal", "separable", "scalable"), "The simplest convex bowl."),
    BenchmarkFunction(
        "step", step, -100.0, 100.0, 0.0, 0.0, None,
        ("unimodal", "separable", "scalable", "discontinuous"),
        "Flat plateaus: gradient information is useless."),
    BenchmarkFunction(
        "zakharov", zakharov, -5.0, 10.0, 0.0, 0.0, None,
        ("unimodal", "scalable"), "Unimodal but strongly coupled between axes."),
    BenchmarkFunction(
        "rosenbrock", rosenbrock, -2.048, 2.048, 0.0, 1.0, None,
        ("valley", "scalable"), "A narrow curved valley; the classic hard case."),
    BenchmarkFunction(
        "rastrigin", rastrigin, -5.12, 5.12, 0.0, 0.0, None,
        ("multimodal", "separable", "scalable"),
        "A regular lattice of deep local minima."),
    BenchmarkFunction(
        "ackley", ackley, -32.768, 32.768, 0.0, 0.0, None,
        ("multimodal", "scalable"),
        "A nearly flat outer region with a narrow central funnel."),
    BenchmarkFunction(
        "griewank", griewank, -600.0, 600.0, 0.0, 0.0, None,
        ("multimodal", "scalable"),
        "Fine ripples on a wide bowl; harder in low dimensions."),
    BenchmarkFunction(
        "schwefel", schwefel, -500.0, 500.0, 0.0, 420.968746, None,
        ("multimodal", "separable", "scalable", "deceptive"),
        "The global minimum sits far from the next best minimum."),
    BenchmarkFunction(
        "levy", levy, -10.0, 10.0, 0.0, 1.0, None,
        ("multimodal", "scalable"), "Many local minima around a single global one."),
    BenchmarkFunction(
        "michalewicz", michalewicz, 0.0, math.pi, float("nan"), None, None,
        ("multimodal", "scalable", "steep"),
        "Steep ridges and flat valleys; the optimum depends on the dimension."),
    BenchmarkFunction(
        "easom", easom, -100.0, 100.0, -1.0, (math.pi, math.pi), 2,
        ("multimodal", "needle"),
        "A single narrow spike in an otherwise flat landscape."),
    BenchmarkFunction(
        "six_hump_camel", six_hump_camel, (-3.0, -2.0), (3.0, 2.0),
        -1.031628453489877, (0.0898420, -0.7126564), 2, ("multimodal",),
        "Six local minima, two of them global."),
    BenchmarkFunction(
        "branin", branin, (-5.0, 0.0), (10.0, 15.0), 0.397887357729739,
        (math.pi, 2.275), 2, ("multimodal",), "Three equal global minima."),
    BenchmarkFunction(
        "booth", booth, -10.0, 10.0, 0.0, (1.0, 3.0), 2, ("unimodal",),
        "A simple quadratic with a tilted valley."),
    BenchmarkFunction(
        "himmelblau", himmelblau, -5.0, 5.0, 0.0, (3.0, 2.0), 2, ("multimodal",),
        "Four identical global minima."),
)

FUNCTIONS: dict[str, BenchmarkFunction] = {f.name: f for f in _ALL}
PAPER_FUNCTIONS: tuple[str, ...] = ("paper_f1", "paper_f2")

#: The standard study: which functions are run at which dimensions.
SUITE: tuple[tuple[str, int], ...] = (
    ("paper_f1", 2),
    ("paper_f2", 2),
    ("easom", 2),
    ("six_hump_camel", 2),
    ("branin", 2),
    ("booth", 2),
    ("himmelblau", 2),
    ("sphere", 2), ("sphere", 10), ("sphere", 30),
    ("step", 10), ("step", 30),
    ("zakharov", 10),
    ("rosenbrock", 2), ("rosenbrock", 10), ("rosenbrock", 30),
    ("rastrigin", 2), ("rastrigin", 10), ("rastrigin", 30),
    ("ackley", 2), ("ackley", 10), ("ackley", 30),
    ("griewank", 10), ("griewank", 30),
    ("schwefel", 10),
    ("levy", 10),
    ("michalewicz", 2), ("michalewicz", 10),
)


def get(name: str) -> BenchmarkFunction:
    """Look a benchmark function up by name."""
    try:
        return FUNCTIONS[name]
    except KeyError:
        raise KeyError(f"unknown benchmark {name!r}; available: {sorted(FUNCTIONS)}") from None


def budget_for(dim: int) -> int:
    """The evaluation budget used for a problem of the given dimension."""
    if dim <= 2:
        return 10_000
    if dim <= 10:
        return 30_000
    return 60_000


def make_suite(entries: Iterable[tuple[str, int]] | None = None) -> list[Problem]:
    """Instantiate the benchmark suite as a list of :class:`Problem` objects."""
    entries = SUITE if entries is None else entries
    return [get(name).to_problem(dim, budget=budget_for(dim)) for name, dim in entries]


def suite_names() -> list[str]:
    return [p.name for p in make_suite()]
