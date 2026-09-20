"""Contagious Diseases Algorithm: a Python implementation of the 2010 paper.

    A. Mohammadi and M. R. Farmani,
    "Development of a New Evolutionary Algorithm Inspired by Outbreak of
    Contagious Diseases".

Quick start
-----------
>>> from cda import CDA, CDAConfig, make_problem
>>> problem = make_problem(lambda x: (x ** 2).sum(1), [(-5, 5)] * 2,
...                        optimum_value=0.0, budget=5000)
>>> result = CDA(CDAConfig(seed=0)).run(problem)
>>> bool(result.best_value < 1e-6)
True
"""

from .core import CDA, CDAConfig, CDAResult, contact_numbers, contact_sigmas, minimise
from .problem import Problem, make_problem
from .benchmarks import BenchmarkFunction, FUNCTIONS, get, make_suite

__version__ = "1.0.0"

__all__ = [
    "CDA",
    "CDAConfig",
    "CDAResult",
    "Problem",
    "make_problem",
    "minimise",
    "contact_numbers",
    "contact_sigmas",
    "BenchmarkFunction",
    "FUNCTIONS",
    "get",
    "make_suite",
    "__version__",
]
