#!/usr/bin/env python3
"""Run every algorithm on the two test functions defined in the manuscript.

For each algorithm this reports the best objective value found, the position of
that solution, the number of evaluations needed to first reach the global
optimum, and the number of evaluations after which the best value stopped
improving.  All figures are measured here; nothing is taken from the paper.

Usage:
    python experiments/run_paper_functions.py [--seeds 30] [--budget 10000]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cda.baselines import ALGORITHMS
from cda.benchmarks import PAPER_FUNCTIONS, get
from cda.core import CDA, CDAConfig
from cda.runner import RESULTS_DIR, write_csv

TOLERANCE = 1e-3
"""How close to the true optimum counts as "found the global optimum"."""


def evaluations_until_settled(problem, epsilon: float = 1e-10) -> int:
    """Evaluations after which the best-so-far value stopped improving."""
    best = np.asarray(problem.history_best, dtype=float)
    evals = np.asarray(problem.history_evals, dtype=int)
    improvements = np.flatnonzero(np.diff(best, prepend=np.inf) < -epsilon)
    return int(evals[improvements[-1]]) if improvements.size else int(evals[0])


def run(function_name: str, seeds: int, budget: int) -> list[dict]:
    function = get(function_name)
    true_optimum = function.optimum_value

    variants = {
        "CDA": lambda seed: CDA(CDAConfig.paper(seed=seed)),
        "CDA (budgeted)": lambda seed: CDA(CDAConfig.budgeted(seed=seed)),
        **{name: (lambda seed, c=cls: c(seed=seed)) for name, cls in ALGORITHMS.items()},
    }

    rows: list[dict] = []
    for label, build in variants.items():
        records = []
        for seed in range(seeds):
            problem = function.to_problem(budget=budget)
            build(seed).run(problem)
            records.append(
                dict(
                    value=problem.best_value,
                    position=problem.best_real(),
                    found=problem.evals_to_target(TOLERANCE),
                    converged=evaluations_until_settled(problem),
                    evaluations=problem.evaluations,
                )
            )

        values = np.array([r["value"] for r in records])
        champion = records[int(np.argmin(values))]
        found = [r["found"] for r in records if r["found"] is not None]
        rows.append(
            {
                "function": function_name,
                "algorithm": label,
                "best_value": float(values.min()),
                "median_value": float(np.median(values)),
                "x1": float(champion["position"][0]),
                "x2": float(champion["position"][1]),
                "evals_to_global_best_run": champion["found"],
                "median_evals_to_global": float(np.median(found)) if found else float("nan"),
                "hit_rate": len(found) / len(records),
                "evals_to_convergence_best_run": champion["converged"],
                "median_evaluations_used": float(np.median([r["evaluations"] for r in records])),
                "true_optimum": true_optimum,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--budget", type=int, default=10_000)
    args = parser.parse_args()

    all_rows: list[dict] = []
    for function_name in PAPER_FUNCTIONS:
        function = get(function_name)
        print(f"\n=== {function_name} ===")
        print(f"true global optimum: {function.optimum_value:.10f} "
              f"at {np.round(function.optimum(2), 6).tolist()}")
        rows = run(function_name, args.seeds, args.budget)
        all_rows.extend(rows)

        header = f"{'algorithm':<16}{'best':>13}{'median':>13}{'x1':>11}{'x2':>11}{'to global':>11}{'hit rate':>10}"
        print(header)
        print("-" * len(header))
        for row in rows:
            found = row["evals_to_global_best_run"]
            print(
                f"{row['algorithm']:<16}{row['best_value']:>13.6f}{row['median_value']:>13.6f}"
                f"{row['x1']:>11.4f}{row['x2']:>11.4f}"
                f"{'-' if found is None else found:>11}{row['hit_rate']:>10.0%}"
            )
    path = write_csv(all_rows, RESULTS_DIR / "paper_functions.csv")
    print(f"\nwritten to {path}")


if __name__ == "__main__":
    main()
