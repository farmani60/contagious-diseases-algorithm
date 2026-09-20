#!/usr/bin/env python3
"""Does the evaluation budget change who wins?

CDA terminates on its own after a few hundred evaluations, so a study that hands
every algorithm tens of thousands of them could be accused of choosing a regime
that suits the baselines. This script removes that objection by sweeping the
budget from 250 evaluations upward and recording who wins at each size.

Usage:
    python experiments/run_budget_sweep.py [--seeds 30]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cda.baselines import ALGORITHMS
from cda.benchmarks import get
from cda.core import CDA, CDAConfig
from cda.runner import RESULTS_DIR, write_csv

PROBLEMS = (
    ("sinc_well", 2), ("ripple_cone", 2), ("sphere", 10), ("rastrigin", 10),
    ("ackley", 10), ("rosenbrock", 10), ("griewank", 10), ("levy", 10),
)
BUDGETS = (250, 500, 1000, 2000, 5000, 10000)
ALGORITHM_NAMES = ("CDA", "CDA (tuned)", "PSO", "GA", "DE", "Random")


def solve(algorithm: str, function: str, dim: int, budget: int, seed: int) -> float:
    problem = get(function).to_problem(dim, budget=budget)
    if algorithm == "CDA":
        CDA(CDAConfig.original(seed=seed)).run(problem)
    elif algorithm == "CDA (tuned)":
        CDA(CDAConfig.tuned(seed=seed)).run(problem)
    else:
        ALGORITHMS[algorithm](seed=seed).run(problem)
    return problem.error()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=30)
    args = parser.parse_args()

    rows: list[dict] = []
    print(f"{len(PROBLEMS)} problems, {args.seeds} seeds, mean rank over the set "
          f"(1 = best), wins in brackets\n")
    header = f"{'budget':>8}  " + "".join(f"{name:>16}" for name in ALGORITHM_NAMES)
    print(header)
    print("-" * len(header))

    for budget in BUDGETS:
        per_algorithm_ranks: dict[str, list[int]] = {n: [] for n in ALGORITHM_NAMES}
        wins: dict[str, int] = {n: 0 for n in ALGORITHM_NAMES}

        for function, dim in PROBLEMS:
            medians = {
                name: float(np.median([
                    solve(name, function, dim, budget, seed)
                    for seed in range(args.seeds)
                ]))
                for name in ALGORITHM_NAMES
            }
            order = sorted(ALGORITHM_NAMES, key=lambda n: medians[n])
            wins[order[0]] += 1
            for position, name in enumerate(order, start=1):
                per_algorithm_ranks[name].append(position)
                rows.append({
                    "budget": budget, "problem": f"{function}_{dim}d",
                    "algorithm": name, "median_error": medians[name], "rank": position,
                })

        cells = "".join(
            f"{np.mean(per_algorithm_ranks[n]):.2f} ({wins[n]}){'':>7}"[:16]
            for n in ALGORITHM_NAMES
        )
        print(f"{budget:>8}  {cells}", flush=True)

    path = write_csv(rows, RESULTS_DIR / "budget_sweep.csv")
    print(f"\nwritten to {path}")


if __name__ == "__main__":
    main()
