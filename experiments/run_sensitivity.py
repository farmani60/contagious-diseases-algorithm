#!/usr/bin/env python3
"""How CDA responds to its two tuning parameters.

The manuscript fixes ``alpha`` at 2 and calls tuning it "the subject of future
works", and it never prints a value for ``ContactNum_max`` at all.  This script
sweeps both, plus the population size and the two readings of Equation 2, and
writes the grids the README reports.

Usage:
    python experiments/run_sensitivity.py [--seeds 10]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cda.benchmarks import budget_for, get
from cda.core import CDA, CDAConfig
from cda.runner import RESULTS_DIR, write_csv

PROBLEMS = (
    ("sinc_well", 2), ("ripple_cone", 2), ("sphere", 10),
    ("rastrigin", 10), ("ackley", 10), ("rosenbrock", 10), ("griewank", 10),
)

ALPHAS = (1.05, 1.15, 1.3, 1.5, 2.0, 3.0)
CONTACTS = (2, 3, 5, 10, 20)
POPULATIONS = (10, 20, 50, 100)


def score(config: CDAConfig, seeds: int) -> tuple[float, list[float]]:
    """Mean log10 error across the problem set; lower is better."""
    per_problem = []
    for name, dim in PROBLEMS:
        errors = []
        for seed in range(seeds):
            problem = get(name).to_problem(dim, budget=budget_for(dim))
            CDA(CDAConfig(**{**vars(config), "seed": seed})).run(problem)
            errors.append(problem.error())
        per_problem.append(float(np.median(errors)))
    combined = float(np.mean(np.log10(np.array(per_problem) + 1e-12)))
    return combined, per_problem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args()

    rows: list[dict] = []

    print(f"=== alpha x ContactNum_max (population 20, {args.seeds} seeds) ===")
    print(f"{'alpha':>7}{'contacts':>10}{'score':>9}   " +
          "".join(f"{n[:9]:>11}" for n, _ in PROBLEMS))
    grid = np.full((len(ALPHAS), len(CONTACTS)), np.nan)
    for i, alpha in enumerate(ALPHAS):
        for j, contacts in enumerate(CONTACTS):
            config = CDAConfig.tuned(alpha=alpha, contact_num_max=contacts, n_transmitters=20)
            combined, per_problem = score(config, args.seeds)
            grid[i, j] = combined
            rows.append({
                "sweep": "alpha_x_contacts", "alpha": alpha, "contact_num_max": contacts,
                "n_transmitters": 20, "sigma_formula": "corrected", "score": combined,
                **{f"{n}_{d}d": v for (n, d), v in zip(PROBLEMS, per_problem)},
            })
            print(f"{alpha:>7}{contacts:>10}{combined:>9.3f}   " +
                  "".join(f"{v:>11.4g}" for v in per_problem), flush=True)

    print(f"\n=== population size (alpha 1.15, 10 contacts) ===")
    for population in POPULATIONS:
        config = CDAConfig.tuned(n_transmitters=population)
        combined, per_problem = score(config, args.seeds)
        rows.append({
            "sweep": "population", "alpha": 1.15, "contact_num_max": 10,
            "n_transmitters": population, "sigma_formula": "corrected", "score": combined,
            **{f"{n}_{d}d": v for (n, d), v in zip(PROBLEMS, per_problem)},
        })
        print(f"  N={population:<5} score={combined:>7.3f}   " +
              "".join(f"{v:>11.4g}" for v in per_problem), flush=True)

    print(f"\n=== Equation 2: corrected vs literal ===")
    for formula in ("corrected", "literal"):
        for preset_name, preset in (("original", CDAConfig.original()), ("tuned", CDAConfig.tuned())):
            config = CDAConfig(**{**vars(preset), "sigma_formula": formula})
            combined, per_problem = score(config, args.seeds)
            rows.append({
                "sweep": "sigma_formula", "alpha": config.alpha,
                "contact_num_max": config.contact_num_max,
                "n_transmitters": config.n_transmitters, "sigma_formula": f"{formula}/{preset_name}",
                "score": combined,
                **{f"{n}_{d}d": v for (n, d), v in zip(PROBLEMS, per_problem)},
            })
            print(f"  {formula:<10}/{preset_name:<9} score={combined:>7.3f}   " +
                  "".join(f"{v:>11.4g}" for v in per_problem), flush=True)

    write_csv(rows, RESULTS_DIR / "sensitivity.csv")
    np.save(RESULTS_DIR / "sensitivity_grid.npy", grid)
    best = min(rows, key=lambda r: r["score"])
    print(f"\nbest configuration: alpha={best['alpha']}, "
          f"ContactNum_max={best['contact_num_max']}, N={best['n_transmitters']}, "
          f"score={best['score']:.3f}")
    print(f"written to {RESULTS_DIR / 'sensitivity.csv'}")


if __name__ == "__main__":
    main()
