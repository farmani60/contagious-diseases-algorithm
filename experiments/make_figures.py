#!/usr/bin/env python3
"""Generate every figure referenced by the README.

Usage:
    python experiments/make_figures.py [--seeds 15]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cda.baselines import ALGORITHMS
from cda.benchmarks import budget_for, get
from cda.core import CDA, CDAConfig
from cda.plotting import (
    plot_convergence,
    plot_equations,
    plot_mechanism,
    plot_one_day,
    plot_population,
    plot_sensitivity,
    plot_spread_snapshots,
    plot_summary_box,
    plot_surface,
)
from cda.runner import RESULTS_DIR

FIGURES = RESULTS_DIR / "figures"

VARIANTS = {
    "CDA": lambda seed: CDA(CDAConfig.original(seed=seed)),
    "CDA (tuned)": lambda seed: CDA(CDAConfig.tuned(seed=seed)),
    "PSO": lambda seed: ALGORITHMS["PSO"](seed=seed),
    "GA": lambda seed: ALGORITHMS["GA"](seed=seed),
    "DE": lambda seed: ALGORITHMS["DE"](seed=seed),
    "Random": lambda seed: ALGORITHMS["Random"](seed=seed),
}

CONVERGENCE_PANELS = (
    ("sinc_well", 2), ("ripple_cone", 2), ("sphere", 10),
    ("rastrigin", 10), ("ackley", 10), ("rosenbrock", 10),
)


def collect(function_name: str, dim: int, seeds: int):
    """Run every algorithm and return its per-seed best-so-far traces."""
    function = get(function_name)
    budget = budget_for(dim)
    traces: dict[str, list] = {}
    finals: dict[str, list[float]] = {}
    for label, build in VARIANTS.items():
        runs, errors = [], []
        for seed in range(seeds):
            problem = function.to_problem(dim, budget=budget)
            build(seed).run(problem)
            runs.append((list(problem.history_evals), list(problem.history_best)))
            errors.append(problem.error())
        traces[label] = runs
        finals[label] = errors
    return traces, finals, function.optimum_value_for(dim)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=15)
    args = parser.parse_args()
    FIGURES.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    print("how the algorithm works...")
    written.append(plot_mechanism(FIGURES / "mechanism.png"))
    written.append(plot_equations(FIGURES / "equations.png"))
    written.append(plot_one_day(FIGURES / "one_day.png"))

    print("landscapes...")
    for name in ("sinc_well", "ripple_cone"):
        written.append(plot_surface(name, FIGURES / f"landscape_{name}.png"))

    print("how the outbreak spreads...")
    written.append(plot_spread_snapshots(FIGURES / "spread_ripple_cone.png", "ripple_cone"))

    print("population dynamics...")
    written.append(plot_population(FIGURES / "population_dynamics.png"))

    print("convergence...")
    box_data: dict[str, dict[str, list[float]]] = {}
    for name, dim in CONVERGENCE_PANELS:
        traces, finals, optimum = collect(name, dim, args.seeds)
        label = name if dim == 2 else f"{name} ({dim}D)"
        written.append(
            plot_convergence(
                traces,
                FIGURES / f"convergence_{name}_{dim}d.png",
                title=f"Convergence on {label}, median of {args.seeds} runs",
                optimum=optimum,
            )
        )
        box_data[label] = finals
        print(f"  {label}")

    print("summary box plots...")
    written.append(
        plot_summary_box(
            box_data, FIGURES / "summary_box.png",
            title=f"Final error across {args.seeds} runs (log scale, lower is better)",
        )
    )

    grid_path = RESULTS_DIR / "sensitivity_grid.npy"
    if grid_path.exists():
        print("sensitivity heatmap...")
        written.append(
            plot_sensitivity(
                np.load(grid_path),
                alphas=(1.05, 1.15, 1.3, 1.5, 2.0, 3.0),
                contacts=(2, 3, 5, 10, 20),
                path=FIGURES / "sensitivity.png",
                title="CDA quality over its two tuning parameters (lower is better)",
                label="mean log10 error across 7 problems",
            )
        )
    else:
        print("sensitivity heatmap skipped: run experiments/run_sensitivity.py first")

    print(f"\n{len(written)} figures written to {FIGURES}")
    for path in written:
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
