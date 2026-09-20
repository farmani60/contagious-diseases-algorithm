#!/usr/bin/env python3
"""The full benchmark study: every algorithm, every problem, 30 seeds each.

Writes three files to ``results/``:

* ``runs.csv``      - one row per (algorithm, problem, seed)
* ``summary.csv``   - best / median / mean / std / success rate per pair
* ``significance.csv`` - Mann-Whitney U tests of CDA against each baseline

Usage:
    python experiments/run_benchmark_suite.py [--seeds 30] [--quick]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cda.benchmarks import SUITE, make_suite
from cda.core import CDAConfig
from cda.runner import (
    RESULTS_DIR,
    average_ranks,
    compare_to_cda,
    run_study,
    summarise,
    write_csv,
    write_json,
)

ALGORITHMS = ("CDA", "CDA (tuned)", "PSO", "GA", "DE", "Random")

CDA_CONFIGS = {
    "CDA": CDAConfig.original(),
    "CDA (tuned)": CDAConfig.tuned(),
}

QUICK_SUITE = (
    ("sinc_well", 2), ("ripple_cone", 2), ("sphere", 10),
    ("rastrigin", 10), ("ackley", 10), ("rosenbrock", 10),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--quick", action="store_true", help="a six-problem subset")
    args = parser.parse_args()

    problems = make_suite(QUICK_SUITE if args.quick else SUITE)
    print(f"{len(problems)} problems x {len(ALGORITHMS)} algorithms x {args.seeds} seeds "
          f"= {len(problems) * len(ALGORITHMS) * args.seeds} runs\n")

    started = time.perf_counter()
    records = run_study(
        problems,
        algorithms=ALGORITHMS,
        seeds=range(args.seeds),
        cda_configs=CDA_CONFIGS,
        verbose=True,
    )
    elapsed = time.perf_counter() - started

    summary = summarise(records)
    significance = compare_to_cda(records, reference="CDA")
    significance += compare_to_cda(records, reference="CDA (tuned)")
    ranks = average_ranks(summary)

    write_csv(records, RESULTS_DIR / "runs.csv")
    write_csv(summary, RESULTS_DIR / "summary.csv")
    write_csv(significance, RESULTS_DIR / "significance.csv")
    write_json(
        {
            "seeds": args.seeds,
            "problems": len(problems),
            "algorithms": list(ALGORITHMS),
            "average_ranks": ranks,
            "total_seconds": elapsed,
        },
        RESULTS_DIR / "study_meta.json",
    )

    print(f"\n=== average rank across {len(problems)} problems (1 = best median) ===")
    for row in ranks:
        print(f"  {row['algorithm']:<18}{row['average_rank']:.2f}")

    print("\n=== CDA (tuned) against each baseline ===")
    tally: dict[str, list[int]] = {}
    for row in significance:
        if row["reference"] != "CDA (tuned)":
            continue
        counts = tally.setdefault(row["other"], [0, 0, 0])
        if row["verdict"] == "tie":
            counts[1] += 1
        elif row["verdict"].startswith("CDA"):
            counts[0] += 1
        else:
            counts[2] += 1
    for other, (wins, ties, losses) in sorted(tally.items()):
        print(f"  vs {other:<10} {wins} wins / {ties} ties / {losses} losses")

    print(f"\ncompleted in {elapsed / 60:.1f} min; results written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
