"""Experiment harness: repeated seeded runs, summary statistics and CSV output.

Every algorithm gets the same problem, the same evaluation budget and the same
list of seeds, so differences in the tables come from the algorithms rather than
from the protocol.  All numbers quoted in the README are produced here and
written to ``results/*.csv``.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from .baselines import ALGORITHMS, BaselineResult
from .benchmarks import budget_for, get, make_suite
from .core import CDA, CDAConfig
from .problem import Problem

__all__ = [
    "RunRecord",
    "run_once",
    "run_study",
    "summarise",
    "compare_to_cda",
    "write_csv",
    "DEFAULT_TOLERANCE",
]

DEFAULT_TOLERANCE = 1e-4
"""A run counts as a success when it gets this close to the known optimum."""

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


@dataclass
class RunRecord:
    """One algorithm, one problem, one seed."""

    algorithm: str
    problem: str
    dim: int
    seed: int
    best_value: float
    error: float
    evaluations: int
    iterations: int
    evals_to_target: int | None
    success: bool
    stop_reason: str
    seconds: float


def _builder(name: str, seed: int, cda_config: CDAConfig | None = None) -> Callable[[], object]:
    """Return a factory for the named algorithm, seeded."""
    if name.startswith("CDA"):
        base = cda_config or CDAConfig()
        config = CDAConfig(**{**{k: v for k, v in vars(base).items()}, "seed": seed})
        return lambda: CDA(config)
    return lambda: ALGORITHMS[name](seed=seed)


def run_once(
    algorithm: str,
    problem: Problem,
    seed: int,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    cda_config: CDAConfig | None = None,
) -> RunRecord:
    """Run one algorithm on one problem with one seed."""
    optimiser = _builder(algorithm, seed, cda_config)()
    started = time.perf_counter()
    result = optimiser.run(problem)
    elapsed = time.perf_counter() - started

    error = problem.error()
    return RunRecord(
        algorithm=algorithm,
        problem=problem.name,
        dim=problem.dim,
        seed=seed,
        best_value=float(problem.best_value),
        error=float(error),
        evaluations=int(problem.evaluations),
        iterations=int(getattr(result, "iterations", 0)),
        evals_to_target=problem.evals_to_target(tolerance),
        success=bool(np.isfinite(error) and error <= tolerance),
        stop_reason=getattr(result, "stop_reason", "budget_exhausted"),
        seconds=elapsed,
    )


def run_study(
    problems: Sequence[Problem] | None = None,
    algorithms: Sequence[str] = ("CDA", "PSO", "GA", "DE", "Random"),
    seeds: Iterable[int] = range(30),
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    cda_configs: dict[str, CDAConfig] | None = None,
    verbose: bool = True,
) -> list[RunRecord]:
    """Run every algorithm on every problem for every seed."""
    problems = list(problems) if problems is not None else make_suite()
    seeds = list(seeds)
    records: list[RunRecord] = []

    for problem in problems:
        for algorithm in algorithms:
            config = (cda_configs or {}).get(algorithm)
            started = time.perf_counter()
            for seed in seeds:
                fresh = problem.copy_fresh()
                records.append(
                    run_once(algorithm, fresh, seed, tolerance=tolerance, cda_config=config)
                )
            if verbose:
                recent = [r for r in records[-len(seeds):]]
                best = min(r.best_value for r in recent)
                median = float(np.median([r.best_value for r in recent]))
                hits = sum(r.success for r in recent)
                print(
                    f"  {problem.name:<20s} {algorithm:<10s} "
                    f"best={best: .6g}  median={median: .6g}  "
                    f"success={hits}/{len(seeds)}  "
                    f"[{time.perf_counter() - started:.1f}s]",
                    flush=True,
                )
    return records


def summarise(records: Sequence[RunRecord]) -> list[dict]:
    """Aggregate per (problem, algorithm): best, mean, std, median, success rate."""
    groups: dict[tuple[str, str], list[RunRecord]] = {}
    for record in records:
        groups.setdefault((record.problem, record.algorithm), []).append(record)

    rows: list[dict] = []
    for (problem, algorithm), group in groups.items():
        values = np.array([r.best_value for r in group], dtype=float)
        errors = np.array([r.error for r in group], dtype=float)
        reached = [r.evals_to_target for r in group if r.evals_to_target is not None]
        rows.append(
            {
                "problem": problem,
                "dim": group[0].dim,
                "algorithm": algorithm,
                "runs": len(group),
                "best": float(values.min()),
                "median": float(np.median(values)),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "worst": float(values.max()),
                "mean_error": float(np.nanmean(errors)),
                "success_rate": float(np.mean([r.success for r in group])),
                "median_evals_to_target": float(np.median(reached)) if reached else float("nan"),
                "mean_evaluations": float(np.mean([r.evaluations for r in group])),
                "mean_seconds": float(np.mean([r.seconds for r in group])),
            }
        )
    rows.sort(key=lambda row: (row["problem"], row["algorithm"]))
    return rows


def compare_to_cda(
    records: Sequence[RunRecord], reference: str = "CDA"
) -> list[dict]:
    """Mann-Whitney U test of the reference algorithm against each other one.

    Reports, per problem, whether the difference in final objective values across
    seeds is statistically significant and which algorithm came out ahead.
    """
    from scipy.stats import mannwhitneyu

    by_problem: dict[str, dict[str, list[float]]] = {}
    for record in records:
        by_problem.setdefault(record.problem, {}).setdefault(record.algorithm, []).append(
            record.best_value
        )

    rows: list[dict] = []
    for problem, algorithms in sorted(by_problem.items()):
        if reference not in algorithms:
            continue
        ref = np.array(algorithms[reference], dtype=float)
        for name, values in sorted(algorithms.items()):
            if name == reference:
                continue
            other = np.array(values, dtype=float)
            if np.allclose(ref, other):
                statistic, p_value = float("nan"), 1.0
            else:
                statistic, p_value = mannwhitneyu(ref, other, alternative="two-sided")
            median_ref, median_other = float(np.median(ref)), float(np.median(other))
            if p_value >= 0.05:
                verdict = "tie"
            elif median_ref < median_other:
                verdict = f"{reference} better"
            else:
                verdict = f"{name} better"
            rows.append(
                {
                    "problem": problem,
                    "reference": reference,
                    "other": name,
                    "median_reference": median_ref,
                    "median_other": median_other,
                    "u_statistic": float(statistic),
                    "p_value": float(p_value),
                    "verdict": verdict,
                }
            )
    return rows


def average_ranks(summary: Sequence[dict], key: str = "median") -> list[dict]:
    """Average rank of each algorithm across problems (1 = best)."""
    by_problem: dict[str, list[dict]] = {}
    for row in summary:
        by_problem.setdefault(row["problem"], []).append(row)

    totals: dict[str, list[float]] = {}
    for rows in by_problem.values():
        values = np.array([r[key] for r in rows], dtype=float)
        order = np.argsort(values, kind="stable")
        ranks = np.empty(len(values), dtype=float)
        ranks[order] = np.arange(1, len(values) + 1)
        # Ties share the average of their ranks.
        for value in np.unique(values):
            tied = values == value
            if tied.sum() > 1:
                ranks[tied] = ranks[tied].mean()
        for row, rank in zip(rows, ranks):
            totals.setdefault(row["algorithm"], []).append(float(rank))

    result = [
        {
            "algorithm": name,
            "average_rank": float(np.mean(ranks)),
            "problems": len(ranks),
        }
        for name, ranks in totals.items()
    ]
    result.sort(key=lambda row: row["average_rank"])
    return result


def write_csv(rows: Sequence[dict] | Sequence[RunRecord], path: str | Path) -> Path:
    """Write a list of dicts or dataclasses to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dicts = [asdict(r) if hasattr(r, "__dataclass_fields__") else dict(r) for r in rows]
    if not dicts:
        path.write_text("")
        return path
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dicts[0].keys()))
        writer.writeheader()
        writer.writerows(dicts)
    return path


def write_json(payload: object, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=float))
    return path
