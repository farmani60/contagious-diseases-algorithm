#!/usr/bin/env python3
"""Render the README's result tables from the committed CSV files.

Every number the README quotes comes from ``results/*.csv``, which these
functions read.  Nothing is typed in by hand.

Usage:
    python experiments/make_readme_tables.py > results/tables.md
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cda.runner import RESULTS_DIR

ORDER = ["CDA", "CDA (tuned)", "PSO", "GA", "DE", "Random"]


def read(name: str) -> list[dict]:
    path = RESULTS_DIR / name
    if not path.exists():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def number(value: str | float, digits: int = 4) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "-"
    if value != value:
        return "-"
    if value == 0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e5:
        return f"{value:.{max(digits - 2, 0)}e}"
    text = f"{value:.{digits}f}"
    # Only trim trailing zeros that sit after a decimal point.  Stripping them
    # from an integer would turn 2000 into 2.
    return text.rstrip("0").rstrip(".") if "." in text else text


def table(header: list[str], rows: list[list[str]], align: str | None = None) -> str:
    align = align or ("l" + "r" * (len(header) - 1))
    separator = ["---" if a == "l" else "---:" for a in align]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(separator) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def radial_tables() -> str:
    rows = read("radial_functions.csv")
    if not rows:
        return "_Run `experiments/run_radial_functions.py` first._"

    true_optimum = {"sinc_well": "-19.966683", "ripple_cone": "-0.247405"}

    blocks = []
    for function in ("sinc_well", "ripple_cone"):
        subset = [r for r in rows if r["function"] == function]
        subset.sort(key=lambda r: ORDER.index(r["algorithm"]) if r["algorithm"] in ORDER else 99)
        body = []
        for row in subset:
            name = row["algorithm"]
            body.append([
                name,
                number(row["best_value"], 6),
                number(row["median_value"], 6),
                f"({number(row['x1'], 4)}, {number(row['x2'], 4)})",
                f"{float(row['hit_rate']):.0%}",
                number(row["median_evals_to_global"], 0),
            ])
        blocks.append(
            f"**{function}** (true global optimum {true_optimum[function]})\n\n"
            + table(
                ["algorithm", "best", "median", "best solution", "hit rate",
                 "median evals to optimum"],
                body,
            )
        )
    return "\n\n".join(blocks)


def rank_table() -> str:
    path = RESULTS_DIR / "study_meta.json"
    if not path.exists():
        return "_Run `experiments/run_benchmark_suite.py` first._"
    meta = json.loads(path.read_text())
    body = [
        [row["algorithm"], f"{row['average_rank']:.2f}", str(row["problems"])]
        for row in meta["average_ranks"]
    ]
    return table(["algorithm", "average rank (1 = best)", "problems"], body)


def summary_table(selection: list[str] | None = None) -> str:
    rows = read("summary.csv")
    if not rows:
        return "_Run `experiments/run_benchmark_suite.py` first._"
    problems = selection or sorted({r["problem"] for r in rows})
    present = [a for a in ORDER if a in {r["algorithm"] for r in rows}]
    body = []
    for problem in problems:
        subset = {r["algorithm"]: r for r in rows if r["problem"] == problem}
        if not subset:
            continue
        medians = {
            name: float(row["median"]) for name, row in subset.items()
        }
        winner = min(medians, key=medians.get)
        line = [problem]
        for name in present:
            row = subset.get(name)
            if row is None:
                line.append("-")
                continue
            cell = number(row["median"], 4)
            if name == winner:
                cell = f"**{cell}**"
            line.append(cell)
        body.append(line)
    return table(["problem (median final value)"] + present, body)


def success_table() -> str:
    rows = read("summary.csv")
    if not rows:
        return ""
    problems = sorted({r["problem"] for r in rows})
    present = [a for a in ORDER if a in {r["algorithm"] for r in rows}]
    body = []
    for problem in problems:
        subset = {r["algorithm"]: r for r in rows if r["problem"] == problem}
        line = [problem]
        for name in present:
            row = subset.get(name)
            line.append("-" if row is None else f"{float(row['success_rate']):.0%}")
        body.append(line)
    return table(["problem (success rate)"] + present, body)


def significance_table() -> str:
    rows = read("significance.csv")
    if not rows:
        return "_Run `experiments/run_benchmark_suite.py` first._"
    tally: dict[tuple[str, str], list[int]] = {}
    for row in rows:
        counts = tally.setdefault((row["reference"], row["other"]), [0, 0, 0])
        if row["verdict"] == "tie":
            counts[1] += 1
        elif row["verdict"].startswith(row["reference"]):
            counts[0] += 1
        else:
            counts[2] += 1
    body = []
    for (reference, other), (wins, ties, losses) in sorted(tally.items()):
        if other.startswith("CDA"):
            continue
        body.append([reference, other, str(wins), str(ties), str(losses)])
    return table(["CDA variant", "versus", "wins", "ties", "losses"], body)


def budget_table() -> str:
    """Mean rank and wins per algorithm at each evaluation budget."""
    rows = read("budget_sweep.csv")
    if not rows:
        return "_Run `experiments/run_budget_sweep.py` first._"
    budgets = sorted({int(r["budget"]) for r in rows})
    present = [a for a in ORDER if a in {r["algorithm"] for r in rows}]
    body = []
    for budget in budgets:
        subset = [r for r in rows if int(r["budget"]) == budget]
        problems = len({r["problem"] for r in subset})
        line = [f"{budget:,}"]
        for name in present:
            ranks = [int(r["rank"]) for r in subset if r["algorithm"] == name]
            wins = sum(1 for r in ranks if r == 1)
            line.append(f"{sum(ranks) / len(ranks):.2f} ({wins})" if ranks else "-")
        body.append(line)
    note = (
        f"\nMean rank over {problems} problems, with problems won in brackets. "
        "1.00 would mean winning every problem.\n"
    )
    return table(["evaluation budget"] + present, body) + "\n" + note


def sensitivity_tables() -> str:
    rows = read("sensitivity.csv")
    if not rows:
        return "_Run `experiments/run_sensitivity.py` first._"

    formula = [r for r in rows if r["sweep"] == "sigma_formula"]
    body = [
        [r["sigma_formula"], number(r["score"], 3),
         number(r.get("rastrigin_10d", "nan"), 3), number(r.get("ackley_10d", "nan"), 3),
         number(r.get("sphere_10d", "nan"), 3)]
        for r in formula
    ]
    formula_table = table(
        ["Equation 2 reading / preset", "score (lower is better)",
         "rastrigin 10D", "ackley 10D", "sphere 10D"],
        body,
    )

    population = [r for r in rows if r["sweep"] == "population"]
    population_table = table(
        ["transmitters", "score (lower is better)"],
        [[r["n_transmitters"], number(r["score"], 3)] for r in population],
    )

    grid = [r for r in rows if r["sweep"] == "alpha_x_contacts"]
    best = min(grid, key=lambda r: float(r["score"])) if grid else None
    note = (
        f"\nBest setting in the sweep: `alpha = {best['alpha']}`, "
        f"`ContactNum_max = {best['contact_num_max']}`, "
        f"score {number(best['score'], 3)}.\n"
        if best else ""
    )
    return f"{formula_table}\n\n{population_table}\n{note}"


def main() -> None:
    print("<!-- Generated by experiments/make_readme_tables.py -->\n")
    print("## RADIAL_FUNCTIONS\n")
    print(radial_tables())
    print("\n## RANKS\n")
    print(rank_table())
    print("\n## SUMMARY\n")
    print(summary_table())
    print("\n## SUCCESS\n")
    print(success_table())
    print("\n## SIGNIFICANCE\n")
    print(significance_table())
    print("\n## BUDGET\n")
    print(budget_table())
    print("\n## SENSITIVITY\n")
    print(sensitivity_tables())


if __name__ == "__main__":
    main()
