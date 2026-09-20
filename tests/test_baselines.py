"""Tests for the baseline optimisers and the experiment harness."""

from __future__ import annotations

import numpy as np
import pytest

from cda.baselines import ALGORITHMS, DifferentialEvolution, RandomSearch
from cda.benchmarks import get
from cda.core import CDAConfig
from cda.runner import (
    average_ranks,
    compare_to_cda,
    run_once,
    run_study,
    summarise,
    write_csv,
)


ALL = sorted(ALGORITHMS)


@pytest.mark.parametrize("name", ALL)
class TestBaselineContract:
    def test_respects_the_budget(self, name):
        for budget in (100, 1234, 5000):
            problem = get("sphere").to_problem(2, budget=budget)
            result = ALGORITHMS[name](seed=0).run(problem)
            assert result.evaluations <= budget

    def test_stays_inside_the_bounds(self, name):
        seen: list[np.ndarray] = []

        def recorder(x):
            seen.append(np.array(x, copy=True))
            return np.sum(np.atleast_2d(x) ** 2, axis=1)

        problem = get("sphere").to_problem(3, budget=3000)
        problem.func = recorder
        ALGORITHMS[name](seed=0).run(problem)
        points = np.vstack(seen)
        assert points.min() >= problem.lower.min() - 1e-9
        assert points.max() <= problem.upper.max() + 1e-9

    def test_is_deterministic_under_a_seed(self, name):
        p1 = get("rastrigin").to_problem(5, budget=3000)
        p2 = get("rastrigin").to_problem(5, budget=3000)
        assert ALGORITHMS[name](seed=7).run(p1).best_value == pytest.approx(
            ALGORITHMS[name](seed=7).run(p2).best_value
        )

    def test_reported_best_matches_the_problem(self, name):
        problem = get("sphere").to_problem(2, budget=2000)
        result = ALGORITHMS[name](seed=0).run(problem)
        assert result.best_value == problem.best_value
        assert np.allclose(result.best_position, problem.best_real())


@pytest.mark.parametrize("name", ["PSO", "GA", "DE"])
def test_search_algorithms_solve_the_sphere(name):
    problem = get("sphere").to_problem(10, budget=30000)
    ALGORITHMS[name](seed=0).run(problem)
    assert problem.best_value < 1e-3


@pytest.mark.parametrize("name", ["PSO", "GA", "DE"])
def test_search_algorithms_beat_random_search(name):
    searched, sampled = [], []
    for seed in range(3):
        p1 = get("sphere").to_problem(10, budget=10000)
        ALGORITHMS[name](seed=seed).run(p1)
        searched.append(p1.best_value)
        p2 = get("sphere").to_problem(10, budget=10000)
        RandomSearch(seed=seed).run(p2)
        sampled.append(p2.best_value)
    assert np.median(searched) < np.median(sampled)


def test_de_handles_a_population_too_small_to_mutate():
    problem = get("sphere").to_problem(2, budget=200)
    result = DifferentialEvolution(population=3, seed=0).run(problem)
    assert result.evaluations <= 200


class TestRunner:
    def test_run_once_produces_a_complete_record(self):
        problem = get("sphere").to_problem(2, budget=2000)
        record = run_once("CDA", problem, seed=0)
        assert record.algorithm == "CDA"
        assert record.problem == "sphere_2d"
        assert record.dim == 2
        assert record.evaluations <= 2000
        assert record.seconds >= 0.0
        assert record.error == pytest.approx(record.best_value)

    def test_run_once_accepts_a_cda_config(self):
        problem = get("sphere").to_problem(2, budget=2000)
        record = run_once("CDA", problem, seed=0, cda_config=CDAConfig.tuned())
        assert record.evaluations == 2000

    def test_cda_config_seed_is_overridden_per_run(self):
        problem = get("rastrigin").to_problem(2, budget=2000)
        a = run_once("CDA", problem.copy_fresh(), 1, cda_config=CDAConfig.tuned())
        b = run_once("CDA", problem.copy_fresh(), 2, cda_config=CDAConfig.tuned())
        assert a.best_value != b.best_value

    def test_success_flag_tracks_the_tolerance(self):
        problem = get("sphere").to_problem(2, budget=5000)
        record = run_once("DE", problem, seed=0, tolerance=1e-3)
        assert record.success is True
        problem = get("rastrigin").to_problem(30, budget=500)
        record = run_once("Random", problem, seed=0, tolerance=1e-12)
        assert record.success is False

    def test_run_study_covers_every_combination(self):
        problems = [get("sphere").to_problem(2, budget=800)]
        records = run_study(problems, algorithms=("CDA", "DE"), seeds=range(3), verbose=False)
        assert len(records) == 6
        assert {r.algorithm for r in records} == {"CDA", "DE"}
        assert {r.seed for r in records} == {0, 1, 2}

    def test_runs_are_independent(self):
        problems = [get("sphere").to_problem(2, budget=800)]
        records = run_study(problems, algorithms=("DE",), seeds=range(3), verbose=False)
        assert all(r.evaluations <= 800 for r in records)

    def test_summarise_reports_the_expected_statistics(self):
        problems = [get("sphere").to_problem(2, budget=800)]
        records = run_study(problems, algorithms=("CDA", "DE"), seeds=range(4), verbose=False)
        rows = summarise(records)
        assert len(rows) == 2
        for row in rows:
            assert row["runs"] == 4
            assert row["best"] <= row["median"] <= row["worst"]
            assert 0.0 <= row["success_rate"] <= 1.0
            assert row["std"] >= 0.0

    def test_compare_to_cda_reports_a_verdict(self):
        problems = [get("sphere").to_problem(5, budget=4000)]
        records = run_study(problems, algorithms=("CDA", "DE", "Random"), seeds=range(6), verbose=False)
        rows = compare_to_cda(records)
        assert {r["other"] for r in rows} == {"DE", "Random"}
        for row in rows:
            assert 0.0 <= row["p_value"] <= 1.0
            assert row["verdict"] in {"tie", "CDA better", "DE better", "Random better"}
        beat_random = next(r for r in rows if r["other"] == "Random")
        assert beat_random["verdict"] == "CDA better"

    def test_average_ranks_orders_algorithms(self):
        problems = [get("sphere").to_problem(5, budget=4000)]
        records = run_study(problems, algorithms=("DE", "Random"), seeds=range(4), verbose=False)
        ranks = average_ranks(summarise(records))
        assert ranks[0]["algorithm"] == "DE"
        assert ranks[0]["average_rank"] < ranks[-1]["average_rank"]

    def test_write_csv_round_trips(self, tmp_path):
        import csv

        problems = [get("sphere").to_problem(2, budget=500)]
        records = run_study(problems, algorithms=("DE",), seeds=range(2), verbose=False)
        path = write_csv(records, tmp_path / "out.csv")
        rows = list(csv.DictReader(path.open()))
        assert len(rows) == 2
        assert rows[0]["algorithm"] == "DE"

    def test_write_csv_handles_an_empty_table(self, tmp_path):
        path = write_csv([], tmp_path / "empty.csv")
        assert path.read_text() == ""
