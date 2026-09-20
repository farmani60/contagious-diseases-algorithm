"""Tests for the CDA optimiser itself: its rules, invariants and termination."""

from __future__ import annotations

import numpy as np
import pytest

from cda.benchmarks import get
from cda.core import CDA, CDAConfig
from cda.problem import make_problem


def sphere(x: np.ndarray) -> np.ndarray:
    return np.sum(np.atleast_2d(x) ** 2, axis=1)


def make(budget: int = 5000, dim: int = 2, bounds=(-5.0, 5.0)):
    return make_problem(sphere, [bounds] * dim, name="sphere", optimum_value=0.0, budget=budget)


class TestInfectionRule:
    """A contact catches the disease only if it is no healthier than its source."""

    def test_no_child_worse_than_its_parent_ever_survives(self):
        # Track the population directly: every surviving transmitter in a
        # generation must be at least as good as the best of the previous one is
        # not guaranteed, but the population minimum must never worsen within an
        # outbreak, because each survivor improves on its own parent.
        problem = get("rosenbrock").to_problem(5, budget=20000)
        result = CDA(CDAConfig(seed=3, on_extinction="stop")).run(problem)
        best = result.history["best"]
        assert np.all(np.diff(best) <= 1e-12), "population best must never worsen"

    def test_equal_objective_counts_as_infection(self):
        # On a flat landscape every contact ties with its parent, so the outbreak
        # must never die out and the population must survive.
        flat = make_problem(
            lambda x: np.zeros(len(np.atleast_2d(x))), [(-1.0, 1.0)] * 2, budget=3000
        )
        result = CDA(CDAConfig(seed=0, on_extinction="stop")).run(flat)
        assert result.stop_reason != "extinction"
        assert all(rate == 1.0 for rate in result.history["acceptance_rate"])

    def test_acceptance_rate_is_a_fraction(self):
        result = CDA(CDAConfig(seed=0)).run(make())
        rates = np.array(result.history["acceptance_rate"])
        assert np.all((rates >= 0.0) & (rates <= 1.0))


class TestInvariants:
    def test_every_evaluated_point_stays_inside_the_bounds(self):
        seen: list[np.ndarray] = []

        def recorder(x):
            seen.append(np.array(x, copy=True))
            return sphere(x)

        problem = make_problem(recorder, [(-5.0, 5.0)] * 3, budget=4000)
        CDA(CDAConfig(seed=1)).run(problem)
        points = np.vstack(seen)
        assert points.min() >= -5.0 - 1e-12
        assert points.max() <= 5.0 + 1e-12

    @pytest.mark.parametrize("mode", ["clip", "reflect", "resample"])
    def test_all_bounds_modes_stay_inside_the_unit_cube(self, mode):
        seen: list[np.ndarray] = []

        def recorder(x):
            seen.append(np.array(x, copy=True))
            return sphere(x)

        problem = make_problem(recorder, [(0.0, 1.0)] * 2, budget=3000)
        CDA(CDAConfig(seed=2, bounds_mode=mode)).run(problem)
        points = np.vstack(seen)
        assert points.min() >= -1e-12 and points.max() <= 1.0 + 1e-12

    def test_unknown_bounds_mode_is_rejected(self):
        with pytest.raises(ValueError, match="unknown bounds mode"):
            CDA(CDAConfig(seed=0, bounds_mode="teleport")).run(make())

    def test_budget_is_never_exceeded(self):
        for budget in (60, 137, 1000, 5000):
            problem = make(budget=budget)
            result = CDA(CDAConfig(seed=0)).run(problem)
            assert result.evaluations <= budget

    def test_population_respects_the_cap(self):
        # The cap truncates each new generation of infected individuals.  The
        # seeded first generation is set by n_transmitters, so it is exempt.
        problem = get("sphere").to_problem(2, budget=20000)
        result = CDA(
            CDAConfig(seed=0, n_transmitters=50, max_transmitters=30, contact_num_max=10)
        ).run(problem)
        populations = result.history["population"]
        assert populations[0] == 50
        assert max(populations[1:]) <= 30

    def test_reported_best_matches_the_problem(self):
        problem = make()
        result = CDA(CDAConfig(seed=5)).run(problem)
        assert result.best_value == problem.best_value
        assert np.allclose(result.best_position, problem.best_real())
        assert float(sphere(result.best_position[None, :])[0]) == pytest.approx(
            result.best_value
        )

    def test_a_tiny_budget_is_handled(self):
        problem = make(budget=10)
        result = CDA(CDAConfig(seed=0, n_transmitters=50)).run(problem)
        assert result.evaluations == 10

    def test_a_budget_below_one_evaluation_is_rejected(self):
        problem = make(budget=0)
        with pytest.raises(ValueError, match="budget is too small"):
            CDA(CDAConfig(seed=0)).run(problem)


class TestDeterminism:
    def test_the_same_seed_reproduces_the_run_exactly(self):
        first = CDA(CDAConfig(seed=42)).run(make())
        second = CDA(CDAConfig(seed=42)).run(make())
        assert first.best_value == second.best_value
        assert np.array_equal(first.best_position, second.best_position)
        assert first.evaluations == second.evaluations
        assert first.history["best"] == second.history["best"]

    def test_different_seeds_give_different_runs(self):
        values = {CDA(CDAConfig(seed=s)).run(make()).best_value for s in range(5)}
        assert len(values) > 1


class TestTermination:
    def test_extinction_stops_the_run_by_default(self):
        # A needle-in-a-haystack landscape: contacts almost never improve.
        problem = get("easom").to_problem(budget=50000)
        result = CDA(CDAConfig.original(seed=1)).run(problem)
        assert result.stop_reason in {"extinction", "mean_gap"}
        assert result.evaluations < problem.budget

    def test_restart_spends_the_whole_budget(self):
        problem = get("easom").to_problem(budget=20000)
        result = CDA(CDAConfig.tuned(seed=1)).run(problem)
        assert result.stop_reason == "budget_exhausted"
        assert result.evaluations == 20000
        assert max(result.history["outbreak"]) > 1

    def test_restart_resets_the_contact_radius(self):
        problem = get("rastrigin").to_problem(10, budget=20000)
        result = CDA(CDAConfig.tuned(seed=0)).run(problem)
        days = np.array(result.history["day"])
        outbreaks = np.array(result.history["outbreak"])
        if outbreaks.max() > 1:
            # Each new outbreak starts again at day 1.
            assert days[np.argmax(outbreaks > 1)] == 1

    def test_iteration_cap_is_honoured(self):
        problem = make(budget=10**7)
        result = CDA(CDAConfig(seed=0, max_iterations=5, mean_gap_tol=0.0)).run(problem)
        assert result.iterations <= 5

    def test_target_value_stops_early(self):
        problem = make(budget=10**6)
        result = CDA(CDAConfig(seed=0, target_value=1.0, max_iterations=10**5)).run(problem)
        assert result.stop_reason == "target_reached"
        assert result.best_value <= 1.0

    def test_mean_gap_criterion_fires_on_a_flat_landscape(self):
        flat = make_problem(
            lambda x: np.zeros(len(np.atleast_2d(x))), [(-1.0, 1.0)] * 2, budget=10000
        )
        result = CDA(CDAConfig(seed=0, mean_gap_tol=1e-9, on_convergence="stop")).run(flat)
        assert result.stop_reason == "mean_gap"

    def test_history_is_consistent(self):
        result = CDA(CDAConfig(seed=0)).run(make())
        lengths = {len(v) for v in result.history.values()}
        assert len(lengths) == 1
        assert np.all(np.diff(result.history["evaluations"]) > 0)


class TestOptimisation:
    @pytest.mark.parametrize("dim", [2, 5, 10])
    def test_solves_the_sphere(self, dim):
        problem = get("sphere").to_problem(dim, budget=30000)
        result = CDA(CDAConfig.tuned(seed=0)).run(problem)
        assert result.best_value < 0.5

    def test_finds_the_optimum_of_the_sinc_well(self):
        problem = get("sinc_well").to_problem(budget=10000)
        CDA(CDAConfig.original(seed=0)).run(problem)
        assert problem.error() < 1e-3
        assert np.allclose(problem.best_real(), [4.0, 4.0], atol=1e-2)

    def test_beats_random_search_on_the_sphere(self):
        from cda.baselines import RandomSearch

        cda_errors, random_errors = [], []
        for seed in range(5):
            p1 = get("sphere").to_problem(10, budget=20000)
            CDA(CDAConfig.tuned(seed=seed)).run(p1)
            cda_errors.append(p1.best_value)
            p2 = get("sphere").to_problem(10, budget=20000)
            RandomSearch(seed=seed).run(p2)
            random_errors.append(p2.best_value)
        assert np.median(cda_errors) < np.median(random_errors)

    def test_corrected_formula_beats_the_literal_one(self):
        """The typeset Equation 2 grows the best transmitter's step, which hurts."""
        corrected, literal = [], []
        for seed in range(8):
            p1 = get("sphere").to_problem(2, budget=5000)
            CDA(CDAConfig(seed=seed, sigma_formula="corrected")).run(p1)
            corrected.append(p1.best_value)
            p2 = get("sphere").to_problem(2, budget=5000)
            CDA(CDAConfig(seed=seed, sigma_formula="literal")).run(p2)
            literal.append(p2.best_value)
        assert np.median(corrected) < np.median(literal)


class TestInitialisation:
    def test_normal_initialisation_uses_the_papers_spread(self):
        seen = []

        def recorder(x):
            seen.append(np.array(x, copy=True))
            return sphere(x)

        problem = make_problem(recorder, [(0.0, 1.0)] * 2, budget=400)
        CDA(CDAConfig(seed=0, n_transmitters=400, init="normal")).run(problem)
        first = seen[0]
        assert first.mean() == pytest.approx(0.5, abs=0.05)
        assert first.std() == pytest.approx(0.288, abs=0.06)

    @pytest.mark.parametrize("init", ["normal", "uniform"])
    def test_both_initialisers_stay_in_range(self, init):
        seen = []

        def recorder(x):
            seen.append(np.array(x, copy=True))
            return sphere(x)

        problem = make_problem(recorder, [(0.0, 1.0)] * 2, budget=200)
        CDA(CDAConfig(seed=0, n_transmitters=200, init=init)).run(problem)
        assert seen[0].min() >= 0.0 and seen[0].max() <= 1.0
