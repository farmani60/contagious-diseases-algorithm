"""Tests for the Problem wrapper: normalisation, budget and bookkeeping."""

from __future__ import annotations

import numpy as np
import pytest

from cda.problem import Problem, make_problem


def sphere(x: np.ndarray) -> np.ndarray:
    return np.sum(np.atleast_2d(x) ** 2, axis=1)


@pytest.fixture
def problem() -> Problem:
    return make_problem(sphere, [(-5.0, 5.0)] * 2, name="sphere", optimum_value=0.0, budget=100)


class TestNormalisation:
    def test_unit_corners_map_to_the_real_bounds(self, problem):
        real = problem.denormalise(np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]]))
        assert np.allclose(real[0], [-5.0, -5.0])
        assert np.allclose(real[1], [5.0, 5.0])
        assert np.allclose(real[2], [0.0, 0.0])

    def test_round_trip_is_the_identity(self, problem):
        rng = np.random.default_rng(0)
        points = rng.random((20, 2))
        assert np.allclose(problem.normalise(problem.denormalise(points)), points)

    def test_asymmetric_bounds(self):
        p = make_problem(sphere, [(-5.0, 10.0), (0.0, 15.0)])
        assert np.allclose(p.denormalise(np.array([[0.0, 1.0]])), [[-5.0, 15.0]])

    def test_rejects_mismatched_or_inverted_bounds(self):
        with pytest.raises(ValueError):
            Problem(func=sphere, lower=[0.0, 0.0], upper=[1.0])
        with pytest.raises(ValueError):
            Problem(func=sphere, lower=[1.0], upper=[1.0])

    def test_rejects_wrong_dimensionality(self, problem):
        with pytest.raises(ValueError, match="expected 2 dimensions"):
            problem.evaluate(np.zeros((3, 5)))


class TestBudget:
    def test_counter_tracks_evaluations(self, problem):
        problem.evaluate(np.zeros((7, 2)))
        assert problem.evaluations == 7
        assert problem.remaining == 93

    def test_batches_are_truncated_at_the_budget(self, problem):
        assert len(problem.evaluate(np.zeros((60, 2)))) == 60
        assert len(problem.evaluate(np.zeros((60, 2)))) == 40
        assert problem.exhausted()
        assert len(problem.evaluate(np.zeros((5, 2)))) == 0

    def test_budget_is_never_exceeded(self, problem):
        for _ in range(20):
            problem.evaluate(np.zeros((30, 2)))
        assert problem.evaluations == 100

    def test_unlimited_budget(self):
        p = make_problem(sphere, [(-1.0, 1.0)], budget=None)
        p.evaluate(np.zeros((1000, 1)))
        assert p.evaluations == 1000
        assert not p.exhausted()


class TestBookkeeping:
    def test_tracks_the_best_point_seen(self, problem):
        problem.evaluate(np.array([[0.9, 0.9], [0.5, 0.5], [0.8, 0.8]]))
        assert problem.best_value == pytest.approx(0.0)
        assert np.allclose(problem.best_real(), [0.0, 0.0])

    def test_best_never_worsens(self, problem):
        rng = np.random.default_rng(1)
        history = []
        for _ in range(10):
            problem.evaluate(rng.random((5, 2)))
            history.append(problem.best_value)
        assert np.all(np.diff(history) <= 0)

    def test_history_length_matches_the_number_of_batches(self, problem):
        for _ in range(4):
            problem.evaluate(np.zeros((2, 2)))
        assert len(problem.history_evals) == 4
        assert problem.history_evals == [2, 4, 6, 8]

    def test_non_finite_values_never_win(self):
        p = make_problem(lambda x: np.full(len(np.atleast_2d(x)), np.nan), [(-1.0, 1.0)])
        values = p.evaluate(np.array([[0.5]]))
        assert np.isinf(values[0]) and values[0] > 0

    def test_error_against_the_known_optimum(self, problem):
        problem.evaluate(np.array([[0.5, 0.5]]))
        assert problem.error() == pytest.approx(0.0)

    def test_error_is_nan_without_a_known_optimum(self):
        p = make_problem(sphere, [(-1.0, 1.0)])
        p.evaluate(np.array([[0.5]]))
        assert np.isnan(p.error())

    def test_evals_to_target(self, problem):
        problem.evaluate(np.array([[0.9, 0.9]]))
        assert problem.evals_to_target(1e-6) is None
        problem.evaluate(np.array([[0.5, 0.5]]))
        assert problem.evals_to_target(1e-6) == 2

    def test_reset_clears_all_state(self, problem):
        problem.evaluate(np.zeros((5, 2)))
        problem.reset()
        assert problem.evaluations == 0
        assert problem.best_position is None
        assert problem.history_evals == []
        assert np.isinf(problem.best_value)

    def test_copy_fresh_is_independent(self, problem):
        problem.evaluate(np.zeros((5, 2)))
        clone = problem.copy_fresh()
        assert clone.evaluations == 0
        assert problem.evaluations == 5
        assert clone.budget == problem.budget
        assert clone.name == problem.name

    def test_copy_fresh_can_change_the_budget(self, problem):
        assert problem.copy_fresh(budget=7).budget == 7

    def test_rejects_an_objective_returning_the_wrong_count(self):
        p = make_problem(lambda x: np.zeros(99), [(-1.0, 1.0)])
        with pytest.raises(ValueError, match="wrong number of values"):
            p.evaluate(np.zeros((3, 1)))
