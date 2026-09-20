"""Tests for the benchmark suite.

The most valuable test here checks every documented optimum by evaluating the
function at its documented minimiser.  That catches transcription errors, which
are the most likely kind of bug in a file of hand-copied formulas.
"""

from __future__ import annotations

import numpy as np
import pytest

from cda import benchmarks as B


ALL_NAMES = sorted(B.FUNCTIONS)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_documented_optimum_is_correct(name):
    """f(documented minimiser) must equal the documented minimum."""
    function = B.get(name)
    dims = [function.fixed_dim] if function.fixed_dim else [2, 10, 30]
    for dim in dims:
        position = function.optimum(dim)
        expected = function.optimum_value_for(dim)
        if position is None or not np.isfinite(expected):
            continue
        actual = float(function.func(position[None, :])[0])
        assert actual == pytest.approx(expected, abs=1e-6), f"{name} in {dim}D"


@pytest.mark.parametrize("name", ALL_NAMES)
def test_optimum_is_not_beaten_by_random_sampling(name):
    """A large random sample must not find anything below the documented minimum."""
    function = B.get(name)
    dim = function.fixed_dim or 4
    low, high = function.bounds(dim)
    rng = np.random.default_rng(0)
    points = low + rng.random((4000, dim)) * (high - low)
    expected = function.optimum_value_for(dim)
    if not np.isfinite(expected):
        return
    assert function.func(points).min() >= expected - 1e-9, name


@pytest.mark.parametrize("name", ALL_NAMES)
def test_is_vectorised_and_finite(name):
    function = B.get(name)
    dim = function.fixed_dim or 5
    low, high = function.bounds(dim)
    rng = np.random.default_rng(1)
    points = low + rng.random((37, dim)) * (high - low)
    values = function.func(points)
    assert values.shape == (37,)
    assert np.all(np.isfinite(values)), name
    # Row-by-row evaluation must agree with the batch.
    for i in (0, 5, 36):
        assert float(function.func(points[i][None, :])[0]) == pytest.approx(values[i])


@pytest.mark.parametrize("name", ALL_NAMES)
def test_optimum_lies_inside_the_bounds(name):
    function = B.get(name)
    dim = function.fixed_dim or 10
    position = function.optimum(dim)
    if position is None:
        return
    low, high = function.bounds(dim)
    assert np.all(position >= low - 1e-9) and np.all(position <= high + 1e-9), name


class TestPaperFunctions:
    def test_f1_matches_the_closed_form_optimum(self):
        # -20 sin(0.1)/0.1 at the centre (4, 4).
        value = float(B.paper_f1(np.array([[4.0, 4.0]]))[0])
        assert value == pytest.approx(-200.0 * np.sin(0.1))
        assert value == pytest.approx(-19.96668, abs=1e-5)

    def test_f1_has_a_single_global_basin_at_four_four(self):
        centre = float(B.paper_f1(np.array([[4.0, 4.0]]))[0])
        around = B.paper_f1(
            np.array([[4.2, 4.0], [3.8, 4.0], [4.0, 4.2], [4.0, 3.8]])
        )
        assert np.all(around > centre)

    def test_f2_global_minimum_is_the_deepest_point_found(self):
        # Verified by a dense grid search plus local refinement.
        true_min = B.get("paper_f2").optimum_value
        rng = np.random.default_rng(0)
        samples = B.paper_f2(rng.uniform(-5.0, 5.0, size=(200_000, 2)))
        assert samples.min() > true_min
        assert true_min == pytest.approx(-0.24740519, abs=1e-7)

    def test_f2_is_symmetric_in_x2(self):
        a = B.paper_f2(np.array([[-0.2, 0.3]]))
        b = B.paper_f2(np.array([[-0.2, -0.3]]))
        assert a[0] == pytest.approx(b[0])


class TestSuite:
    def test_suite_builds_problems_with_budgets(self):
        problems = B.make_suite()
        assert len(problems) == len(B.SUITE)
        for problem in problems:
            assert problem.budget is not None and problem.budget > 0
            assert problem.name

    def test_names_are_unique(self):
        names = [p.name for p in B.make_suite()]
        assert len(names) == len(set(names))

    def test_budget_grows_with_dimension(self):
        assert B.budget_for(2) < B.budget_for(10) < B.budget_for(30)

    def test_fixed_dimension_functions_reject_other_dimensions(self):
        with pytest.raises(ValueError, match="only defined in 2 dimensions"):
            B.get("easom").to_problem(5)

    def test_unknown_name_is_rejected(self):
        with pytest.raises(KeyError, match="unknown benchmark"):
            B.get("not_a_function")

    def test_michalewicz_optimum_depends_on_dimension(self):
        function = B.get("michalewicz")
        assert function.optimum_value_for(2) != function.optimum_value_for(10)
        assert function.optimum_value_for(2) == pytest.approx(-1.8013034, abs=1e-6)
