"""Tests for Equations 1 and 2 of the manuscript."""

from __future__ import annotations

import numpy as np
import pytest

from cda.core import CDAConfig, contact_numbers, contact_sigmas


class TestContactNumbers:
    """Equation 1: how many high-risk contacts each transmitter makes."""

    def test_best_gets_the_maximum_and_worst_gets_one(self):
        values = np.array([1.0, 5.0, 9.0])
        counts = contact_numbers(values, contact_num_max=5)
        assert counts[0] == 5, "the lowest HCF must make the most contacts"
        assert counts[-1] == 1, "the highest HCF must make exactly one contact"

    def test_counts_decrease_with_worsening_hcf(self):
        values = np.linspace(0.0, 10.0, 11)
        counts = contact_numbers(values, contact_num_max=7)
        assert np.all(np.diff(counts) <= 0)

    def test_never_below_one(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            values = rng.normal(size=20) * rng.uniform(0.1, 1000)
            assert np.all(contact_numbers(values, 5) >= 1)

    @pytest.mark.parametrize("contact_num_max", [1, 2, 3, 10, 50])
    def test_never_above_the_maximum(self, contact_num_max):
        values = np.linspace(-5.0, 5.0, 25)
        counts = contact_numbers(values, contact_num_max)
        assert counts.max() <= contact_num_max
        assert counts.min() >= 1

    def test_degenerate_population_is_treated_as_all_best(self):
        # The paper does not define the ratio when every HCF is equal.
        counts = contact_numbers(np.array([3.0, 3.0, 3.0]), 5)
        assert np.all(counts == 5)

    def test_single_transmitter(self):
        assert contact_numbers(np.array([2.0]), 6)[0] == 6

    def test_contact_num_max_of_one_gives_everyone_one(self):
        assert np.all(contact_numbers(np.array([1.0, 2.0, 3.0]), 1) == 1)


class TestContactSigmas:
    """Equation 2: the standard deviation of each contact radius."""

    def test_worst_transmitter_always_uses_the_maximum(self):
        values = np.array([1.0, 5.0, 9.0])
        for day in (1, 2, 5, 20):
            sigmas = contact_sigmas(values, day, alpha=2.0)
            assert sigmas[-1] == pytest.approx(1.0 / 6.0)

    def test_best_transmitter_uses_the_shrinking_minimum(self):
        values = np.array([1.0, 5.0, 9.0])
        for day in (1, 2, 5, 20):
            sigmas = contact_sigmas(values, day, alpha=2.0)
            assert sigmas[0] == pytest.approx((1.0 / 6.0) / 2.0 ** (day - 1))

    def test_first_day_is_uniform_across_the_population(self):
        values = np.array([1.0, 5.0, 9.0])
        sigmas = contact_sigmas(values, 1, alpha=2.0)
        assert np.allclose(sigmas, 1.0 / 6.0)

    def test_best_sigma_shrinks_monotonically_with_the_day(self):
        values = np.array([1.0, 5.0, 9.0])
        best = [contact_sigmas(values, m, alpha=2.0)[0] for m in range(1, 15)]
        assert np.all(np.diff(best) < 0)

    def test_sigma_increases_with_worsening_hcf(self):
        values = np.linspace(0.0, 10.0, 11)
        sigmas = contact_sigmas(values, 5, alpha=2.0)
        assert np.all(np.diff(sigmas) > 0), "higher HCF must give a larger radius"

    def test_never_exceeds_the_maximum(self):
        values = np.linspace(-3.0, 7.0, 13)
        for day in range(1, 30):
            assert contact_sigmas(values, day, alpha=2.0).max() <= 1.0 / 6.0 + 1e-15

    def test_larger_alpha_shrinks_faster(self):
        values = np.array([1.0, 9.0])
        slow = contact_sigmas(values, 6, alpha=1.2)[0]
        fast = contact_sigmas(values, 6, alpha=3.0)[0]
        assert fast < slow

    def test_literal_formula_reproduces_the_typeset_equation(self):
        # As typeset, the best transmitter's radius GROWS towards 1/3.
        values = np.array([1.0, 5.0, 9.0])
        sigmas = contact_sigmas(values, 8, alpha=2.0, formula="literal")
        expected_best = 1.0 / 6.0 - (1.0 / 6.0) / 2.0 ** 7 + 1.0 / 6.0
        assert sigmas[0] == pytest.approx(expected_best)
        assert sigmas[0] > 1.0 / 6.0
        assert sigmas[-1] == pytest.approx(1.0 / 6.0)

    def test_literal_and_corrected_agree_on_day_one(self):
        values = np.array([1.0, 5.0, 9.0])
        assert np.allclose(
            contact_sigmas(values, 1, 2.0, formula="corrected"),
            contact_sigmas(values, 1, 2.0, formula="literal"),
        )

    def test_unknown_formula_is_rejected(self):
        with pytest.raises(ValueError, match="unknown sigma formula"):
            contact_sigmas(np.array([1.0, 2.0]), 1, 2.0, formula="nonsense")

    def test_degenerate_population_is_treated_as_all_best(self):
        sigmas = contact_sigmas(np.array([4.0, 4.0]), 3, alpha=2.0)
        assert np.allclose(sigmas, (1.0 / 6.0) / 4.0)


class TestConfigValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"n_transmitters": 0},
            {"contact_num_max": 0},
            {"alpha": 0.0},
            {"alpha": -1.0},
            {"sigma_max": 0.0},
        ],
    )
    def test_invalid_parameters_are_rejected(self, kwargs):
        with pytest.raises(ValueError):
            CDAConfig(**kwargs)

    def test_paper_preset_matches_table_one(self):
        config = CDAConfig.paper()
        assert config.n_transmitters == 50
        assert config.alpha == 2.0
        assert config.on_extinction == "stop"

    def test_budgeted_preset_restarts(self):
        config = CDAConfig.budgeted()
        assert config.on_extinction == "restart"
        assert config.on_convergence == "restart"
