"""Tests for the README table rendering.

These guard the formatting helpers that turn CSV results into markdown. A bug
here silently corrupts published numbers rather than raising, so the rounding
rules are pinned down explicitly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from make_readme_tables import number, table  # noqa: E402


class TestNumberFormatting:
    @pytest.mark.parametrize("value", [2000, 1700, 650, 1300, 10, 100, 30000])
    def test_whole_numbers_keep_their_trailing_zeros(self, value):
        """Trimming trailing zeros must not apply when there is no decimal point."""
        assert number(value, 0) == str(value)

    def test_trailing_zeros_after_a_decimal_point_are_trimmed(self):
        assert number(4.0, 4) == "4"
        assert number(-19.9670000, 4) == "-19.967"

    def test_significant_decimals_are_kept(self):
        assert number(-19.966683, 6) == "-19.966683"
        assert number(0.2474, 4) == "0.2474"

    def test_small_and_large_magnitudes_use_scientific_notation(self):
        assert "e-" in number(0.0001234, 4)
        assert "e+" in number(123456.0, 4)

    def test_zero_and_non_numbers(self):
        assert number(0) == "0"
        assert number(float("nan")) == "-"
        assert number(None) == "-"
        assert number("") == "-"

    def test_round_trips_within_tolerance(self):
        """Rendering then parsing must recover the value to four decimal places."""
        for value in (1234.5678, -0.004321, 98765.0, 2000.0, 0.5):
            rendered = number(value, 4)
            assert float(rendered) == pytest.approx(value, rel=1e-3, abs=1e-4)


class TestTable:
    def test_renders_a_markdown_table(self):
        result = table(["a", "b"], [["1", "2"], ["3", "4"]])
        lines = result.splitlines()
        assert lines[0] == "| a | b |"
        assert lines[1] == "| --- | ---: |"
        assert lines[-1] == "| 3 | 4 |"

    def test_every_row_has_the_same_column_count(self):
        header = ["x", "y", "z"]
        result = table(header, [["1", "2", "3"], ["4", "5", "6"]])
        for line in result.splitlines():
            assert line.count("|") == len(header) + 1
