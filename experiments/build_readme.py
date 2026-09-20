#!/usr/bin/env python3
"""Splice generated result tables into README.md.

The README contains marker pairs::

    <!-- BEGIN:SUMMARY -->
    ...generated...
    <!-- END:SUMMARY -->

This script replaces whatever sits between each pair with the table rendered
from ``results/*.csv``, so the README can never drift from the committed data.
Running it twice in a row changes nothing.

Usage:
    python experiments/build_readme.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import make_readme_tables as tables  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"

SECTIONS = {
    "RADIAL_FUNCTIONS": tables.radial_tables,
    "RANKS": tables.rank_table,
    "SUMMARY": tables.summary_table,
    "SUCCESS": tables.success_table,
    "SIGNIFICANCE": tables.significance_table,
    "BUDGET": tables.budget_table,
    "SENSITIVITY": tables.sensitivity_tables,
}


def main() -> None:
    text = README.read_text()
    for name, render in SECTIONS.items():
        pattern = re.compile(
            rf"<!-- BEGIN:{name} -->.*?<!-- END:{name} -->", re.DOTALL
        )
        if not pattern.search(text):
            print(f"  no markers for {name}, skipped")
            continue
        body = render()
        replacement = f"<!-- BEGIN:{name} -->\n{body}\n<!-- END:{name} -->"
        text = pattern.sub(lambda _: replacement, text)
        print(f"  filled {name}")
    README.write_text(text)
    print(f"written to {README}")


if __name__ == "__main__":
    main()
