"""Figure generation.

Chart conventions used throughout, so every figure reads the same way:

* one fixed categorical colour per algorithm, assigned by identity and never
  cycled, so a colour means the same thing in every figure;
* a line style per algorithm as well, so the series stay separable in greyscale
  and for colour-vision deficiency;
* two-pixel lines, recessive grid and axes, a legend whenever more than one
  series is present;
* a single y-axis per panel.  Quantities on different scales go in stacked
  panels rather than on a second axis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from .benchmarks import get
from .core import CDA, CDAConfig, contact_numbers, contact_sigmas

__all__ = [
    "plot_mechanism",
    "plot_equations",
    "plot_one_day",
    "ALGORITHM_COLOURS",
    "ALGORITHM_STYLES",
    "apply_style",
    "plot_surface",
    "plot_convergence",
    "plot_spread_snapshots",
    "plot_population",
    "plot_sensitivity",
    "plot_summary_box",
]

# Categorical slots 1-5 of the reference palette, in fixed order.  Validated for
# colour-vision deficiency on the adjacent pairlist.
ALGORITHM_COLOURS: dict[str, str] = {
    "CDA": "#2a78d6",              # blue, slot 1
    "CDA (tuned)": "#4a3aa7",      # violet, slot 7
    "PSO": "#eb6834",         # orange
    "GA": "#1baf7a",          # aqua
    "DE": "#eda100",          # yellow
    "Random": "#e87ba4",      # magenta
}

# Secondary encoding: the palette's contrast warning is relieved by line style
# and by the result tables in the README.
ALGORITHM_STYLES: dict[str, tuple] = {
    "CDA": (0, ()),
    "CDA (tuned)": (0, (4, 1, 1, 1)),
    "PSO": (0, (6, 2)),
    "GA": (0, (1, 1.6)),
    "DE": (0, (5, 1.5, 1, 1.5)),
    "Random": (0, (2, 2)),
}

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#dedcd6"
SURFACE = "#ffffff"


def apply_style() -> None:
    """Set the shared Matplotlib defaults."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.titlecolor": TEXT_PRIMARY,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.9,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "legend.labelcolor": TEXT_SECONDARY,
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
            "font.size": 9.5,
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        }
    )


def _colour(name: str) -> str:
    return ALGORITHM_COLOURS.get(name, "#52514e")


def _style(name: str) -> tuple:
    return ALGORITHM_STYLES.get(name, (0, ()))


def _save(fig: plt.Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Landscapes
# --------------------------------------------------------------------------
def plot_surface(name: str, path: str | Path, resolution: int = 220) -> Path:
    """A 3-D surface and a contour of a two-dimensional benchmark function."""
    apply_style()
    function = get(name)
    low, high = function.bounds(2)
    grid_x = np.linspace(low[0], high[0], resolution)
    grid_y = np.linspace(low[1], high[1], resolution)
    mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
    values = function.func(
        np.column_stack([mesh_x.ravel(), mesh_y.ravel()])
    ).reshape(mesh_x.shape)

    fig = plt.figure(figsize=(10.5, 4.4))
    surface_axis = fig.add_subplot(1, 2, 1, projection="3d")
    surface_axis.plot_surface(
        mesh_x, mesh_y, values, cmap="Blues_r", linewidth=0, antialiased=True, alpha=0.95
    )
    surface_axis.set_xlabel("$x_1$")
    surface_axis.set_ylabel("$x_2$")
    surface_axis.set_zlabel("f(x)")
    surface_axis.set_title(f"{name}: landscape")
    surface_axis.view_init(elev=32, azim=-128)
    surface_axis.grid(False)

    contour_axis = fig.add_subplot(1, 2, 2)
    contour_axis.contourf(mesh_x, mesh_y, values, levels=40, cmap="Blues_r")
    contour_axis.contour(mesh_x, mesh_y, values, levels=18, colors=GRID, linewidths=0.4)
    optimum = function.optimum(2)
    if optimum is not None:
        contour_axis.plot(
            optimum[0], optimum[1], marker="*", markersize=15,
            color="#e34948", markeredgecolor=SURFACE, markeredgewidth=1.2,
            linestyle="none", label="global minimum",
        )
        legend = contour_axis.legend(
            loc="lower left", frameon=True, facecolor=SURFACE, edgecolor=GRID, framealpha=0.95
        )
        legend.get_frame().set_linewidth(0.8)
    contour_axis.set_xlabel("$x_1$")
    contour_axis.set_ylabel("$x_2$")
    contour_axis.set_title(f"{name}: contours")
    contour_axis.grid(False)

    fig.tight_layout()
    return _save(fig, path)


# --------------------------------------------------------------------------
# Convergence
# --------------------------------------------------------------------------
def plot_convergence(
    traces: Mapping[str, Sequence[tuple[Sequence[int], Sequence[float]]]],
    path: str | Path,
    *,
    title: str,
    optimum: float | None = None,
    ylabel: str = "best objective value",
    logy: bool | None = None,
) -> Path:
    """Median best-so-far against function evaluations, one line per algorithm.

    ``traces`` maps an algorithm name to a list of ``(evaluations, best)`` pairs,
    one per seed.  The median across seeds is drawn as a line and the
    inter-quartile range as a light band.
    """
    apply_style()
    fig, axis = plt.subplots(figsize=(7.2, 4.3))

    if logy is None:
        logy = optimum is not None

    overall_longest = max(
        (max(max(evals) for evals, _ in runs) for runs in traces.values() if runs),
        default=0,
    )
    for name, runs in traces.items():
        if not runs:
            continue
        longest = max(max(evals) for evals, _ in runs)
        grid = np.linspace(1, longest, 400)
        stacked = []
        for evals, best in runs:
            evals = np.asarray(evals, dtype=float)
            best = np.asarray(best, dtype=float)
            # Step interpolation: the best-so-far holds until it improves.
            index = np.searchsorted(evals, grid, side="right") - 1
            index = np.clip(index, 0, len(best) - 1)
            series = best[index]
            series[grid < evals[0]] = best[0]
            stacked.append(series)
        stacked = np.vstack(stacked)

        if logy and optimum is not None:
            stacked = np.maximum(stacked - optimum, 1e-16)

        median = np.median(stacked, axis=0)
        lower = np.percentile(stacked, 25, axis=0)
        upper = np.percentile(stacked, 75, axis=0)

        axis.fill_between(grid, lower, upper, color=_colour(name), alpha=0.13, linewidth=0)
        axis.plot(grid, median, color=_colour(name), dashes=_style(name)[1] or (None, None),
                  label=name, solid_capstyle="round")
        # An algorithm that terminates before the budget runs out gets a dot at
        # the end of its line, so a short line reads as "stopped here" rather
        # than as a missing series.
        if longest < 0.995 * overall_longest:
            axis.plot(
                grid[-1], median[-1], marker="o", markersize=7, linestyle="none",
                color=_colour(name), markeredgecolor=SURFACE, markeredgewidth=1.4,
                zorder=5,
            )

    if logy:
        axis.set_yscale("log")
        axis.set_ylabel(f"{ylabel} - optimum" if optimum is not None else ylabel)
    else:
        axis.set_ylabel(ylabel)

    axis.set_xlabel("function evaluations")
    axis.set_title(title)
    axis.grid(True, which="major")
    if logy:
        axis.grid(True, which="minor", alpha=0.35)
    handles, labels = axis.get_legend_handles_labels()
    if any(len(runs) and max(max(e) for e, _ in runs) < 0.995 * overall_longest
           for runs in traces.values()):
        handles.append(
            Line2D([], [], marker="o", markersize=6, linestyle="none", color=TEXT_SECONDARY,
                   label="terminated early")
        )
        labels.append("terminated early")
    axis.legend(handles=handles, labels=labels, loc="best", ncol=2)
    fig.tight_layout()
    return _save(fig, path)


# --------------------------------------------------------------------------
# How CDA moves through the search space
# --------------------------------------------------------------------------
def plot_spread_snapshots(
    path: str | Path,
    function_name: str = "ripple_cone",
    iterations: Sequence[int] = (1, 3, 5, 7),
    config: CDAConfig | None = None,
    resolution: int = 300,
) -> Path:
    """Transmitter positions over the contours of a function, day by day.

    This is the figure that shows what the algorithm actually does: the outbreak
    starts spread across the community and collapses onto the deepest minimum.
    """
    apply_style()
    function = get(function_name)
    problem = function.to_problem(budget=200_000)
    config = config or CDAConfig.original(seed=0, track_positions=True)
    config = CDAConfig(**{**vars(config), "track_positions": True})
    result = CDA(config).run(problem)
    positions = result.history["positions"]

    low, high = function.bounds(2)
    grid_x = np.linspace(low[0], high[0], resolution)
    grid_y = np.linspace(low[1], high[1], resolution)
    mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
    values = function.func(
        np.column_stack([mesh_x.ravel(), mesh_y.ravel()])
    ).reshape(mesh_x.shape)

    shown = [i for i in iterations if i <= len(positions)]
    fig, axes = plt.subplots(1, len(shown), figsize=(3.3 * len(shown), 3.5), squeeze=False)
    for axis, iteration in zip(axes[0], shown):
        axis.contour(mesh_x, mesh_y, values, levels=26, colors="#b9d4f3", linewidths=0.5)
        points = positions[iteration - 1]
        axis.plot(
            points[:, 0], points[:, 1], linestyle="none", marker="o", markersize=4.5,
            color="#2a78d6", markeredgecolor=SURFACE, markeredgewidth=0.7,
        )
        optimum = function.optimum(2)
        if optimum is not None:
            axis.plot(
                optimum[0], optimum[1], marker="*", markersize=14, linestyle="none",
                color="#e34948", markeredgecolor=SURFACE, markeredgewidth=1.0,
            )
        axis.set_title(f"day {iteration}  ({len(points)} transmitters)")
        axis.set_xlabel("$x_1$")
        axis.set_xlim(low[0], high[0])
        axis.set_ylim(low[1], high[1])
        axis.grid(False)
    axes[0][0].set_ylabel("$x_2$")

    handles = [
        Line2D([], [], linestyle="none", marker="o", markersize=6, color="#2a78d6", label="transmitter"),
        Line2D([], [], linestyle="none", marker="*", markersize=11, color="#e34948", label="global minimum"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"How the outbreak spreads on {function_name}", y=1.02, fontsize=11, fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


def plot_population(
    path: str | Path,
    function_name: str = "rastrigin",
    dim: int = 10,
    configs: Mapping[str, CDAConfig] | None = None,
    max_days: int = 150,
    smoothing: int = 9,
) -> Path:
    """Population size and infection rate per day, in stacked panels.

    The two quantities live on different scales, so they get their own panels
    rather than a shared second axis.  Each series is drawn twice: the raw
    per-day value faintly, and a rolling median on top, because the day-to-day
    signal is noisy enough to hide the trend.
    """
    apply_style()
    configs = configs or {
        "CDA": CDAConfig.original(seed=0),
        "CDA (tuned)": CDAConfig.tuned(seed=0),
    }
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.2, 5.4), sharex=True)

    def smooth(series: np.ndarray) -> np.ndarray:
        if smoothing < 3 or series.size < smoothing:
            return series
        padded = np.pad(series, smoothing // 2, mode="edge")
        return np.array(
            [np.median(padded[i : i + smoothing]) for i in range(series.size)]
        )

    for name, config in configs.items():
        problem = get(function_name).to_problem(dim, budget=30_000)
        result = CDA(config).run(problem)
        iterations = np.asarray(result.history["iteration"])[:max_days]
        colour, dashes = _colour(name), _style(name)[1] or (None, None)
        for axis, key in ((top, "population"), (bottom, "acceptance_rate")):
            values = np.asarray(result.history[key], dtype=float)[:max_days]
            axis.plot(iterations, values, color=colour, linewidth=0.8, alpha=0.3)
            axis.plot(iterations, smooth(values), color=colour, dashes=dashes, label=name)
            if len(iterations) < max_days:
                axis.plot(
                    iterations[-1], smooth(values)[-1], marker="o", markersize=7,
                    linestyle="none", color=colour, markeredgecolor=SURFACE,
                    markeredgewidth=1.4, zorder=5,
                )

    top.set_ylabel("transmitters alive")
    top.set_title(f"Population dynamics on {function_name} ({dim}D), first {max_days} days")
    handles, labels = top.get_legend_handles_labels()
    handles.append(
        Line2D([], [], marker="o", markersize=6, linestyle="none", color=TEXT_SECONDARY)
    )
    labels.append("outbreak ends")
    top.legend(handles=handles, labels=labels, loc="best", ncol=3)
    bottom.set_ylabel("infection rate")
    bottom.set_xlabel("iteration (day)")
    bottom.set_ylim(0, 1)
    fig.tight_layout()
    return _save(fig, path)


def plot_sensitivity(
    grid: np.ndarray,
    alphas: Sequence[float],
    contacts: Sequence[int],
    path: str | Path,
    *,
    title: str = "Parameter sensitivity",
    label: str = "median error (log scale)",
) -> Path:
    """Heatmap of solution quality over the two main tuning parameters."""
    apply_style()
    fig, axis = plt.subplots(figsize=(6.4, 4.4))
    image = axis.imshow(grid, cmap="Blues_r", aspect="auto", origin="lower")
    axis.set_xticks(range(len(contacts)), [str(c) for c in contacts])
    axis.set_yticks(range(len(alphas)), [str(a) for a in alphas])
    axis.set_xlabel("ContactNum$_{max}$")
    axis.set_ylabel(r"$\alpha$")
    axis.set_title(title)
    axis.grid(False)

    # Direct labels on every cell: the relief for the palette's contrast warning.
    # The ink colour follows where the cell sits in the colour ramp, not where it
    # sits in the data, so the label tracks the actual background lightness.
    finite = grid[np.isfinite(grid)]
    low = float(finite.min()) if finite.size else 0.0
    high = float(finite.max()) if finite.size else 1.0
    extent = (high - low) or 1.0
    for row in range(grid.shape[0]):
        for column in range(grid.shape[1]):
            value = grid[row, column]
            position = (value - low) / extent  # 0 = darkest cell, 1 = lightest
            axis.text(
                column, row, f"{value:.2f}", ha="center", va="center", fontsize=8,
                color=SURFACE if position < 0.42 else TEXT_PRIMARY,
            )
    bar = fig.colorbar(image, ax=axis)
    bar.set_label(label, color=TEXT_SECONDARY, fontsize=9)
    bar.outline.set_edgecolor(GRID)
    fig.tight_layout()
    return _save(fig, path)


def plot_summary_box(
    data: Mapping[str, Mapping[str, Sequence[float]]],
    path: str | Path,
    *,
    title: str = "Final error by algorithm",
) -> Path:
    """Box plots of final error per algorithm, one panel per problem."""
    apply_style()
    problems = list(data)
    columns = min(3, len(problems))
    rows = int(np.ceil(len(problems) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 3.2 * rows), squeeze=False)

    for index, problem in enumerate(problems):
        axis = axes[index // columns][index % columns]
        series = data[problem]
        names = list(series)
        samples = [np.maximum(np.asarray(series[n], dtype=float), 1e-16) for n in names]
        box = axis.boxplot(
            samples, patch_artist=True, widths=0.6, showfliers=False,
            medianprops=dict(color=TEXT_PRIMARY, linewidth=1.4),
            whiskerprops=dict(color=GRID, linewidth=1.0),
            capprops=dict(color=GRID, linewidth=1.0),
        )
        for patch, name in zip(box["boxes"], names):
            patch.set_facecolor(_colour(name))
            patch.set_alpha(0.75)
            patch.set_edgecolor(SURFACE)
            patch.set_linewidth(1.5)
        axis.set_yscale("log")
        axis.set_xticks(range(1, len(names) + 1), names, rotation=30, ha="right")
        axis.set_title(problem)
        axis.set_ylabel("error")

    for index in range(len(problems), rows * columns):
        axes[index // columns][index % columns].axis("off")

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


# --------------------------------------------------------------------------
# Explanatory figures: what the algorithm actually does
# --------------------------------------------------------------------------
ACCEPTED = "#2a78d6"   # categorical slot 1
REJECTED = "#9b9a94"   # neutral; a state, not a series


def plot_mechanism(path: str | Path, contact_num_max: int = 8, day: int = 3,
                   alpha: float = 2.0, seed: int = 4) -> Path:
    """The core mechanism: how a transmitter's own quality sets its behaviour.

    Three transmitters with different health care factors, each in its own panel
    on identical axes so the difference in reach is directly comparable.  The
    contact counts and radii come from Equations 1 and 2 as implemented, not
    from hand placement.
    """
    apply_style()
    rng = np.random.default_rng(seed)

    values = np.array([1.0, 5.0, 9.0])          # best, middle, worst
    headings = ["Best solution in the population",
                "A middling solution",
                "Worst solution in the population"]
    counts = contact_numbers(values, contact_num_max)
    sigmas = contact_sigmas(values, day, alpha)
    extent = 2.6 * float(sigmas.max())

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.3), sharex=True, sharey=True)
    for axis, heading, count, sigma in zip(axes, headings, counts, sigmas):
        axis.add_patch(plt.Circle((0.0, 0.0), 2.0 * sigma, fill=True,
                                  facecolor=ACCEPTED, alpha=0.07, linewidth=0))
        axis.add_patch(plt.Circle((0.0, 0.0), 2.0 * sigma, fill=False,
                                  linestyle=(0, (4, 3)), edgecolor=ACCEPTED,
                                  linewidth=1.2, alpha=0.8))
        directions = rng.normal(size=(count, 2))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        radii = np.abs(rng.normal(0.0, sigma, size=count))[:, None]
        targets = radii * directions
        for target in targets:
            axis.annotate("", xy=target, xytext=(0.0, 0.0),
                          arrowprops=dict(arrowstyle="-|>", color=ACCEPTED,
                                          linewidth=1.5, alpha=0.9,
                                          shrinkA=6, shrinkB=0))
        axis.plot(targets[:, 0], targets[:, 1], linestyle="none", marker="o",
                  markersize=6, color=ACCEPTED, markeredgecolor=SURFACE,
                  markeredgewidth=0.9)
        axis.plot(0.0, 0.0, marker="*", markersize=22, linestyle="none",
                  color="#e34948", markeredgecolor=SURFACE, markeredgewidth=1.3)

        plural = "contact" if count == 1 else "contacts"
        axis.set_title(f"{heading}\n{count} {plural}, reach {2 * sigma:.2f}",
                       fontsize=10, fontweight="bold")
        axis.set_xlim(-extent, extent)
        axis.set_ylim(-extent, extent)
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.grid(False)
        for spine in axis.spines.values():
            spine.set_edgecolor(GRID)

    handles = [
        Line2D([], [], marker="*", markersize=14, linestyle="none", color="#e34948",
               label="transmitter"),
        Line2D([], [], marker="o", markersize=7, linestyle="none", color=ACCEPTED,
               label="individual it contacts"),
        Line2D([], [], linestyle=(0, (4, 3)), color=ACCEPTED, label="reach of its contacts"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("A better solution makes more contacts, but reaches less far",
                 fontsize=12, fontweight="bold", y=1.06)
    fig.tight_layout()
    return _save(fig, path)


def plot_equations(path: str | Path, alpha: float = 2.0, days: int = 12) -> Path:
    """The two equations, drawn. Left: how many contacts. Right: how far."""
    apply_style()
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.0, 4.0))

    # --- Equation 1 --------------------------------------------------------
    ratios = np.linspace(0.0, 1.0, 400)
    # Categorical slots 1-3, assigned in fixed order.
    slots = ("#2a78d6", "#eb6834", "#1baf7a")
    patterns = ((None, None), (6, 2), (1, 1.6))
    for index, maximum in enumerate((3, 5, 10)):
        counts = np.floor(ratios * (maximum - 1)) + 1
        left.step(ratios, counts, where="post", color=slots[index],
                  dashes=patterns[index], label=f"ContactNum$_{{max}}$ = {maximum}")
    left.set_xlabel("worst HCF  $\\leftarrow$   rank $r$   $\\rightarrow$  best HCF")
    left.set_ylabel("contacts made")
    left.set_title("Equation 1: how many contacts")
    left.legend(loc="upper left")

    # --- Equation 2 --------------------------------------------------------
    grid = np.arange(1, days + 1)
    for index, (ratio, name) in enumerate(((0.0, "worst HCF"), (0.5, "middle HCF"), (1.0, "best HCF"))):
        sigma_min = (1.0 / 6.0) / (alpha ** (grid - 1))
        sigma = ratio * (sigma_min - 1.0 / 6.0) + 1.0 / 6.0
        colour = ("#eb6834", "#1baf7a", "#2a78d6")[index]
        dashes = ((6, 2), (1, 1.6), (None, None))[index]
        right.plot(grid, sigma, color=colour, marker="o", markersize=4,
                   dashes=dashes, label=name)
    right.set_yscale("log")
    right.set_xlabel("day of the outbreak ($m$)")
    right.set_ylabel(r"contact reach $\sigma$")
    right.set_title(f"Equation 2: how far they reach ($\\alpha$ = {alpha})")
    right.legend(loc="lower left")

    fig.suptitle("One quantity, how good a solution is, drives both behaviours",
                 fontsize=11.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save(fig, path)


def plot_one_day(path: str | Path, function_name: str = "ripple_cone",
                 n_transmitters: int = 10, seed: int = 11,
                 resolution: int = 300) -> Path:
    """A single day of the outbreak, in three steps.

    Shows the selection rule directly: a contacted individual joins the search
    only if its objective value is no worse than the transmitter that reached it.
    """
    apply_style()
    function = get(function_name)
    problem = function.to_problem(budget=100_000)
    config = CDAConfig.original(seed=seed, n_transmitters=n_transmitters, contact_num_max=6)
    engine = CDA(config)
    rng = np.random.default_rng(seed)

    transmitters = engine._initial_transmitters(rng, problem.dim)
    values = problem.evaluate(transmitters)
    children, parent_index, _ = engine._spread(transmitters, values, 1, rng)
    child_values = problem.evaluate(children)
    caught = child_values <= values[parent_index]

    real_parents = problem.denormalise(transmitters)
    real_children = problem.denormalise(children)

    low, high = function.bounds(2)
    grid_x = np.linspace(low[0], high[0], resolution)
    grid_y = np.linspace(low[1], high[1], resolution)
    mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
    surface = function.func(
        np.column_stack([mesh_x.ravel(), mesh_y.ravel()])
    ).reshape(mesh_x.shape)

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.4))
    titles = [
        f"1. Morning: {len(transmitters)} transmitters",
        f"2. They make {len(children)} contacts",
        f"3. Evening: {int(caught.sum())} caught it",
    ]
    for axis, title in zip(axes, titles):
        axis.contour(mesh_x, mesh_y, surface, levels=22, colors="#cfe2f8", linewidths=0.5)
        axis.set_title(title, fontsize=10.5)
        axis.set_xlabel("$x_1$")
        axis.set_xlim(low[0], high[0])
        axis.set_ylim(low[1], high[1])
        axis.grid(False)
    axes[0].set_ylabel("$x_2$")

    axes[0].plot(real_parents[:, 0], real_parents[:, 1], linestyle="none", marker="*",
                 markersize=15, color="#e34948", markeredgecolor=SURFACE, markeredgewidth=1.0)

    for child, parent in zip(real_children, real_parents[parent_index]):
        axes[1].annotate("", xy=child, xytext=parent,
                         arrowprops=dict(arrowstyle="-|>", color=REJECTED,
                                         linewidth=0.9, alpha=0.8, shrinkA=4, shrinkB=1))
    axes[1].plot(real_parents[:, 0], real_parents[:, 1], linestyle="none", marker="*",
                 markersize=15, color="#e34948", markeredgecolor=SURFACE, markeredgewidth=1.0)

    axes[2].plot(real_children[~caught, 0], real_children[~caught, 1], linestyle="none",
                 marker="x", markersize=6, color=REJECTED, markeredgewidth=1.6)
    axes[2].plot(real_children[caught, 0], real_children[caught, 1], linestyle="none",
                 marker="o", markersize=7, color=ACCEPTED, markeredgecolor=SURFACE,
                 markeredgewidth=1.0)

    handles = [
        Line2D([], [], marker="*", markersize=13, linestyle="none", color="#e34948",
               label="transmitter"),
        Line2D([], [], marker="o", markersize=7, linestyle="none", color=ACCEPTED,
               label="caught it: no worse than its transmitter, so it searches tomorrow"),
        Line2D([], [], marker="x", markersize=7, linestyle="none", color=REJECTED,
               markeredgewidth=1.6, label="shrugged it off: worse, so discarded"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("One day of the outbreak: yesterday's transmitters are then dropped",
                 fontsize=11.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save(fig, path)
