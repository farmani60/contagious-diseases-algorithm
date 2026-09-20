"""Baseline optimisers used for comparison.

The manuscript compares CDA against Particle Swarm Optimisation and a Genetic
Algorithm, using the parameters of its Table 1 (population 50, cognitive and
social weights 1.5, crossover rate 0.8, mutation rate 0.01).  Those settings are
reproduced here.  Differential Evolution and uniform random search are added: DE
because it is the standard strong baseline for this class of problem, random
search because it is the floor any optimiser must clear to be worth anything.

All four share one interface: ``run(problem) -> BaselineResult``.  They search the
same normalised unit cube as CDA and spend the same evaluation budget through the
same :class:`~cda.problem.Problem` accounting, so the comparison is fair despite
CDA having a population size that changes every iteration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .problem import Problem

__all__ = [
    "BaselineResult",
    "ParticleSwarm",
    "GeneticAlgorithm",
    "DifferentialEvolution",
    "RandomSearch",
    "ALGORITHMS",
]


@dataclass
class BaselineResult:
    best_value: float
    best_position: np.ndarray
    evaluations: int
    iterations: int
    stop_reason: str = "budget_exhausted"
    history: dict[str, list] = field(default_factory=dict)


class ParticleSwarm:
    """Global-best PSO with an inertia weight.

    Parameters follow Table 1 of the manuscript: 50 particles, cognitive and
    social weights both 1.5.  The inertia weight uses Clerc's standard 0.729,
    which the paper does not specify.
    """

    name = "PSO"

    def __init__(
        self,
        n_particles: int = 50,
        inertia: float = 0.729,
        cognitive: float = 1.5,
        social: float = 1.5,
        velocity_clamp: float = 0.5,
        seed: int | None = None,
    ) -> None:
        self.n_particles = n_particles
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.velocity_clamp = velocity_clamp
        self.seed = seed

    def run(self, problem: Problem) -> BaselineResult:
        rng = np.random.default_rng(self.seed)
        problem.reset()
        dim = problem.dim

        position = rng.random((self.n_particles, dim))
        velocity = rng.uniform(-self.velocity_clamp, self.velocity_clamp, (self.n_particles, dim))
        value = problem.evaluate(position)
        position = position[: len(value)]
        velocity = velocity[: len(value)]

        pbest, pbest_value = position.copy(), value.copy()
        gbest_idx = int(np.argmin(pbest_value))
        gbest, gbest_value = pbest[gbest_idx].copy(), float(pbest_value[gbest_idx])

        history: dict[str, list] = {"evaluations": [], "best": [], "mean": []}
        iteration = 0
        while not problem.exhausted():
            iteration += 1
            r1 = rng.random((len(position), dim))
            r2 = rng.random((len(position), dim))
            velocity = (
                self.inertia * velocity
                + self.cognitive * r1 * (pbest - position)
                + self.social * r2 * (gbest - position)
            )
            np.clip(velocity, -self.velocity_clamp, self.velocity_clamp, out=velocity)
            position = np.clip(position + velocity, 0.0, 1.0)

            value = problem.evaluate(position)
            if len(value) == 0:
                break
            position = position[: len(value)]
            velocity = velocity[: len(value)]
            pbest, pbest_value = pbest[: len(value)], pbest_value[: len(value)]

            improved = value < pbest_value
            pbest[improved] = position[improved]
            pbest_value[improved] = value[improved]
            best_idx = int(np.argmin(pbest_value))
            if pbest_value[best_idx] < gbest_value:
                gbest_value = float(pbest_value[best_idx])
                gbest = pbest[best_idx].copy()

            history["evaluations"].append(problem.evaluations)
            history["best"].append(gbest_value)
            history["mean"].append(float(np.mean(value[np.isfinite(value)])) if np.isfinite(value).any() else float("inf"))

        return BaselineResult(
            best_value=float(problem.best_value),
            best_position=problem.best_real(),
            evaluations=problem.evaluations,
            iterations=iteration,
            history=history,
        )


class GeneticAlgorithm:
    """Real-coded GA with tournament selection, BLX-alpha crossover and elitism.

    Crossover rate 0.8 and mutation rate 0.01 are the manuscript's Table 1 values.
    Mutation is Gaussian with a fixed width, applied per gene at that rate.
    """

    name = "GA"

    def __init__(
        self,
        population: int = 50,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.01,
        mutation_scale: float = 0.1,
        blend_alpha: float = 0.5,
        tournament_size: int = 3,
        elite: int = 2,
        seed: int | None = None,
    ) -> None:
        self.population = population
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.blend_alpha = blend_alpha
        self.tournament_size = tournament_size
        self.elite = elite
        self.seed = seed

    def run(self, problem: Problem) -> BaselineResult:
        rng = np.random.default_rng(self.seed)
        problem.reset()
        dim = problem.dim

        pop = rng.random((self.population, dim))
        fitness = problem.evaluate(pop)
        pop = pop[: len(fitness)]

        history: dict[str, list] = {"evaluations": [], "best": [], "mean": []}
        iteration = 0
        while not problem.exhausted():
            iteration += 1
            n = len(pop)
            order = np.argsort(fitness, kind="stable")
            elites = pop[order[: self.elite]].copy()

            # Tournament selection.
            contenders = rng.integers(0, n, size=(n, self.tournament_size))
            winners = contenders[np.arange(n), np.argmin(fitness[contenders], axis=1)]
            parents = pop[winners]

            # BLX-alpha crossover on consecutive pairs.
            children = parents.copy()
            pairs = n // 2
            if pairs:
                a, b = parents[:pairs * 2:2], parents[1:pairs * 2:2]
                do_cross = rng.random(pairs) < self.crossover_rate
                low = np.minimum(a, b)
                high = np.maximum(a, b)
                spread = high - low
                lo = low - self.blend_alpha * spread
                hi = high + self.blend_alpha * spread
                c1 = lo + rng.random((pairs, dim)) * (hi - lo)
                c2 = lo + rng.random((pairs, dim)) * (hi - lo)
                children[:pairs * 2:2] = np.where(do_cross[:, None], c1, a)
                children[1:pairs * 2:2] = np.where(do_cross[:, None], c2, b)

            # Per-gene Gaussian mutation.
            mutate = rng.random((n, dim)) < self.mutation_rate
            children = np.where(
                mutate, children + rng.normal(0.0, self.mutation_scale, (n, dim)), children
            )
            children = np.clip(children, 0.0, 1.0)

            child_fitness = problem.evaluate(children)
            if len(child_fitness) == 0:
                break
            children = children[: len(child_fitness)]

            # Elitism: the best of the old generation always survives.
            pop = np.vstack([children, elites])
            fitness = np.concatenate([child_fitness, np.sort(fitness)[: self.elite]])
            keep = np.argsort(fitness, kind="stable")[: self.population]
            pop, fitness = pop[keep], fitness[keep]

            history["evaluations"].append(problem.evaluations)
            history["best"].append(float(problem.best_value))
            history["mean"].append(float(np.mean(fitness[np.isfinite(fitness)])) if np.isfinite(fitness).any() else float("inf"))

        return BaselineResult(
            best_value=float(problem.best_value),
            best_position=problem.best_real(),
            evaluations=problem.evaluations,
            iterations=iteration,
            history=history,
        )


class DifferentialEvolution:
    """DE/rand/1/bin, the standard strong baseline for continuous problems."""

    name = "DE"

    def __init__(
        self,
        population: int = 50,
        differential_weight: float = 0.5,
        crossover_probability: float = 0.9,
        seed: int | None = None,
    ) -> None:
        self.population = population
        self.differential_weight = differential_weight
        self.crossover_probability = crossover_probability
        self.seed = seed

    def run(self, problem: Problem) -> BaselineResult:
        rng = np.random.default_rng(self.seed)
        problem.reset()
        dim = problem.dim

        pop = rng.random((self.population, dim))
        fitness = problem.evaluate(pop)
        pop = pop[: len(fitness)]

        history: dict[str, list] = {"evaluations": [], "best": [], "mean": []}
        iteration = 0
        while not problem.exhausted():
            iteration += 1
            n = len(pop)
            if n < 4:
                break
            # Three distinct donors per target, none equal to the target.
            # Giving each row's own index an infinite sort key excludes the
            # target from its own donor set without a Python loop.
            keys = rng.random((n, n))
            keys[np.arange(n), np.arange(n)] = np.inf
            idx = np.argpartition(keys, 3, axis=1)[:, :3]
            a, b, c = pop[idx[:, 0]], pop[idx[:, 1]], pop[idx[:, 2]]
            mutant = np.clip(a + self.differential_weight * (b - c), 0.0, 1.0)

            cross = rng.random((n, dim)) < self.crossover_probability
            forced = rng.integers(0, dim, size=n)
            cross[np.arange(n), forced] = True
            trial = np.where(cross, mutant, pop)

            trial_fitness = problem.evaluate(trial)
            if len(trial_fitness) == 0:
                break
            m = len(trial_fitness)
            better = trial_fitness <= fitness[:m]
            pop[:m][better] = trial[:m][better]
            fitness[:m][better] = trial_fitness[better]

            history["evaluations"].append(problem.evaluations)
            history["best"].append(float(problem.best_value))
            history["mean"].append(float(np.mean(fitness[np.isfinite(fitness)])) if np.isfinite(fitness).any() else float("inf"))

        return BaselineResult(
            best_value=float(problem.best_value),
            best_position=problem.best_real(),
            evaluations=problem.evaluations,
            iterations=iteration,
            history=history,
        )


class RandomSearch:
    """Uniform random sampling: the floor every other algorithm must clear."""

    name = "Random"

    def __init__(self, batch: int = 50, seed: int | None = None) -> None:
        self.batch = batch
        self.seed = seed

    def run(self, problem: Problem) -> BaselineResult:
        rng = np.random.default_rng(self.seed)
        problem.reset()
        history: dict[str, list] = {"evaluations": [], "best": [], "mean": []}
        iteration = 0
        while not problem.exhausted():
            iteration += 1
            values = problem.evaluate(rng.random((self.batch, problem.dim)))
            if len(values) == 0:
                break
            history["evaluations"].append(problem.evaluations)
            history["best"].append(float(problem.best_value))
            history["mean"].append(float(np.mean(values[np.isfinite(values)])) if np.isfinite(values).any() else float("inf"))
        return BaselineResult(
            best_value=float(problem.best_value),
            best_position=problem.best_real(),
            evaluations=problem.evaluations,
            iterations=iteration,
            history=history,
        )


ALGORITHMS = {
    "PSO": ParticleSwarm,
    "GA": GeneticAlgorithm,
    "DE": DifferentialEvolution,
    "Random": RandomSearch,
}
