"""Elitism strategy for ternary evolutionary systems."""

from __future__ import annotations
from typing import Sequence


class Elitism:
    """Preserves top-performing genomes across generations.

    Elitism ensures that the best individuals survive unchanged into the next
    generation, preventing fitness regression.
    """

    def __init__(self, elite_count: int = 1, elite_fraction: float | None = None) -> None:
        """Initialize elitism.

        Args:
            elite_count: Fixed number of elites to preserve.
            elite_fraction: If set, preserve this fraction of the population.
                            Overrides elite_count when set.
        """
        if elite_count < 0:
            raise ValueError("elite_count must be non-negative")
        if elite_fraction is not None and not (0.0 <= elite_fraction <= 1.0):
            raise ValueError("elite_fraction must be between 0 and 1")
        self.elite_count = elite_count
        self.elite_fraction = elite_fraction

    def compute_elite_count(self, population_size: int) -> int:
        """Compute how many elites to preserve for a given population size."""
        if self.elite_fraction is not None:
            return max(1, int(population_size * self.elite_fraction))
        return min(self.elite_count, population_size)

    def extract_elites(
        self, genomes: Sequence[Sequence[int]], fitnesses: Sequence[float]
    ) -> list[tuple[Sequence[int], float]]:
        """Extract elite genomes (highest fitness).

        Returns list of (genome, fitness) tuples sorted by fitness descending.
        """
        if len(genomes) != len(fitnesses):
            raise ValueError("genomes and fitnesses must have same length")
        n = self.compute_elite_count(len(genomes))
        indexed = sorted(
            range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True
        )
        return [(genomes[i], fitnesses[i]) for i in indexed[:n]]

    def apply_elitism(
        self,
        genomes: Sequence[Sequence[int]],
        fitnesses: Sequence[float],
        offspring: list[Sequence[int]],
    ) -> list[Sequence[int]]:
        """Replace worst offspring with elite parents.

        Returns new offspring list with elites preserved.
        """
        elites = self.extract_elites(genomes, fitnesses)
        result = list(offspring)
        # Replace from the end (worst offspring positions)
        for i, (genome, _) in enumerate(elites):
            pos = len(result) - 1 - i
            if pos >= 0:
                result[pos] = genome
        return result
