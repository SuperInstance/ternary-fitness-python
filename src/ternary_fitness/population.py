"""Population-level fitness statistics and tracking."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence
import math


@dataclass
class PopulationFitness:
    """Tracks and computes fitness statistics for a population of ternary genomes."""

    genomes: list[Sequence[int]] = field(default_factory=list)
    fitnesses: list[float] = field(default_factory=list)

    def add(self, genome: Sequence[int], fitness: float) -> None:
        """Add a genome and its fitness to the population."""
        self.genomes.append(genome)
        self.fitnesses.append(fitness)

    def min(self) -> float:
        """Minimum fitness in population."""
        if not self.fitnesses:
            raise ValueError("Population is empty")
        return min(self.fitnesses)

    def max(self) -> float:
        """Maximum fitness in population."""
        if not self.fitnesses:
            raise ValueError("Population is empty")
        return max(self.fitnesses)

    def mean(self) -> float:
        """Mean fitness in population."""
        if not self.fitnesses:
            raise ValueError("Population is empty")
        return sum(self.fitnesses) / len(self.fitnesses)

    def std(self) -> float:
        """Standard deviation of fitness."""
        if len(self.fitnesses) < 2:
            return 0.0
        m = self.mean()
        variance = sum((f - m) ** 2 for f in self.fitnesses) / len(self.fitnesses)
        return math.sqrt(variance)

    def median(self) -> float:
        """Median fitness."""
        if not self.fitnesses:
            raise ValueError("Population is empty")
        sorted_f = sorted(self.fitnesses)
        n = len(sorted_f)
        if n % 2 == 1:
            return sorted_f[n // 2]
        return (sorted_f[n // 2 - 1] + sorted_f[n // 2]) / 2

    def best_genome(self) -> Sequence[int]:
        """Return the genome with highest fitness."""
        if not self.fitnesses:
            raise ValueError("Population is empty")
        idx = self.fitnesses.index(max(self.fitnesses))
        return self.genomes[idx]

    def worst_genome(self) -> Sequence[int]:
        """Return the genome with lowest fitness."""
        if not self.fitnesses:
            raise ValueError("Population is empty")
        idx = self.fitnesses.index(min(self.fitnesses))
        return self.genomes[idx]

    def rank(self) -> list[tuple[int, float]]:
        """Return indices sorted by fitness descending."""
        indexed = [(i, f) for i, f in enumerate(self.fitnesses)]
        return sorted(indexed, key=lambda x: x[1], reverse=True)

    def top_n(self, n: int) -> list[tuple[Sequence[int], float]]:
        """Return top N genomes with their fitness."""
        ranked = self.rank()
        return [(self.genomes[i], f) for i, f in ranked[:n]]

    def summary(self) -> dict[str, float]:
        """Return summary statistics."""
        return {
            "size": len(self.fitnesses),
            "min": self.min(),
            "max": self.max(),
            "mean": self.mean(),
            "std": self.std(),
            "median": self.median(),
        }

    @property
    def size(self) -> int:
        return len(self.fitnesses)
