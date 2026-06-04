"""Fitness landscape with Gaussian peaks over ternary genome space."""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class GaussianPeak:
    """A single Gaussian peak in the fitness landscape.

    The peak is centered on a ternary genome with a given height and width.
    """

    center: Sequence[int]
    height: float = 1.0
    width: float = 2.0

    def __post_init__(self) -> None:
        for v in self.center:
            if v not in (-1, 0, 1):
                raise ValueError(f"Center must be ternary, got {v!r}")

    def distance(self, genome: Sequence[int]) -> float:
        """Euclidean distance from genome to peak center."""
        if len(genome) != len(self.center):
            raise ValueError("Genome length must match center length")
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(genome, self.center)))

    def contribution(self, genome: Sequence[int]) -> float:
        """Gaussian contribution to fitness for a given genome."""
        d = self.distance(genome)
        return self.height * math.exp(-0.5 * (d / self.width) ** 2)


@dataclass
class FitnessLandscape:
    """A fitness landscape composed of multiple Gaussian peaks.

    The overall fitness at any point is the max contribution from all peaks
    (or the sum, configurable).
    """

    peaks: list[GaussianPeak] = field(default_factory=list)
    combine: str = "max"  # "max" or "sum"
    baseline: float = 0.0

    def add_peak(self, center: Sequence[int], height: float = 1.0, width: float = 2.0) -> None:
        """Add a Gaussian peak to the landscape."""
        self.peaks.append(GaussianPeak(center=center, height=height, width=width))

    def fitness(self, genome: Sequence[int]) -> float:
        """Compute fitness of a genome in this landscape."""
        if not self.peaks:
            return self.baseline
        contributions = [p.contribution(genome) for p in self.peaks]
        if self.combine == "sum":
            return self.baseline + sum(contributions)
        return self.baseline + max(contributions)

    def fitness_profile(self, genome: Sequence[int]) -> dict[int, float]:
        """Get per-peak contributions for a genome."""
        return {i: p.contribution(genome) for i, p in enumerate(self.peaks)}

    def global_optimum(self) -> tuple[float, Sequence[int]]:
        """Return the highest peak's fitness and center."""
        if not self.peaks:
            return (self.baseline, [])
        best = max(self.peaks, key=lambda p: p.height)
        return (self.baseline + best.height, best.center)

    def ruggedness(self) -> float:
        """Estimate landscape ruggedness as std of peak heights."""
        if len(self.peaks) < 2:
            return 0.0
        heights = [p.height for p in self.peaks]
        mean_h = sum(heights) / len(heights)
        variance = sum((h - mean_h) ** 2 for h in heights) / len(heights)
        return math.sqrt(variance)
