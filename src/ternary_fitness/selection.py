"""Selection pressure methods for ternary evolutionary systems."""

from __future__ import annotations
import math
from typing import Sequence


class SelectionPressure:
    """Applies selection pressure to rank and select genomes.

    Supports rank-based, exponential, and sigma-scaling methods.
    """

    def __init__(self, method: str = "rank", pressure: float = 1.5) -> None:
        """Initialize selection pressure.

        Args:
            method: One of 'rank', 'exponential', or 'sigma'.
            pressure: Selection pressure parameter (interpretation varies by method).
        """
        if method not in ("rank", "exponential", "sigma"):
            raise ValueError(f"Unknown method: {method!r}")
        self.method = method
        self.pressure = pressure

    def weights(self, fitnesses: Sequence[float]) -> list[float]:
        """Compute selection weights from fitness values.

        Higher fitness = higher weight. Returns normalized weights summing to 1.0.
        """
        if not fitnesses:
            return []
        if len(fitnesses) == 1:
            return [1.0]

        if self.method == "rank":
            return self._rank_weights(fitnesses)
        elif self.method == "exponential":
            return self._exponential_weights(fitnesses)
        else:
            return self._sigma_weights(fitnesses)

    def _rank_weights(self, fitnesses: Sequence[float]) -> list[float]:
        """Rank-based selection: weight proportional to rank position."""
        n = len(fitnesses)
        ranked = sorted(range(n), key=lambda i: fitnesses[i])
        mu = self.pressure
        raw = []
        for rank_pos in range(n):
            idx = ranked[rank_pos]
            w = (2 - mu) / n + (2 * (mu - 1) * rank_pos) / (n * (n - 1))
            raw.append((idx, w))
        raw.sort(key=lambda x: x[0])
        weights = [w for _, w in raw]
        total = sum(weights)
        return [w / total for w in weights] if total > 0 else [1.0 / n] * n

    def _exponential_weights(self, fitnesses: Sequence[float]) -> list[float]:
        """Exponential selection: weight = exp(pressure * fitness)."""
        raw = [math.exp(self.pressure * f) for f in fitnesses]
        total = sum(raw)
        return [r / total for r in raw] if total > 0 else [1.0 / len(fitnesses)] * len(fitnesses)

    def _sigma_weights(self, fitnesses: Sequence[float]) -> list[float]:
        """Sigma scaling: weight based on standard deviations from mean."""
        n = len(fitnesses)
        mean_f = sum(fitnesses) / n
        std_f = math.sqrt(sum((f - mean_f) ** 2 for f in fitnesses) / n) if n > 1 else 1.0
        if std_f == 0:
            return [1.0 / n] * n
        c = self.pressure
        raw = [max(0.0, 1.0 + (f - mean_f) / (c * std_f)) for f in fitnesses]
        total = sum(raw)
        return [r / total for r in raw] if total > 0 else [1.0 / n] * n

    def select_indices(self, fitnesses: Sequence[float], count: int) -> list[int]:
        """Select indices based on weighted probabilities (roulette wheel)."""
        import random
        w = self.weights(fitnesses)
        total = sum(w)
        if total == 0:
            return random.choices(range(len(fitnesses)), k=count)
        # Weighted selection
        cumulative = []
        s = 0.0
        for weight in w:
            s += weight
            cumulative.append(s)
        result = []
        for _ in range(count):
            r = random.random() * total
            for i, c in enumerate(cumulative):
                if r <= c:
                    result.append(i)
                    break
            else:
                result.append(len(w) - 1)
        return result
