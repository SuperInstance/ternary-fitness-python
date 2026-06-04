"""Fitness evaluation functions for ternary genomes."""

from __future__ import annotations
from typing import Callable, Sequence


class FitnessEvaluator:
    """Evaluates fitness of ternary genomes (sequences of -1, 0, +1).

    Supports pluggable fitness functions, caching, and multi-objective scoring.
    """

    def __init__(self) -> None:
        self._functions: dict[str, Callable[[Sequence[int]], float]] = {}
        self._cache: dict[tuple[str, tuple[int, ...]], float] = {}

    def register(self, name: str, fn: Callable[[Sequence[int]], float]) -> None:
        """Register a named fitness function."""
        if not callable(fn):
            raise TypeError("Fitness function must be callable")
        self._functions[name] = fn

    def evaluate(self, name: str, genome: Sequence[int]) -> float:
        """Evaluate a genome using a named fitness function."""
        if name not in self._functions:
            raise KeyError(f"No fitness function named '{name}'")
        self._validate_genome(genome)
        key = (name, tuple(genome))
        if key not in self._cache:
            self._cache[key] = self._functions[name](genome)
        return self._cache[key]

    def evaluate_all(self, genome: Sequence[int]) -> dict[str, float]:
        """Evaluate genome against all registered functions."""
        return {name: self.evaluate(name, genome) for name in self._functions}

    def clear_cache(self) -> None:
        """Clear the evaluation cache."""
        self._cache.clear()

    @staticmethod
    def _validate_genome(genome: Sequence[int]) -> None:
        """Validate that genome contains only ternary values."""
        for v in genome:
            if v not in (-1, 0, 1):
                raise ValueError(f"Invalid ternary value: {v!r}. Must be -1, 0, or +1.")

    @staticmethod
    def weighted_sum(scores: dict[str, float], weights: dict[str, float]) -> float:
        """Compute weighted sum of multi-objective scores."""
        total = 0.0
        for name, score in scores.items():
            total += weights.get(name, 0.0) * score
        return total
