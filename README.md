# ternary-fitness-python

![Python](https://img.shields.io/badge/language-Python%203.10-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![SuperInstance](https://img.shields.io/badge/fleet-SuperInstance-orange)

Fitness evaluation, landscape modeling, and selection pressure for ternary evolutionary systems. Operates on genomes composed of balanced trits {-1, 0, +1}.

## Why This Exists

Evolutionary computation over ternary representations needs its own fitness infrastructure. Binary GA tools don't handle the ternary alphabet; general-purpose optimization libraries don't model Gaussian fitness landscapes over {-1, 0, +1} space or apply selection pressure calibrated for ternary genomes. This library fills that gap in the SuperInstance ecosystem — it evaluates, ranks, and selects ternary genomes so the mutation and crossover operators in `ternary-protocol` have something to act on.

The conservation law γ + η = C constrains the fitness landscape: growth (γ) and dissipation (η) trade off, so selection pressure must balance exploration against exploitation within a fixed resource budget.

## Installation

```bash
pip install ternary-fitness
```

Requires Python ≥ 3.10.

## Usage

### Evaluate genomes with custom fitness functions

```python
from ternary_fitness import FitnessEvaluator

evaluator = FitnessEvaluator()

# Register fitness functions
evaluator.register("ones_count", lambda g: sum(1 for v in g if v == 1))
evaluator.register("balance", lambda g: sum(g) / len(g))

genome = [1, 0, -1, 1, 1, 0, -1, 0]

# Single objective
score = evaluator.evaluate("ones_count", genome)  # 3.0

# Multi-objective
all_scores = evaluator.evaluate_all(genome)
# {"ones_count": 3.0, "balance": 0.125}

# Weighted sum for single metric
combined = FitnessEvaluator.weighted_sum(all_scores, {"ones_count": 0.7, "balance": 0.3})
```

### Build fitness landscapes with Gaussian peaks

```python
from ternary_fitness import FitnessLandscape

landscape = FitnessLandscape(combine="max", baseline=0.1)

# Add peaks at ternary genome coordinates
landscape.add_peak(center=[1, 1, 1, 1], height=1.0, width=2.0)
landscape.add_peak(center=[-1, -1, -1, -1], height=0.8, width=1.5)

genome = [1, 1, 0, 1]
fitness = landscape.fitness(genome)  # max contribution from peaks + baseline

# Per-peak breakdown
profile = landscape.fitness_profile(genome)  # {0: 0.88, 1: 0.0}

# Landscape properties
best_fitness, best_center = landscape.global_optimum()
ruggedness = landscape.ruggedness()  # std of peak heights
```

### Apply selection pressure

```python
from ternary_fitness import SelectionPressure

# Rank-based (pressure 1.0-2.0, where 2.0 = strong selection)
selector = SelectionPressure(method="rank", pressure=1.5)

fitnesses = [0.3, 0.9, 0.5, 0.1, 0.7]
weights = selector.weights(fitnesses)  # normalized, sums to 1.0

# Select indices (roulette wheel)
chosen = selector.select_indices(fitnesses, count=3)

# Also available: "exponential" and "sigma" methods
exp_selector = SelectionPressure(method="exponential", pressure=2.0)
sigma_selector = SelectionPressure(method="sigma", pressure=2.0)
```

### Preserve elites across generations

```python
from ternary_fitness import Elitism

elitism = Elitism(elite_count=2)  # or Elitism(elite_fraction=0.1)

genomes = [[1,1,1], [-1,0,1], [0,0,0], [1,-1,1]]
fitnesses = [0.9, 0.3, 0.5, 0.7]

# Extract top performers
elites = elitism.extract_elites(genomes, fitnesses)
# [([1,1,1], 0.9), ([1,-1,1], 0.7)]

# Replace worst offspring with elites
offspring = [[0,0,1], [1,0,-1], [-1,-1,0], [0,1,0]]
preserved = elitism.apply_elitism(genomes, fitnesses, offspring)
```

### Track population statistics

```python
from ternary_fitness import PopulationFitness

pop = PopulationFitness()
pop.add([1, 1, 1, 1], 0.95)
pop.add([-1, 0, 1, 0], 0.6)
pop.add([0, 0, 0, 0], 0.3)

pop.mean()    # 0.617
pop.std()     # 0.265
pop.median()  # 0.6
pop.best_genome()   # [1, 1, 1, 1]
pop.worst_genome()  # [0, 0, 0, 0]
pop.top_n(2)  # [([1,1,1,1], 0.95), ([-1,0,1,0], 0.6)]
```

## API Reference

### FitnessEvaluator

```python
class FitnessEvaluator:
    def register(self, name: str, fn: Callable[[Sequence[int]], float]) -> None
    def evaluate(self, name: str, genome: Sequence[int]) -> float
    def evaluate_all(self, genome: Sequence[int]) -> dict[str, float]
    def clear_cache(self) -> None
    
    @staticmethod
    def weighted_sum(scores: dict[str, float], weights: dict[str, float]) -> float
```

Raises `ValueError` if any genome element is not in {-1, 0, +1}. Results are cached by (name, genome) tuple.

### FitnessLandscape

```python
@dataclass
class GaussianPeak:
    center: Sequence[int]     # must be ternary
    height: float = 1.0
    width: float = 2.0
    
    def distance(self, genome: Sequence[int]) -> float     # Euclidean
    def contribution(self, genome: Sequence[int]) -> float  # Gaussian kernel

@dataclass
class FitnessLandscape:
    peaks: list[GaussianPeak]
    combine: str = "max"       # "max" or "sum"
    baseline: float = 0.0
    
    def add_peak(self, center, height=1.0, width=2.0) -> None
    def fitness(self, genome: Sequence[int]) -> float
    def fitness_profile(self, genome: Sequence[int]) -> dict[int, float]
    def global_optimum(self) -> tuple[float, Sequence[int]]
    def ruggedness(self) -> float  # std of peak heights
```

### SelectionPressure

```python
class SelectionPressure:
    def __init__(self, method: str = "rank", pressure: float = 1.5)
    # method: "rank" | "exponential" | "sigma"
    
    def weights(self, fitnesses: Sequence[float]) -> list[float]
    def select_indices(self, fitnesses: Sequence[float], count: int) -> list[int]
```

| Method | Formula | `pressure` meaning |
|--------|---------|--------------------|
| `rank` | w = (2−μ)/n + 2(μ−1)rank/(n(n−1)) | μ ∈ [1.0, 2.0], higher = stronger |
| `exponential` | w = exp(μ · f) | μ scales fitness exponent |
| `sigma` | w = max(0, 1 + (f−mean)/(μ·σ)) | μ is sigma scaling factor |

### Elitism

```python
class Elitism:
    def __init__(self, elite_count: int = 1, elite_fraction: float | None = None)
    
    def compute_elite_count(self, population_size: int) -> int
    def extract_elites(self, genomes, fitnesses) -> list[tuple[Sequence[int], float]]
    def apply_elitism(self, genomes, fitnesses, offspring) -> list[Sequence[int]]
```

`elite_fraction` overrides `elite_count` when set.

### PopulationFitness

```python
@dataclass
class PopulationFitness:
    genomes: list[Sequence[int]]
    fitnesses: list[float]
    
    def add(self, genome, fitness) -> None
    def min(self) -> float
    def max(self) -> float
    def mean(self) -> float
    def std(self) -> float
    def median(self) -> float
    def best_genome(self) -> Sequence[int]
    def worst_genome(self) -> Sequence[int]
    def rank(self) -> list[tuple[int, float]]
    def top_n(self, n: int) -> list[tuple[Sequence[int], float]]
```

## Architecture

```
ternary-fitness-python/
├── src/ternary_fitness/
│   ├── __init__.py       # Public API re-exports
│   ├── evaluator.py      # FitnessEvaluator — pluggable scoring + cache
│   ├── landscape.py      # FitnessLandscape, GaussianPeak — terrain model
│   ├── selection.py      # SelectionPressure — rank/exponential/sigma
│   ├── elitism.py        # Elitism — elite preservation
│   ├── population.py     # PopulationFitness — statistics + tracking
│   └── tests/
│       └── test_fitness.py
├── pyproject.toml
└── docs/
```

```
┌──────────────┐     ┌────────────────┐
│  Evaluator   │────→│    Landscape    │
│  (scoring)   │     │  (Gaussian peaks│
└──────┬───────┘     │   over {-1,0,+1})│
       │             └───────┬────────┘
       ↓                     ↓
┌──────────────┐     ┌────────────────┐
│  Population  │────→│   Selection    │
│  (tracking)  │     │   Pressure     │
└──────────────┘     └───────┬────────┘
                             ↓
                     ┌────────────────┐
                     │    Elitism     │
                     │  (preserve top)│
                     └────────────────┘
```

## Related SuperInstance Crates

- **ternary-protocol-python** — Message passing and synchronization for the ternary genomes this library evaluates
- **conservation-spectral-topology** — Conservation law (γ + η = C) verification for the broader ecosystem
- **metal-lathe** — Research wheel that generates hypotheses about ternary fitness landscapes

## License

MIT
