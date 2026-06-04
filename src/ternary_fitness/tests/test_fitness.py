"""Tests for ternary-fitness package."""

import pytest
from ternary_fitness import (
    FitnessEvaluator,
    FitnessLandscape,
    GaussianPeak,
    PopulationFitness,
    SelectionPressure,
    Elitism,
)


# --- FitnessEvaluator ---

class TestFitnessEvaluator:
    def test_register_and_evaluate(self):
        ev = FitnessEvaluator()
        ev.register("ones", lambda g: sum(v == 1 for v in g))
        assert ev.evaluate("ones", [1, 1, -1, 0]) == 2.0

    def test_invalid_genome_raises(self):
        ev = FitnessEvaluator()
        ev.register("ones", lambda g: sum(g))
        with pytest.raises(ValueError, match="Invalid ternary"):
            ev.evaluate("ones", [1, 2, 0])

    def test_missing_function_raises(self):
        ev = FitnessEvaluator()
        with pytest.raises(KeyError):
            ev.evaluate("nonexistent", [1, 0, -1])

    def test_caching(self):
        call_count = 0
        def counting_fn(g):
            nonlocal call_count
            call_count += 1
            return sum(g)
        ev = FitnessEvaluator()
        ev.register("sum", counting_fn)
        ev.evaluate("sum", [1, 0, -1])
        ev.evaluate("sum", [1, 0, -1])
        assert call_count == 1  # cached

    def test_clear_cache(self):
        ev = FitnessEvaluator()
        ev.register("sum", lambda g: sum(g))
        ev.evaluate("sum", [1, 0, -1])
        ev.clear_cache()
        # After clear, should recompute (no error)
        assert ev.evaluate("sum", [1, 0, -1]) == 0.0

    def test_evaluate_all(self):
        ev = FitnessEvaluator()
        ev.register("a", lambda g: sum(g))
        ev.register("b", lambda g: len(g))
        result = ev.evaluate_all([1, -1, 0])
        assert "a" in result and "b" in result

    def test_weighted_sum(self):
        scores = {"a": 1.0, "b": 2.0}
        weights = {"a": 0.5, "b": 0.5}
        assert FitnessEvaluator.weighted_sum(scores, weights) == 1.5

    def test_register_non_callable_raises(self):
        ev = FitnessEvaluator()
        with pytest.raises(TypeError):
            ev.register("bad", "not_a_function")


# --- FitnessLandscape / GaussianPeak ---

class TestGaussianPeak:
    def test_perfect_match(self):
        p = GaussianPeak(center=[1, 0, -1], height=2.0, width=1.0)
        assert p.contribution([1, 0, -1]) == pytest.approx(2.0)

    def test_distance(self):
        p = GaussianPeak(center=[1, 1, 1])
        assert p.distance([0, 0, 0]) == pytest.approx(3**0.5)

    def test_invalid_center(self):
        with pytest.raises(ValueError):
            GaussianPeak(center=[1, 2, 0])

    def test_length_mismatch(self):
        p = GaussianPeak(center=[1, 0])
        with pytest.raises(ValueError):
            p.distance([1, 0, -1])


class TestFitnessLandscape:
    def test_empty_landscape_returns_baseline(self):
        fl = FitnessLandscape(baseline=0.5)
        assert fl.fitness([1, 0]) == 0.5

    def test_single_peak(self):
        fl = FitnessLandscape()
        fl.add_peak([1, 1], height=3.0, width=1.0)
        assert fl.fitness([1, 1]) == pytest.approx(3.0)

    def test_max_combine(self):
        fl = FitnessLandscape(combine="max")
        fl.add_peak([1, 1], height=2.0)
        fl.add_peak([-1, -1], height=5.0)
        assert fl.fitness([-1, -1]) > fl.fitness([1, 1])

    def test_sum_combine(self):
        fl = FitnessLandscape(combine="sum")
        fl.add_peak([1, 1], height=1.0, width=100.0)
        fl.add_peak([-1, -1], height=1.0, width=100.0)
        # Both peaks contribute at [0, 0]
        f = fl.fitness([0, 0])
        assert f > 1.0  # sum of two contributions

    def test_global_optimum(self):
        fl = FitnessLandscape()
        fl.add_peak([1, 1], height=2.0)
        fl.add_peak([-1, -1], height=5.0)
        opt_fitness, opt_center = fl.global_optimum()
        assert opt_fitness == pytest.approx(5.0)
        assert list(opt_center) == [-1, -1]

    def test_ruggedness_single_peak(self):
        fl = FitnessLandscape()
        fl.add_peak([1], height=1.0)
        assert fl.ruggedness() == 0.0

    def test_fitness_profile(self):
        fl = FitnessLandscape()
        fl.add_peak([1, 1], height=2.0, width=1.0)
        fl.add_peak([-1, -1], height=3.0, width=1.0)
        profile = fl.fitness_profile([0, 0])
        assert len(profile) == 2


# --- PopulationFitness ---

class TestPopulationFitness:
    def test_empty_raises(self):
        pf = PopulationFitness()
        with pytest.raises(ValueError):
            pf.mean()

    def test_min_max_mean(self):
        pf = PopulationFitness()
        pf.add([1, 1], 3.0)
        pf.add([0, 0], 1.0)
        pf.add([-1, -1], 5.0)
        assert pf.min() == 1.0
        assert pf.max() == 5.0
        assert pf.mean() == pytest.approx(3.0)

    def test_std(self):
        pf = PopulationFitness()
        pf.add([1], 2.0)
        pf.add([0], 4.0)
        assert pf.std() == pytest.approx(1.0)

    def test_median_odd(self):
        pf = PopulationFitness()
        pf.add([1], 1.0)
        pf.add([0], 3.0)
        pf.add([-1], 5.0)
        assert pf.median() == 3.0

    def test_median_even(self):
        pf = PopulationFitness()
        pf.add([1], 1.0)
        pf.add([0], 2.0)
        pf.add([-1], 3.0)
        pf.add([0], 4.0)
        assert pf.median() == pytest.approx(2.5)

    def test_best_and_worst_genome(self):
        pf = PopulationFitness()
        pf.add([1, 1], 3.0)
        pf.add([-1, -1], 0.5)
        assert list(pf.best_genome()) == [1, 1]
        assert list(pf.worst_genome()) == [-1, -1]

    def test_top_n(self):
        pf = PopulationFitness()
        pf.add([1], 5.0)
        pf.add([0], 3.0)
        pf.add([-1], 1.0)
        top = pf.top_n(2)
        assert len(top) == 2
        assert top[0][1] == 5.0

    def test_summary(self):
        pf = PopulationFitness()
        pf.add([1], 2.0)
        s = pf.summary()
        assert s["size"] == 1
        assert s["min"] == s["max"] == 2.0


# --- SelectionPressure ---

class TestSelectionPressure:
    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            SelectionPressure(method="invalid")

    def test_rank_weights_sum_to_one(self):
        sp = SelectionPressure(method="rank", pressure=1.5)
        w = sp.weights([1.0, 2.0, 3.0])
        assert sum(w) == pytest.approx(1.0)

    def test_exponential_weights(self):
        sp = SelectionPressure(method="exponential", pressure=1.0)
        w = sp.weights([1.0, 2.0, 3.0])
        assert sum(w) == pytest.approx(1.0)
        # Higher fitness should have higher weight
        assert w[2] > w[1] > w[0]

    def test_sigma_weights(self):
        sp = SelectionPressure(method="sigma", pressure=2.0)
        w = sp.weights([1.0, 2.0, 3.0])
        assert sum(w) == pytest.approx(1.0)

    def test_empty_returns_empty(self):
        sp = SelectionPressure()
        assert sp.weights([]) == []

    def test_single_returns_one(self):
        sp = SelectionPressure()
        assert sp.weights([5.0]) == [1.0]

    def test_select_indices_count(self):
        sp = SelectionPressure(method="rank")
        indices = sp.select_indices([1.0, 2.0, 3.0], count=5)
        assert len(indices) == 5
        assert all(0 <= i < 3 for i in indices)


# --- Elitism ---

class TestElitism:
    def test_extract_elites(self):
        e = Elitism(elite_count=2)
        elites = e.extract_elites([[1, 1], [0, 0], [-1, -1]], [1.0, 5.0, 3.0])
        assert len(elites) == 2
        assert elites[0][1] == 5.0  # best first
        assert elites[1][1] == 3.0

    def test_elite_fraction(self):
        e = Elitism(elite_fraction=0.5)
        assert e.compute_elite_count(10) == 5

    def test_elite_fraction_minimum_one(self):
        e = Elitism(elite_fraction=0.01)
        assert e.compute_elite_count(10) == 1

    def test_negative_count_raises(self):
        with pytest.raises(ValueError):
            Elitism(elite_count=-1)

    def test_invalid_fraction_raises(self):
        with pytest.raises(ValueError):
            Elitism(elite_fraction=1.5)

    def test_apply_elitism(self):
        e = Elitism(elite_count=1)
        offspring = [[0, 0], [0, 0], [0, 0]]
        result = e.apply_elitism([[1, 1]], [10.0], offspring)
        assert result[-1] == [1, 1]  # worst replaced by elite

    def test_mismatched_lengths_raises(self):
        e = Elitism()
        with pytest.raises(ValueError):
            e.extract_elites([[1]], [1.0, 2.0])
