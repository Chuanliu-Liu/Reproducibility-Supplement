import itertools
import math
import unittest
from unittest.mock import patch
from fractions import Fraction as F
import numpy as np
from disclosure.model import Model, Release, latent_count_model
from disclosure.compiled import Compiled
from disclosure.exact_tree import ExactProductTree, exp_minus_bounds, integer_inverse, transform
from disclosure.baselines import Enumerated
from disclosure.lineage import LineageWMC
from disclosure.estimate import path,PathBudgetExceeded


class ProductTree(unittest.TestCase):
    def test_budget_cut_preserves_prefix_without_counting_partial_sample(self):
        model = latent_count_model(5, 1, 41)
        release = Release((2.2,), (1.,))
        engine = Enumerated(model, release)
        world = np.ones(5,dtype=int)
        with patch('disclosure.estimate.time.perf_counter',side_effect=[0.,2.]):
            with self.assertRaises(PathBudgetExceeded) as caught:
                path(engine,world,[1,2,3,4],deadline=1.)
        partial = caught.exception.partial
        self.assertEqual(partial['next_position'],1)
        self.assertEqual(partial['evidence'],{1:1})
        self.assertEqual(partial['world'],world.tolist())
        self.assertAlmostEqual(sum(partial['increments']),partial['initial_risk']-partial['risk_before_next'])

    def test_weighted_lineage_same_game_all_prefixes(self):
        model = latent_count_model(5, 2, 41)
        release = Release((1.8, .8), (.5, .5))
        engine, reference = LineageWMC(model, release), Enumerated(model, release)
        for states in itertools.product([-1, 0, 1], repeat=4):
            engine.reset();reference.reset()
            for i, value in enumerate(states, 1):
                if value >= 0:engine.update(i, value);reference.update(i, value)
            np.testing.assert_allclose(engine.posterior_target(), reference.posterior_target(), atol=1e-12)

    def test_exact_rational_prefixes_and_balanced_update_bound(self):
        model = latent_count_model(6, 2, 531)
        release = Release((3, 2), (1, 1), kind='exact')
        cached = ExactProductTree(model, release)
        fresh = ExactProductTree(model, release, cache=False)
        for states in itertools.product([-1, 0, 1], repeat=5):
            cached.reset(); fresh.reset()
            for owner, value in enumerate(states, 1):
                if value < 0:continue
                before = cached.counters['evaluations']
                cached.update(owner, value);fresh.update(owner, value)
                self.assertLessEqual(cached.counters['evaluations']-before, math.ceil(math.log2(6))+1)
                self.assertEqual(cached.root.table, fresh.root.table)
            masses = [F(0), F(0)]
            for world in itertools.product([0, 1], repeat=6):
                if any(v >= 0 and world[i] != v for i, v in enumerate(states, 1)):
                    continue
                if tuple(model.counts(world)) != (3, 2):continue
                for h in range(len(model.mixture)):
                    weight = F(float(model.mixture[h]))
                    for i, value in enumerate(world):weight *= F(float(model.probs[i][h, value]))
                    masses[world[0]] += weight
            if sum(masses):
                risk = 1-max(masses)/sum(masses)
                self.assertEqual(cached.risk_bounds(), (risk, risk))

    def test_bijection_all_signed_multicategory_attainable_sums(self):
        matrices = [((1, 0), (-1, 1)), ((1, -1), (0, 1))]
        contributions = [[[0, -1], [1, 0], [2, 2]], [[-2, 1], [0, 0]], [[1, 1], [3, 0]]]
        support = {tuple(sum(v[d] for v in world) for d in range(2))
                   for world in itertools.product(*contributions)}
        for matrix in matrices:
            inverse = integer_inverse(matrix)
            mapped = {transform(matrix, s) for s in support}
            self.assertEqual(len(mapped), len(support))
            self.assertEqual({transform(inverse, s) for s in mapped}, support)
        with self.assertRaises(ValueError):integer_inverse(((1, 1), (1, 1)))
        # A singular map makes different likelihood inputs indistinguishable.
        self.assertEqual(transform(((1, 1), (1, 1)), (1, 0)), transform(((1, 1), (1, 1)), (0, 1)))

    def test_certified_laplace_risk_and_original_coordinate_likelihood(self):
        model = latent_count_model(5, 2, 41)
        release = Release((1.8, .8), (.5, .5))
        reference = Enumerated(model, release)
        for matrix in [((1, 0), (0, 1)), ((1, 0), (-1, 1))]:
            exact = ExactProductTree(model, release, matrix=matrix)
            for states in itertools.product([-1, 0, 1], repeat=4):
                exact.reset();reference.reset()
                for i, v in enumerate(states, 1):
                    if v >= 0:exact.update(i, v);reference.update(i, v)
                for loss in ['zero_one', 'brier']:
                    lower, upper, _ = exact.adaptive_risk_bounds(loss)
                    risk = F(reference.risk(loss))
                    self.assertLessEqual(float(upper-lower), 1e-12)
                    self.assertLessEqual(max(float(lower-risk), float(risk-upper), 0), 1e-14)

    def test_exp_rational_enclosure_against_exact_known_bounds(self):
        lo, hi = exp_minus_bounds(F(0))
        self.assertEqual((lo, hi), (F(1), F(1)))
        for x in [F(1, 7), F(1), F(10), F(1000)]:
            low, high = exp_minus_bounds(x)
            self.assertLessEqual(low, high)
            self.assertGreaterEqual(low, 0)
            self.assertLessEqual(high, 1)
            self.assertLessEqual(float(low)-math.exp(-float(x)), 1e-15)
            self.assertLessEqual(math.exp(-float(x))-float(high), 1e-15)

    def test_safe_separation_current_prefix_full_release(self):
        probabilities = [np.array([[.4, .6]]) for _ in range(4)]
        contributions = [np.array([[0], [1]]), np.array([[0], [1]]),
                         np.array([[0], [0]]), np.array([[0], [0]])]
        model = Model(probabilities, np.array([1.]), contributions)
        release = Release((.9,), (.5,))
        engine = Compiled(model, release);reference = Enumerated(model, release)
        self.assertFalse(engine.pruning_certificate(1)['safe_zero_current_state'])
        for states in itertools.product([-1, 0, 1], repeat=3):
            engine.reset();reference.reset()
            for i, value in enumerate(states, 1):
                if value >= 0:engine.update(i, value);reference.update(i, value)
            for owner in range(1, 4):
                if owner in engine.evidence:continue
                cert = engine.pruning_certificate(owner)
                if cert['safe_zero_current_state']:
                    before = reference.risk()
                    for value in [0, 1]:
                        reference.evidence = dict(engine.evidence)
                        reference.update(owner, value)
                        self.assertAlmostEqual(reference.risk(), before, places=12)
                        reference.evidence = dict(engine.evidence)


if __name__ == '__main__':unittest.main()
